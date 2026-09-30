"""Stage 11: validation-only comparison of additional model families.

The full Stage 10 decision is a mandatory input. All full-data candidates use
the retained feature set and class-weight treatment. The held-out test split is
counted in the frozen manifest but is never materialised or evaluated.
"""

# %% Imports and runtime configuration
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_modeling_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURE_REPO_PATH,
    PROJECT_DIR,
    TARGET_COLUMN,
    best_f1_threshold,
    classification_metrics,
    configure_mlflow,
    dataset_metadata,
    load_modeling_table,
    make_optional_sample,
    predict_scores,
)
from model_family_comparison_utils import (
    DEFAULT_RANDOM_STATE,
    FROZEN_FULL_SPLIT_HASH,
    ModelFamilyConfig,
    build_model_pipeline,
    frozen_model_family_configs,
    select_model_family,
    stratified_sample_indices,
    validate_stage_10_handoff,
)
from oversampling_experiment_utils import FEATURE_SET_NAME, MODEL_FEATURES
from predictive_quality_utils import TEST_CODE, build_split_manifest, make_stratified_random_partition
from sender_location_history_utils import build_development_frames, fixed_workload_table, restore_raw_source_order

# %% Frozen experiment definition
RAW_DATA_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
DEFAULT_EXPERIMENT = "financial-fraud-stage-11-model-family-comparison"
STAGE_10_SUMMARY_RUN_ID = "7d1e57bd9c0a49a59e0898bb3c7bad21"
STAGE_10_SUMMARY_ARTIFACT = "oversampling_full_data_summary.json"
DEFAULT_SMOKE_ROWS = 50_000
FULL_DATA_ROWS = 5_000_000
KNN_TRAIN_ROWS = 100_000
KNN_VALIDATION_ROWS = 25_000


# %% Command-line interface
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen model families using validation evidence only. " "Smoke mode cannot select a model."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-source", choices=("feast", "csv"), default="feast")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--feature-repo-path", type=Path, default=DEFAULT_FEATURE_REPO_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--stage-10-summary-run-id", default=STAGE_10_SUMMARY_RUN_ID)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


# %% Stage 10 handoff and reproducibility helpers
def load_stage_10_handoff(run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Download and validate the frozen full-data Stage 10 summary."""

    if not run_id.strip():
        raise ValueError("A Stage 10 summary run ID is required.")
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    if run.data.tags.get("run_role") != "full_data_summary":
        raise ValueError("The supplied Stage 10 run is not a full-data summary run.")
    local_path = Path(client.download_artifacts(run_id, STAGE_10_SUMMARY_ARTIFACT))
    summary = json.loads(local_path.read_text(encoding="utf-8"))
    summary["summary_run_id"] = run_id
    handoff = validate_stage_10_handoff(summary)
    if handoff["selected_strategy"] != "class_weight_reference":
        raise ValueError("This frozen Stage 11 implementation expects the recorded Stage 10 " "class-weight decision.")
    return summary, handoff


def tracked_source_files() -> list[Path]:
    return [
        PROJECT_DIR / "11_model_family_comparison_with_mlflow.py",
        PROJECT_DIR / "model_family_comparison_utils.py",
        PROJECT_DIR / "sender_location_history_utils.py",
        PROJECT_DIR / "oversampling_experiment_utils.py",
        PROJECT_DIR / "fraud_modeling_utils.py",
        PROJECT_DIR / "predictive_quality_utils.py",
        PROJECT_DIR / "project_io_utils.py",
        PROJECT_DIR / "requirements.txt",
        PROJECT_DIR / "Markdown files" / "11_model_family_comparison_plan.md",
        PROJECT_DIR / "Markdown files" / "11_model_family_comparison_implementation_guide.md",
        PROJECT_DIR / "Markdown files" / "11_model_family_comparison_results.md",
    ]


def library_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("scikit-learn", "lightgbm", "xgboost", "catboost", "mlflow", "pandas", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def positive_weight(labels: pd.Series | np.ndarray) -> float:
    values = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("Training labels must be a one-dimensional binary array.")
    fraud = int(values.sum())
    legitimate = int(len(values) - fraud)
    if fraud == 0 or legitimate == 0:
        raise ValueError("Both training classes are required.")
    return legitimate / fraud


# %% One controlled model run
def run_model(
    *,
    config: ModelFamilyConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    split_manifest: dict[str, Any],
    handoff: dict[str, str],
    evidence_scope: str,
    model_role: str,
    class_weight_ratio: float,
    dataset_info: dict[str, Any],
    feature_lineage: dict[str, Any],
) -> dict[str, Any]:
    """Fit and evaluate one model without accepting a test frame."""

    pipeline = build_model_pipeline(config, positive_class_weight=class_weight_ratio, random_state=DEFAULT_RANDOM_STATE)
    x_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    x_validation = validation_df[MODEL_FEATURES]
    y_validation = validation_df[TARGET_COLUMN]

    fit_start = perf_counter()
    pipeline.fit(x_train, y_train)
    training_seconds = perf_counter() - fit_start

    inference_start = perf_counter()
    validation_scores, _, score_type = predict_scores(pipeline, x_validation)
    inference_seconds = perf_counter() - inference_start
    if len(validation_scores) != len(validation_df) or not np.isfinite(validation_scores).all():
        raise ValueError(f"{config.name} produced invalid validation scores.")

    threshold, _ = best_f1_threshold(y_validation, validation_scores)
    threshold_metrics = classification_metrics(y_validation, validation_scores, threshold)
    workloads = fixed_workload_table(y_validation, validation_scores, validation_df["amount"])
    workload_five = workloads.loc[np.isclose(workloads["requested_alert_rate"], 0.05)].iloc[0]
    try:
        transformed_dimensions = int(pipeline.named_steps["preprocessing"].get_feature_names_out().size)
    except (AttributeError, ValueError):
        transformed_dimensions = -1

    result = {
        "model": config.name,
        "family": config.family,
        "model_role": model_role,
        "full_data_candidate": bool(config.full_data_candidate),
        "feasible": True,
        "run_id": "pending",
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "transformed_feature_dimensions": transformed_dimensions,
        "validation_average_precision": float(average_precision_score(y_validation, validation_scores)),
        "validation_roc_auc": float(roc_auc_score(y_validation, validation_scores)),
        "validation_precision_at_max_f1": float(threshold_metrics["precision"]),
        "validation_recall_at_max_f1": float(threshold_metrics["recall"]),
        "validation_f1": float(threshold_metrics["f1"]),
        "workload_5pct_fraud_count_recall": float(workload_five["fraud_count_recall"]),
        "workload_5pct_fraud_value_recall": float(workload_five["fraud_value_recall"]),
        "training_seconds": float(training_seconds),
        "validation_inference_seconds": float(inference_seconds),
    }

    with mlflow.start_run(run_name=f"model_family_{evidence_scope}__{config.name}") as run:
        mlflow.set_tags(
            {
                "stage": "11_model_family_comparison",
                "run_role": model_role,
                "evidence_scope": evidence_scope,
                "test_split_evaluated": "false",
                "eligible_for_model_selection": str(evidence_scope == "full_data" and config.full_data_candidate).lower(),
            }
        )
        mlflow.log_params(
            {
                "model": config.name,
                "family": config.family,
                "preprocessing": config.preprocessing,
                "feature_set": FEATURE_SET_NAME,
                "imbalance_strategy": handoff["selected_strategy"],
                "stage_10_summary_run_id": handoff["summary_run_id"],
                "split_assignment_sha256": split_manifest["split_assignment_sha256"],
                "score_type": score_type,
                "class_weight_ratio": class_weight_ratio,
                "random_state": DEFAULT_RANDOM_STATE,
            }
        )
        mlflow.log_metrics(
            {
                key: value
                for key, value in result.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        mlflow.log_dict(asdict(config), "model_configuration.json")
        mlflow.log_dict(split_manifest, "split_manifest.json")
        mlflow.log_dict(dataset_info, "dataset_info.json")
        mlflow.log_dict(feature_lineage, "feature_lineage.json")
        mlflow.log_dict(library_versions(), "library_versions.json")
        mlflow.log_text(workloads.to_csv(index=False), "validation_workloads.csv")
        for source in tracked_source_files():
            mlflow.log_artifact(str(source), artifact_path="source_snapshot")
        result["run_id"] = run.info.run_id
        mlflow.log_dict(result, "model_result.json")
    return result


def log_memory_exclusion(
    config: ModelFamilyConfig, *, evidence_scope: str, model_role: str, error: MemoryError) -> dict[str, Any]:
    """Record a measured memory exclusion instead of substituting another model."""

    exclusion = {
        "model": config.name,
        "family": config.family,
        "model_role": model_role,
        "evidence_scope": evidence_scope,
        "reason": "memory_error",
        "message": str(error) or "Python raised MemoryError.",
        "eligible_for_selection": False,
    }
    with mlflow.start_run(run_name=f"model_family_{evidence_scope}__{config.name}__excluded") as run:
        mlflow.set_tags(
            {
                "stage": "11_model_family_comparison",
                "run_role": "feasibility_exclusion",
                "evidence_scope": evidence_scope,
                "test_split_evaluated": "false",
                "eligible_for_model_selection": "false",
            }
        )
        exclusion["run_id"] = run.info.run_id
        mlflow.log_dict(exclusion, "feasibility_exclusion.json")
    return exclusion


# %% Experiment orchestration
def main() -> None:
    args = parse_args()
    if args.random_state != DEFAULT_RANDOM_STATE:
        raise ValueError("Stage 11 is frozen to random state 42.")
    is_smoke = args.mode == "smoke"
    evidence_scope = "smoke_test" if is_smoke else "full_data"

    tracking_uri = configure_mlflow(args.experiment_name)
    stage_10_summary, handoff = load_stage_10_handoff(args.stage_10_summary_run_id)
    modeling_table, dataset_info, feature_lineage = load_modeling_table(
        data_source=args.data_source,
        data_path=args.data_path,
        feature_repo_path=args.feature_repo_path,
        calculate_hash=not is_smoke,
    )
    source_rows = len(modeling_table)
    raw_info = dataset_metadata(args.raw_data_path, calculate_hash=not is_smoke)
    if is_smoke:
        modeling_table = make_optional_sample(modeling_table, DEFAULT_SMOKE_ROWS, DEFAULT_RANDOM_STATE)
    elif source_rows != FULL_DATA_ROWS:
        raise ValueError("The Stage 11 full run requires exactly 5,000,000 rows.")

    modeling_table = restore_raw_source_order(modeling_table, args.raw_data_path)
    partition = make_stratified_random_partition(
        modeling_table[TARGET_COLUMN], train_fraction=0.70, validation_fraction=0.15, seed=DEFAULT_RANDOM_STATE
    )
    split_manifest = build_split_manifest(
        modeling_table, partition, seed=DEFAULT_RANDOM_STATE, train_fraction=0.70, validation_fraction=0.15
    )
    if not is_smoke and split_manifest["split_assignment_sha256"] != FROZEN_FULL_SPLIT_HASH:
        raise ValueError("The Stage 11 full run does not match the frozen split hash.")
    if not is_smoke and split_manifest["split_assignment_sha256"] != handoff["split_assignment_sha256"]:
        raise ValueError("Stage 11 and the Stage 10 handoff use different splits.")

    split_rows = split_manifest["splits"]
    print(
        "Running the Stage 11 validation-only model-family "
        f"{'smoke workflow' if is_smoke else 'full-data comparison'}."
    )
    print(
        "Frozen stratified random split: "
        f"train={split_rows['train']['rows']:,}, "
        f"validation={split_rows['validation']['rows']:,}, "
        f"test={split_rows['test']['rows']:,}."
    )
    print("The test split will not be materialised or evaluated.")
    print("Frozen Stage 10 treatment: class_weight_reference.")

    train_df, validation_df = build_development_frames(modeling_table, partition, args.raw_data_path)
    if int((partition == TEST_CODE).sum()) != split_rows["test"]["rows"]:
        raise ValueError("The held-out test row count changed unexpectedly.")
    del modeling_table, partition
    gc.collect()

    class_weight_ratio = positive_weight(train_df[TARGET_COLUMN])
    results: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    configs = frozen_model_family_configs()
    for config in configs:
        if not config.full_data_candidate:
            continue
        try:
            result = run_model(
                config=config,
                train_df=train_df,
                validation_df=validation_df,
                split_manifest=split_manifest,
                handoff=handoff,
                evidence_scope=evidence_scope,
                model_role="full_data_candidate",
                class_weight_ratio=class_weight_ratio,
                dataset_info=dataset_info,
                feature_lineage=feature_lineage,
            )
        except MemoryError as error:
            if config.name == "random_forest_reference" or is_smoke:
                raise
            exclusion = log_memory_exclusion(config, evidence_scope=evidence_scope, model_role="full_data_candidate", error=error)
            exclusions.append(exclusion)
            print(f"{config.name}: excluded after a measured memory failure.")
            gc.collect()
            continue
        results.append(result)
        print(
            f"{config.name}: validation Average Precision="
            f"{result['validation_average_precision']:.6f}, run={result['run_id']}"
        )
        gc.collect()

    knn_config = next(config for config in configs if not config.full_data_candidate)
    knn_train_count = min(KNN_TRAIN_ROWS, len(train_df))
    knn_validation_count = min(KNN_VALIDATION_ROWS, len(validation_df))
    train_positions = stratified_sample_indices(train_df[TARGET_COLUMN], knn_train_count)
    validation_positions = stratified_sample_indices(validation_df[TARGET_COLUMN], knn_validation_count)
    knn_train = train_df.iloc[train_positions].reset_index(drop=True)
    knn_validation = validation_df.iloc[validation_positions].reset_index(drop=True)
    try:
        knn_result = run_model(
            config=knn_config,
            train_df=knn_train,
            validation_df=knn_validation,
            split_manifest=split_manifest,
            handoff=handoff,
            evidence_scope=evidence_scope,
            model_role="sampled_feasibility_only",
            class_weight_ratio=positive_weight(knn_train[TARGET_COLUMN]),
            dataset_info=dataset_info,
            feature_lineage=feature_lineage,
        )
    except MemoryError as error:
        if is_smoke:
            raise
        exclusions.append(
            log_memory_exclusion(
                knn_config, evidence_scope=evidence_scope, model_role="sampled_feasibility_only", error=error
            )
        )
        print("knn_sampled_feasibility: excluded after a measured memory failure.")
    else:
        results.append(knn_result)
        print(
            "knn_sampled_feasibility: validation Average Precision="
            f"{knn_result['validation_average_precision']:.6f}, "
            f"run={knn_result['run_id']} (ineligible for selection)"
        )

    decision = select_model_family(results, evidence_scope=evidence_scope)
    summary = {
        "purpose": "Compare model families under the frozen Stage 10 treatment.",
        "evidence_scope": evidence_scope,
        "feature_set": FEATURE_SET_NAME,
        "stage_10_handoff": handoff,
        "stage_10_decision": stage_10_summary["decision"],
        "split_manifest": split_manifest,
        "dataset_info": dataset_info,
        "raw_dataset_info": raw_info,
        "feature_lineage": feature_lineage,
        "model_results": results,
        "feasibility_exclusions": exclusions,
        "decision": decision,
        "test_split_evaluated": False,
    }
    with mlflow.start_run(run_name=f"model_family_{evidence_scope}_summary") as run:
        mlflow.set_tags(
            {
                "stage": "11_model_family_comparison",
                "run_role": f"{evidence_scope}_summary",
                "evidence_scope": evidence_scope,
                "test_split_evaluated": "false",
                "eligible_for_model_selection": str(not is_smoke).lower(),
            }
        )
        mlflow.log_params(
            {
                "feature_set": FEATURE_SET_NAME,
                "imbalance_strategy": handoff["selected_strategy"],
                "stage_10_summary_run_id": handoff["summary_run_id"],
                "split_assignment_sha256": split_manifest["split_assignment_sha256"],
                "selected_model": decision["selected_model"] or "none",
                "recommendation": decision["recommendation"],
            }
        )
        mlflow.log_dict(summary, f"model_family_{evidence_scope}_summary.json")
        mlflow.log_text(pd.DataFrame(results).to_csv(index=False), "model_results.csv")
        summary_run_id = run.info.run_id

    print("Stage 11 complete; the test split was not evaluated.")
    print(f"Summary run: {summary_run_id}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    if is_smoke:
        print("This was a smoke run and no model family was selected.")
    else:
        print(f"Selected model: {decision['selected_model']}.")
        print(f"Progression decision: {decision['recommendation']}.")


# %% Cell-aware executable entry point
if __name__ == "__main__":
    main()
