"""Stage 10 controlled smoke and full-data fraud-class resampling workflow."""

# %% Imports and runtime limits
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mlflow
import numpy as np
import pandas as pd

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
)
from oversampling_experiment_utils import (
    FEATURE_SET_NAME,
    MODEL_FEATURES,
    RANDOM_STATE,
    build_development_frames,
    build_fixed_random_forest,
    categorical_levels_from_training,
    dataframe_fingerprint,
    fit_train_representation,
    frozen_strategies,
    resample_training_data,
    restore_raw_source_order,
    select_validation_strategy,
    validate_full_run_identity,
)
from predictive_quality_utils import (
    ID_COLUMN,
    TEST_CODE,
    TIME_COLUMN,
    build_split_manifest,
    make_stratified_random_partition,
)
from sender_location_history_utils import DEFAULT_ALERT_RATES, fixed_workload_table

# %% Frozen experiment definition
RAW_DATA_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
DEFAULT_EXPERIMENT = "financial-fraud-stage-10-oversampling"
FROZEN_FULL_SPLIT_HASH = "e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242"
DEFAULT_SMOKE_ROWS = 50_000


# %% Command-line interface
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the validation-only Stage 10 oversampling workflow. Smoke "
            "scores cannot select an imbalance strategy."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--data-source", choices=("feast", "csv"), default="feast")
    parser.add_argument("--feature-repo-path", type=Path, default=DEFAULT_FEATURE_REPO_PATH)
    parser.add_argument(
        "--sample-rows", type=int, default=None, help="Optional smoke size; Stage 10 currently requires exactly 50,000."
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    return parser.parse_args()


# %% Reproducibility and logging helpers
def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def tracked_source_files() -> list[Path]:
    paths = [
        Path(__file__).resolve(),
        PROJECT_DIR / "oversampling_experiment_utils.py",
        PROJECT_DIR / "sender_location_history_utils.py",
        PROJECT_DIR / "predictive_quality_utils.py",
        PROJECT_DIR / "fraud_modeling_utils.py",
        PROJECT_DIR / "project_io_utils.py",
        PROJECT_DIR / "Markdown files" / "10_oversampling_experiment_plan.md",
        PROJECT_DIR / "Markdown files" / "10_oversampling_experiment_implementation_guide.md",
        PROJECT_DIR / "Markdown files" / "10_oversampling_experiment_results.md",
    ]
    return [path for path in paths if path.exists()]


def log_strategy_run(
    *,
    strategy: Any,
    diagnostics: dict[str, Any],
    metrics: dict[str, float | int],
    workload: pd.DataFrame,
    split_manifest: dict[str, Any],
    feature_lineage: dict[str, Any],
    dataset_info: dict[str, Any],
    timings: dict[str, float],
    evidence_scope: str,
) -> str:
    with mlflow.start_run(run_name=f"oversampling_{evidence_scope}__{strategy.name}") as run:
        mlflow.set_tags(
            {
                "stage": "10_oversampling_experiment",
                "run_role": f"strategy_{evidence_scope}",
                "evidence_scope": evidence_scope,
                "selection_data": "validation_only",
                "test_split_evaluated": "false",
                "eligible_for_strategy_selection": str(evidence_scope == "full_data").lower(),
                "strategy": strategy.name,
            }
        )
        mlflow.log_params(
            {
                "strategy": strategy.name,
                "sampler_family": strategy.sampler_family,
                "sampling_strategy": (strategy.sampling_strategy if strategy.sampling_strategy is not None else "none"),
                "class_weight": strategy.class_weight or "none",
                "random_state": RANDOM_STATE,
                "feature_set": FEATURE_SET_NAME,
                "feature_count": len(MODEL_FEATURES),
                "split_assignment_sha256": split_manifest["split_assignment_sha256"],
                "rf_n_estimators": 100,
                "rf_max_depth": 12,
                "rf_min_samples_leaf": 100,
                "rf_max_features": "sqrt",
                "imbalanced_learn_version": importlib.metadata.version("imbalanced-learn"),
            }
        )
        mlflow.log_metrics(
            {
                "validation_pr_auc": float(metrics["pr_auc"]),
                "validation_roc_auc": float(metrics["roc_auc"]),
                "validation_precision_at_max_f1": float(metrics["precision"]),
                "validation_recall_at_max_f1": float(metrics["recall"]),
                "validation_f1": float(metrics["f1"]),
                **timings,
                "training_rows_before_resampling": diagnostics["rows_before"],
                "training_rows_after_resampling": diagnostics["rows_after"],
            }
        )
        for row in workload.to_dict(orient="records"):
            label = f"{int(round(row['requested_alert_rate'] * 100))}pct"
            mlflow.log_metrics(
                {
                    f"workload_{label}_precision": float(row["precision"]),
                    f"workload_{label}_fraud_count_recall": float(row["fraud_count_recall"]),
                    f"workload_{label}_fraud_value_recall": float(row["fraud_value_recall"]),
                }
            )
        mlflow.log_dict(asdict(strategy), "strategy_configuration.json")
        mlflow.log_dict(diagnostics, "resampling_diagnostics.json")
        mlflow.log_dict(split_manifest, "split_manifest.json")
        mlflow.log_dict(feature_lineage, "feature_lineage.json")
        mlflow.log_dict(dataset_info, "dataset_metadata.json")
        mlflow.log_dict(metrics, "validation_metrics.json")
        mlflow.log_text(workload.to_csv(index=False), "validation_fixed_workload.csv")
        for source in tracked_source_files():
            mlflow.log_artifact(str(source), artifact_path="source_snapshot")
        return run.info.run_id


# %% One controlled strategy run
def run_strategy(
    *,
    strategy: Any,
    train_array: np.ndarray,
    train_labels: pd.Series,
    validation_array: np.ndarray,
    validation_evaluation: pd.DataFrame,
    validation_fingerprint: str,
    validation_array_fingerprint: str,
    split_manifest: dict[str, Any],
    feature_lineage: dict[str, Any],
    dataset_info: dict[str, Any],
    categorical_levels: list[np.ndarray],
    evidence_scope: str,
) -> dict[str, Any]:
    resampling_started = time.perf_counter()
    resampled = resample_training_data(train_array, train_labels, strategy, random_state=RANDOM_STATE)
    resampling_seconds = time.perf_counter() - resampling_started

    reproducible = True
    if strategy.sampler_family == "smotenc" and evidence_scope == "smoke_test":
        repeated = resample_training_data(train_array, train_labels, strategy, random_state=RANDOM_STATE)
        reproducible = bool(
            array_sha256(resampled.features) == array_sha256(repeated.features)
            and array_sha256(resampled.labels) == array_sha256(repeated.labels)
        )
        if not reproducible:
            raise ValueError(f"{strategy.name} is not reproducible with seed 42.")
        del repeated
        gc.collect()
    elif strategy.sampler_family == "smotenc":
        resampled.diagnostics["reproducibility_confirmed_by_smoke_gate"] = True
    resampled.diagnostics["reproducible_with_seed_42"] = reproducible
    resampled.diagnostics["feature_array_bytes"] = int(resampled.features.nbytes)

    pipeline = build_fixed_random_forest(strategy, random_state=RANDOM_STATE, categorical_levels=categorical_levels)
    training_started = time.perf_counter()
    pipeline.fit(resampled.features, resampled.labels)
    training_seconds = time.perf_counter() - training_started
    inference_started = time.perf_counter()
    validation_scores = pipeline.predict_proba(validation_array)[:, 1]
    validation_inference_seconds = time.perf_counter() - inference_started

    threshold, _ = best_f1_threshold(validation_evaluation[TARGET_COLUMN], validation_scores)
    metrics = classification_metrics(validation_evaluation[TARGET_COLUMN], validation_scores, threshold)
    metrics["maximum_f1_threshold"] = float(threshold)
    metrics["validation_fraud_prevalence"] = float(validation_evaluation[TARGET_COLUMN].mean())
    metrics["average_precision_lift_over_prevalence"] = float(
        metrics["pr_auc"] / metrics["validation_fraud_prevalence"]
    )
    workload = fixed_workload_table(
        validation_evaluation[TARGET_COLUMN], validation_scores, validation_evaluation["amount"], DEFAULT_ALERT_RATES
    )
    if dataframe_fingerprint(validation_evaluation) != validation_fingerprint:
        raise ValueError("A strategy changed the validation frame.")
    if array_sha256(validation_array) != validation_array_fingerprint:
        raise ValueError("A strategy changed the transformed validation features.")

    timings = {
        "resampling_seconds": float(resampling_seconds),
        "training_seconds": float(training_seconds),
        "validation_inference_seconds": float(validation_inference_seconds),
    }
    run_id = log_strategy_run(
        strategy=strategy,
        diagnostics=resampled.diagnostics,
        metrics=metrics,
        workload=workload,
        split_manifest=split_manifest,
        feature_lineage=feature_lineage,
        dataset_info=dataset_info,
        timings=timings,
        evidence_scope=evidence_scope,
    )
    workload_5pct = workload.loc[np.isclose(workload["requested_alert_rate"], 0.05)].iloc[0]
    result = {
        "strategy": strategy.name,
        "run_id": run_id,
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
        "validation_precision_at_max_f1": float(metrics["precision"]),
        "validation_recall_at_max_f1": float(metrics["recall"]),
        "validation_f1": float(metrics["f1"]),
        "workload_5pct_fraud_count_recall": float(workload_5pct["fraud_count_recall"]),
        "workload_5pct_fraud_value_recall": float(workload_5pct["fraud_value_recall"]),
        "training_rows_after_resampling": resampled.diagnostics["rows_after"],
        **timings,
    }
    del pipeline, validation_scores, resampled
    gc.collect()
    return result


# %% Full-data feasibility exclusion logging
def log_memory_exclusion(strategy: Any, error: MemoryError) -> dict[str, Any]:
    reason = f"MemoryError: {error}" if str(error) else "MemoryError"
    with mlflow.start_run(run_name=f"oversampling_full_data__{strategy.name}") as run:
        mlflow.set_tags(
            {
                "stage": "10_oversampling_experiment",
                "run_role": "full_data_feasibility_exclusion",
                "evidence_scope": "full_data",
                "strategy": strategy.name,
                "test_split_evaluated": "false",
                "eligible_for_strategy_selection": "false",
                "run_status": "excluded_for_measured_memory_failure",
            }
        )
        mlflow.log_params(
            {
                "strategy": strategy.name,
                "sampler_family": strategy.sampler_family,
                "sampling_strategy": strategy.sampling_strategy,
                "exclusion_reason": reason,
            }
        )
        mlflow.log_dict(
            {
                "strategy": strategy.name,
                "status": "excluded_for_measured_memory_failure",
                "reason": reason,
                "test_split_evaluated": False,
            },
            "feasibility_exclusion.json",
        )
        return {
            "strategy": strategy.name,
            "status": "excluded_for_measured_memory_failure",
            "reason": reason,
            "run_id": run.info.run_id,
        }


# %% Smoke and full-data orchestration
def main() -> None:
    args = parse_args()
    if args.random_state != RANDOM_STATE:
        raise ValueError("Stage 10 is frozen to random state 42.")
    is_smoke = args.mode == "smoke"
    if is_smoke:
        sample_rows = args.sample_rows or DEFAULT_SMOKE_ROWS
        if sample_rows != DEFAULT_SMOKE_ROWS:
            raise ValueError("The Stage 10 smoke run is frozen to 50,000 rows.")
    else:
        if args.sample_rows is not None:
            raise ValueError("--sample-rows cannot be used with --mode full.")
        sample_rows = None
    evidence_scope = "smoke_test" if is_smoke else "full_data"

    tracking_uri = configure_mlflow(args.experiment_name)
    modeling_table, dataset_info, feature_lineage = load_modeling_table(
        data_source=args.data_source,
        data_path=args.data_path,
        feature_repo_path=args.feature_repo_path,
        calculate_hash=not is_smoke,
    )
    source_rows = len(modeling_table)
    raw_metadata = dataset_metadata(args.raw_data_path, calculate_hash=not is_smoke)
    dataset_info = {**dataset_info, **{f"raw_{key}": value for key, value in raw_metadata.items()}}
    if is_smoke:
        modeling_table = make_optional_sample(modeling_table, sample_rows, args.random_state)
    modeling_table = restore_raw_source_order(modeling_table, args.raw_data_path)
    partition = make_stratified_random_partition(
        modeling_table[TARGET_COLUMN], train_fraction=0.70, validation_fraction=0.15, seed=args.random_state
    )
    split_manifest = build_split_manifest(
        modeling_table, partition, seed=args.random_state, train_fraction=0.70, validation_fraction=0.15
    )
    if not is_smoke:
        validate_full_run_identity(
            source_rows=source_rows,
            observed_split_hash=split_manifest["split_assignment_sha256"],
            expected_split_hash=FROZEN_FULL_SPLIT_HASH,
        )
    split_rows = split_manifest["splits"]
    print(
        "Running the Stage 10 validation-only oversampling "
        f"{'smoke workflow' if is_smoke else 'full-data comparison'}."
    )
    print(
        "Frozen stratified random split: "
        f"train={split_rows['train']['rows']:,}, "
        f"validation={split_rows['validation']['rows']:,}, "
        f"test={split_rows['test']['rows']:,}."
    )
    print("Test rows will not be materialised, resampled, or evaluated.")

    train_df, validation_df = build_development_frames(modeling_table, partition, args.raw_data_path)
    if int((partition == TEST_CODE).sum()) != split_rows["test"]["rows"]:
        raise ValueError("The held-out test row count changed unexpectedly.")
    del modeling_table, partition
    gc.collect()

    _, train_array, validation_array = fit_train_representation(train_df, validation_df)
    train_labels = train_df[TARGET_COLUMN].copy()
    validation_evaluation = validation_df[[ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, "amount"]].copy()
    validation_fingerprint = dataframe_fingerprint(validation_evaluation)
    validation_array_fingerprint = array_sha256(validation_array)
    categorical_levels = categorical_levels_from_training(train_array)
    del train_df, validation_df
    gc.collect()

    results: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for strategy in frozen_strategies():
        try:
            result = run_strategy(
                strategy=strategy,
                train_array=train_array,
                train_labels=train_labels,
                validation_array=validation_array,
                validation_evaluation=validation_evaluation,
                validation_fingerprint=validation_fingerprint,
                validation_array_fingerprint=validation_array_fingerprint,
                split_manifest=split_manifest,
                feature_lineage=feature_lineage,
                dataset_info=dataset_info,
                categorical_levels=categorical_levels,
                evidence_scope=evidence_scope,
            )
        except MemoryError as error:
            if not is_smoke and strategy.sampler_family == "smotenc":
                exclusion = log_memory_exclusion(strategy, error)
                exclusions.append(exclusion)
                print(f"{strategy.name}: excluded after a measured memory failure, " f"run={exclusion['run_id']}")
                gc.collect()
                continue
            raise
        results.append(result)
        print(
            f"{strategy.name}: validation Average Precision="
            f"{result['validation_pr_auc']:.6f}, run={result['run_id']}"
        )

    decision = select_validation_strategy(results, evidence_scope=evidence_scope)
    summary = {
        "purpose": (
            "Technical feasibility check for the frozen imbalance comparison."
            if is_smoke
            else "Select an imbalance strategy using full validation evidence."
        ),
        "evidence_scope": evidence_scope,
        "feature_set": FEATURE_SET_NAME,
        "split_manifest": split_manifest,
        "frozen_full_split_hash": FROZEN_FULL_SPLIT_HASH,
        "strategies": results,
        "feasibility_exclusions": exclusions,
        "decision": decision,
        "test_split_evaluated": False,
    }
    with mlflow.start_run(run_name=f"oversampling_experiment_{evidence_scope}_summary") as run:
        mlflow.set_tags(
            {
                "stage": "10_oversampling_experiment",
                "run_role": f"{evidence_scope}_summary",
                "evidence_scope": evidence_scope,
                "test_split_evaluated": "false",
                "eligible_for_strategy_selection": str(not is_smoke).lower(),
            }
        )
        mlflow.log_params(
            {
                "sample_rows": sample_rows if sample_rows is not None else "all",
                "strategy_count": len(results),
                "excluded_strategy_count": len(exclusions),
                "feature_set": FEATURE_SET_NAME,
                "split_assignment_sha256": split_manifest["split_assignment_sha256"],
                "recommendation": decision["recommendation"],
                "selected_strategy": decision["selected_strategy"] or "none",
            }
        )
        mlflow.log_dict(summary, f"oversampling_{evidence_scope}_summary.json")
        mlflow.log_text(pd.DataFrame(results).to_csv(index=False), "strategy_results.csv")
        summary_run_id = run.info.run_id

    print(
        f"Stage 10 {'smoke workflow' if is_smoke else 'full-data comparison'} "
        "complete; the test split was not evaluated."
    )
    print(f"Summary run: {summary_run_id}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    if is_smoke:
        print("This was a smoke run and no imbalance strategy was selected.")
    else:
        print(f"Selected imbalance strategy: {decision['selected_strategy']}.")
        print(f"Progression decision: {decision['recommendation']}.")


# %% Cell-aware executable entry point
if __name__ == "__main__":
    main()
