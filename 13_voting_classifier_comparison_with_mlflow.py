"""Stage 13: validation-only soft voting and threshold diagnostics.

The fixed Stage 12 handoff, split, behavioural features, and class-weight
treatment are mandatory. Test rows are counted but never materialised.
"""

# %% Imports and runtime configuration
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mlflow
import pandas as pd

from fraud_modeling_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURE_REPO_PATH,
    PROJECT_DIR,
    TARGET_COLUMN,
    configure_mlflow,
    dataset_metadata,
    load_modeling_table,
    make_optional_sample,
    predict_scores,
)
from lightgbm_tuning_utils import frozen_lightgbm_configs
from model_family_comparison_utils import DEFAULT_RANDOM_STATE, FROZEN_FULL_SPLIT_HASH, frozen_model_family_configs
from oversampling_experiment_utils import FEATURE_SET_NAME, MODEL_FEATURES
from predictive_quality_utils import TEST_CODE, build_split_manifest, make_stratified_random_partition
from sender_location_history_utils import build_development_frames, restore_raw_source_order
from validation_evaluation_utils import evaluate_validation_scores, positive_class_weight
from voting_classifier_utils import (
    CANDIDATES,
    build_voting_candidate,
    select_voting_candidate,
    validate_stage_12_handoff,
    validation_threshold_table,
)

# %% Frozen experiment inputs
DEFAULT_EXPERIMENT = "financial-fraud-stage-13-voting-classifier-comparison"
STAGE_12_SUMMARY_RUN_ID = "cceb57941a914ba2b3e015573b3e34b1"
STAGE_12_ARTIFACT = "lightgbm_tuning_full_data_summary.json"
RAW_DATA_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
SMOKE_ROWS = 50_000
FULL_ROWS = 5_000_000


# %% CLI and handoff
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed soft-voting candidates and validation threshold rules.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-source", choices=("feast", "csv"), default="feast")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--feature-repo-path", type=Path, default=DEFAULT_FEATURE_REPO_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--stage-12-summary-run-id", default=STAGE_12_SUMMARY_RUN_ID)
    return parser.parse_args()


def load_stage_12_handoff(run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    if run.data.tags.get("run_role") != "full_data_summary":
        raise ValueError("The supplied Stage 12 run is not a full-data summary.")
    path = Path(client.download_artifacts(run_id, STAGE_12_ARTIFACT))
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["summary_run_id"] = run_id
    return summary, validate_stage_12_handoff(summary)


# %% Reproducibility snapshots
def tracked_source_files() -> list[Path]:
    names = (
        "13_voting_classifier_comparison_with_mlflow.py",
        "voting_classifier_utils.py",
        "validation_evaluation_utils.py",
        "model_family_comparison_utils.py",
        "lightgbm_tuning_utils.py",
        "sender_location_history_utils.py",
        "oversampling_experiment_utils.py",
        "fraud_modeling_utils.py",
        "predictive_quality_utils.py",
        "project_io_utils.py",
        "requirements.txt",
        "Markdown files/13_voting_classifier_comparison_plan.md",
        "Markdown files/13_voting_classifier_comparison_implementation_guide.md",
        "Markdown files/13_voting_classifier_comparison_results.md",
    )
    return [PROJECT_DIR / name for name in names]


def candidate_definition(name: str) -> dict[str, Any]:
    configs = {config.name: asdict(config) for config in frozen_model_family_configs()}
    configs["lightgbm"] = asdict(frozen_lightgbm_configs()[0].as_model_family_config())
    members = CANDIDATES[name]
    return {
        "name": name,
        "voting": "soft" if len(members) > 1 else "single_model",
        "weights": [1 / len(members)] * len(members),
        "members": [configs[member] for member in members],
        "probabilities_calibrated": False,
    }


# %% One validation candidate, including decision-rule diagnostics
def run_candidate(
    name: str,
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    split_manifest: dict[str, Any],
    handoff: dict[str, str],
    evidence_scope: str,
    class_weight_ratio: float,
    dataset_info: dict[str, Any],
    feature_lineage: dict[str, Any],
) -> dict[str, Any]:
    model = build_voting_candidate(name, positive_class_weight=class_weight_ratio)
    with mlflow.start_run(run_name=f"voting_{evidence_scope}__{name}") as run:
        mlflow.set_tags(
            {
                "stage": "13_voting_classifier_comparison",
                "run_role": "comparison_candidate",
                "evidence_scope": evidence_scope,
                "test_split_evaluated": "false",
                "eligible_for_model_selection": str(evidence_scope == "full_data").lower(),
            }
        )
        mlflow.log_params(
            {
                "configuration": name,
                "feature_set": FEATURE_SET_NAME,
                "imbalance_strategy": "class_weight_reference",
                "class_weight_ratio": class_weight_ratio,
                "stage_12_summary_run_id": handoff["summary_run_id"],
                "random_state": DEFAULT_RANDOM_STATE,
                "split_assignment_sha256": split_manifest["split_assignment_sha256"],
            }
        )
        mlflow.log_dict(candidate_definition(name), "candidate_definition.json")
        mlflow.log_dict(split_manifest, "split_manifest.json")
        mlflow.log_dict(dataset_info, "dataset_info.json")
        mlflow.log_dict(feature_lineage, "feature_lineage.json")
        mlflow.log_dict(
            {
                package: version(package)
                for package in ("scikit-learn", "lightgbm", "catboost", "mlflow", "pandas", "numpy")
            },
            "library_versions.json",
        )
        for source in tracked_source_files():
            mlflow.log_artifact(str(source), artifact_path="source_snapshot")

        fit_start = perf_counter()
        model.fit(train_df[MODEL_FEATURES], train_df[TARGET_COLUMN])
        training_seconds = perf_counter() - fit_start
        inference_start = perf_counter()
        scores, _, score_type = predict_scores(model, validation_df[MODEL_FEATURES])
        inference_seconds = perf_counter() - inference_start
        metrics, workloads = evaluate_validation_scores(validation_df[TARGET_COLUMN], scores, validation_df["amount"])
        thresholds = validation_threshold_table(validation_df[TARGET_COLUMN], scores, validation_df["amount"])
        fraud_rows = int(validation_df[TARGET_COLUMN].sum())
        legitimate_rows = len(validation_df) - fraud_rows
        workloads["false_negatives"] = fraud_rows - workloads["fraud_cases_captured"]
        workloads["true_negatives"] = legitimate_rows - workloads["legitimate_alerts"]
        total_fraud_value = float(validation_df.loc[validation_df[TARGET_COLUMN] == 1, "amount"].sum())
        workloads["missed_fraud_value"] = total_fraud_value - workloads["fraudulent_value_captured"]
        result = {
            "configuration": name,
            "run_id": run.info.run_id,
            "training_rows": len(train_df),
            "validation_rows": len(validation_df),
            "training_seconds": training_seconds,
            "validation_inference_seconds": inference_seconds,
            **metrics,
        }
        mlflow.log_param("score_type", score_type)
        mlflow.log_metrics({key: value for key, value in result.items() if isinstance(value, (int, float))})
        mlflow.log_text(workloads.to_csv(index=False), "validation_workloads.csv")
        mlflow.log_text(thresholds.to_csv(index=False), "validation_thresholds.csv")
        confusion_columns = [
            "rule",
            "threshold",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]
        mlflow.log_text(thresholds[confusion_columns].to_csv(index=False), "validation_confusion_matrices.csv")
        mlflow.log_dict(result, "voting_result.json")
    del model, scores
    return result


# %% Orchestration with the frozen split
def main() -> None:
    args = parse_args()
    is_smoke = args.mode == "smoke"
    scope = "smoke_test" if is_smoke else "full_data"
    tracking_uri = configure_mlflow(args.experiment_name)
    stage_12_summary, handoff = load_stage_12_handoff(args.stage_12_summary_run_id)
    table, dataset_info, lineage = load_modeling_table(
        data_source=args.data_source,
        data_path=args.data_path,
        feature_repo_path=args.feature_repo_path,
        calculate_hash=not is_smoke,
    )
    raw_info = dataset_metadata(args.raw_data_path, calculate_hash=not is_smoke)
    if is_smoke:
        table = make_optional_sample(table, SMOKE_ROWS, DEFAULT_RANDOM_STATE)
    elif len(table) != FULL_ROWS:
        raise ValueError("The Stage 13 full comparison requires exactly 5,000,000 rows.")
    table = restore_raw_source_order(table, args.raw_data_path)
    partition = make_stratified_random_partition(
        table[TARGET_COLUMN], train_fraction=0.70, validation_fraction=0.15, seed=DEFAULT_RANDOM_STATE
    )
    manifest = build_split_manifest(
        table, partition, seed=DEFAULT_RANDOM_STATE, train_fraction=0.70, validation_fraction=0.15
    )
    if not is_smoke and manifest["split_assignment_sha256"] != handoff["split_assignment_sha256"]:
        raise ValueError("Stage 13 does not match the frozen Stage 12 split.")
    if not is_smoke and manifest["split_assignment_sha256"] != FROZEN_FULL_SPLIT_HASH:
        raise ValueError("The Stage 13 full split hash changed.")
    splits = manifest["splits"]
    print(f"Running the Stage 13 validation-only voting {'smoke workflow' if is_smoke else 'full-data comparison'}.")
    print(
        f"Frozen stratified random split: train={splits['train']['rows']:,}, "
        f"validation={splits['validation']['rows']:,}, test={splits['test']['rows']:,}."
    )
    print("The test split will not be materialised or evaluated.")
    train_df, validation_df = build_development_frames(table, partition, args.raw_data_path)
    if int((partition == TEST_CODE).sum()) != splits["test"]["rows"]:
        raise ValueError("The test partition count changed.")
    del table, partition
    gc.collect()
    ratio = positive_class_weight(train_df[TARGET_COLUMN])
    results = []
    for name in CANDIDATES:
        result = run_candidate(
            name,
            train_df=train_df,
            validation_df=validation_df,
            split_manifest=manifest,
            handoff=handoff,
            evidence_scope=scope,
            class_weight_ratio=ratio,
            dataset_info=dataset_info,
            feature_lineage=lineage,
        )
        results.append(result)
        print(
            f"{name}: validation Average Precision={result['validation_average_precision']:.6f}, "
            f"F1={result['validation_f1']:.6f}, run={result['run_id']}"
        )
        gc.collect()
    decision = select_voting_candidate(results, evidence_scope=scope)
    summary = {
        "purpose": "Compare fixed soft voting and inspect validation decision thresholds.",
        "evidence_scope": scope,
        "stage_12_handoff": handoff,
        "stage_12_decision": stage_12_summary["decision"],
        "feature_set": FEATURE_SET_NAME,
        "imbalance_strategy": "class_weight_reference",
        "split_manifest": manifest,
        "dataset_info": dataset_info,
        "raw_dataset_info": raw_info,
        "feature_lineage": lineage,
        "candidate_results": results,
        "decision": decision,
        "test_split_evaluated": False,
        "operating_rule_frozen": False,
    }
    with mlflow.start_run(run_name=f"voting_{scope}_summary") as run:
        mlflow.set_tags(
            {
                "stage": "13_voting_classifier_comparison",
                "run_role": f"{scope}_summary",
                "evidence_scope": scope,
                "test_split_evaluated": "false",
            }
        )
        mlflow.log_dict(summary, f"voting_{scope}_summary.json")
        mlflow.log_text(pd.DataFrame(results).to_csv(index=False), "voting_comparison_results.csv")
        mlflow.log_param("selected_configuration", decision["selected_configuration"] or "none")
        summary_run_id = run.info.run_id
    print("Stage 13 complete; the test split was not evaluated.")
    print(f"Summary run: {summary_run_id}\nTracking URI: {tracking_uri}\nExperiment: {args.experiment_name}")
    if is_smoke:
        print("This was a smoke run and no model was selected.")
    else:
        print(f"Selected candidate: {decision['selected_configuration']}.")
        print(f"Progression decision: {decision['recommendation']}.")
    print("Threshold diagnostics are recorded; a final operating rule has not been frozen.")


# %% Cell-aware executable entry point
if __name__ == "__main__":
    main()
