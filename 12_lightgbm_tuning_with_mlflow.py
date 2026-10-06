"""Stage 12: limited validation-only tuning of the selected LightGBM model.

The Stage 11 full-data decision is a mandatory input. The feature set,
class-weight treatment, split, and candidate list are fixed before execution.
The held-out test partition is counted but never materialised or evaluated.
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
)
from lightgbm_tuning_utils import (
    LightGBMTuningConfig,
    frozen_lightgbm_configs,
    select_lightgbm_configuration,
    validate_stage_11_handoff,
)
from model_family_comparison_utils import DEFAULT_RANDOM_STATE, FROZEN_FULL_SPLIT_HASH, build_model_pipeline
from oversampling_experiment_utils import FEATURE_SET_NAME, MODEL_FEATURES
from predictive_quality_utils import TEST_CODE, build_split_manifest, make_stratified_random_partition
from sender_location_history_utils import build_development_frames, restore_raw_source_order
from validation_evaluation_utils import fit_and_evaluate_validation, positive_class_weight

# %% Frozen experiment definition
RAW_DATA_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
DEFAULT_EXPERIMENT = "financial-fraud-stage-12-lightgbm-tuning"
STAGE_11_SUMMARY_RUN_ID = "ba890331d5a149f0bb17ee1a7f63c589"
STAGE_11_SUMMARY_ARTIFACT = "model_family_full_data_summary.json"
DEFAULT_SMOKE_ROWS = 50_000
FULL_DATA_ROWS = 5_000_000


# %% Command-line interface
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed LightGBM tuning search on validation data only. Smoke mode cannot select a configuration."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-source", choices=("feast", "csv"), default="feast")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--feature-repo-path", type=Path, default=DEFAULT_FEATURE_REPO_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--stage-11-summary-run-id", default=STAGE_11_SUMMARY_RUN_ID)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


# %% Handoff and reproducibility helpers
def load_stage_11_handoff(run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Download and validate the frozen full-data Stage 11 summary."""

    if not run_id.strip():
        raise ValueError("A Stage 11 summary run ID is required.")
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    if run.data.tags.get("run_role") != "full_data_summary":
        raise ValueError("The supplied Stage 11 run is not a full-data summary run.")
    local_path = Path(client.download_artifacts(run_id, STAGE_11_SUMMARY_ARTIFACT))
    summary = json.loads(local_path.read_text(encoding="utf-8"))
    summary["summary_run_id"] = run_id
    return summary, validate_stage_11_handoff(summary)


def tracked_source_files() -> list[Path]:
    return [
        PROJECT_DIR / "12_lightgbm_tuning_with_mlflow.py",
        PROJECT_DIR / "lightgbm_tuning_utils.py",
        PROJECT_DIR / "validation_evaluation_utils.py",
        PROJECT_DIR / "model_family_comparison_utils.py",
        PROJECT_DIR / "sender_location_history_utils.py",
        PROJECT_DIR / "oversampling_experiment_utils.py",
        PROJECT_DIR / "fraud_modeling_utils.py",
        PROJECT_DIR / "predictive_quality_utils.py",
        PROJECT_DIR / "project_io_utils.py",
        PROJECT_DIR / "requirements.txt",
        PROJECT_DIR / "Markdown files" / "12_lightgbm_tuning_plan.md",
        PROJECT_DIR / "Markdown files" / "12_lightgbm_tuning_implementation_guide.md",
        PROJECT_DIR / "Markdown files" / "12_lightgbm_tuning_results.md",
    ]


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("scikit-learn", "lightgbm", "mlflow", "pandas", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


# %% One controlled tuning run
def run_configuration(
    *,
    config: LightGBMTuningConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    split_manifest: dict[str, Any],
    handoff: dict[str, str],
    evidence_scope: str,
    class_weight_ratio: float,
    dataset_info: dict[str, Any],
    feature_lineage: dict[str, Any],
) -> dict[str, Any]:
    """Fit one declared configuration without accepting a test frame."""

    pipeline = build_model_pipeline(config.as_model_family_config(), positive_class_weight=class_weight_ratio, random_state=DEFAULT_RANDOM_STATE)
    metrics, workloads, score_type = fit_and_evaluate_validation(pipeline, train_df, validation_df, feature_columns=MODEL_FEATURES, target_column=TARGET_COLUMN)
    result = {"configuration": config.name, "model_family": "lightgbm", "run_id": "pending", **metrics}

    with mlflow.start_run(run_name=f"lightgbm_tuning_{evidence_scope}__{config.name}") as run:
        mlflow.set_tags(
            {
                "stage": "12_lightgbm_tuning",
                "run_role": "tuning_candidate",
                "evidence_scope": evidence_scope,
                "test_split_evaluated": "false",
                "eligible_for_tuning_selection": str(evidence_scope == "full_data").lower(),
            }
        )
        mlflow.log_params(
            {
                "configuration": config.name,
                "model_family": "lightgbm",
                "feature_set": FEATURE_SET_NAME,
                "imbalance_strategy": "class_weight_reference",
                "stage_11_summary_run_id": handoff["summary_run_id"],
                "split_assignment_sha256": split_manifest["split_assignment_sha256"],
                "score_type": score_type,
                "class_weight_ratio": class_weight_ratio,
                "random_state": DEFAULT_RANDOM_STATE,
                **dict(config.parameters),
            }
        )
        mlflow.log_metrics(
            {
                key: value
                for key, value in result.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        mlflow.log_dict(asdict(config), "lightgbm_configuration.json")
        mlflow.log_dict(split_manifest, "split_manifest.json")
        mlflow.log_dict(dataset_info, "dataset_info.json")
        mlflow.log_dict(feature_lineage, "feature_lineage.json")
        mlflow.log_dict(library_versions(), "library_versions.json")
        mlflow.log_text(workloads.to_csv(index=False), "validation_workloads.csv")
        for source in tracked_source_files():
            mlflow.log_artifact(str(source), artifact_path="source_snapshot")
        result["run_id"] = run.info.run_id
        mlflow.log_dict(result, "tuning_result.json")
    del pipeline
    return result


# %% Experiment orchestration
def main() -> None:
    args = parse_args()
    if args.random_state != DEFAULT_RANDOM_STATE:
        raise ValueError("Stage 12 is frozen to random state 42.")
    is_smoke = args.mode == "smoke"
    evidence_scope = "smoke_test" if is_smoke else "full_data"

    tracking_uri = configure_mlflow(args.experiment_name)
    stage_11_summary, handoff = load_stage_11_handoff(args.stage_11_summary_run_id)
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
        raise ValueError("The Stage 12 full run requires exactly 5,000,000 rows.")

    modeling_table = restore_raw_source_order(modeling_table, args.raw_data_path)
    partition = make_stratified_random_partition(
        modeling_table[TARGET_COLUMN], train_fraction=0.70, validation_fraction=0.15, seed=DEFAULT_RANDOM_STATE
    )
    split_manifest = build_split_manifest(
        modeling_table, partition, seed=DEFAULT_RANDOM_STATE, train_fraction=0.70, validation_fraction=0.15
    )
    if not is_smoke and split_manifest["split_assignment_sha256"] != FROZEN_FULL_SPLIT_HASH:
        raise ValueError("The Stage 12 full run does not match the frozen split hash.")
    if not is_smoke and split_manifest["split_assignment_sha256"] != handoff["split_assignment_sha256"]:
        raise ValueError("Stage 12 and the Stage 11 handoff use different splits.")

    split_rows = split_manifest["splits"]
    print(
        "Running the Stage 12 validation-only LightGBM "
        f"{'tuning smoke workflow' if is_smoke else 'full-data tuning comparison'}."
    )
    print(
        "Frozen stratified random split: "
        f"train={split_rows['train']['rows']:,}, validation={split_rows['validation']['rows']:,}, "
        f"test={split_rows['test']['rows']:,}."
    )
    print("The test split will not be materialised or evaluated.")
    print("Frozen inputs: behavioural feature set and class_weight_reference.")

    train_df, validation_df = build_development_frames(modeling_table, partition, args.raw_data_path)
    if int((partition == TEST_CODE).sum()) != split_rows["test"]["rows"]:
        raise ValueError("The held-out test row count changed unexpectedly.")
    del modeling_table, partition
    gc.collect()

    class_weight_ratio = positive_class_weight(train_df[TARGET_COLUMN])
    results: list[dict[str, Any]] = []
    for config in frozen_lightgbm_configs():
        result = run_configuration(
            config=config,
            train_df=train_df,
            validation_df=validation_df,
            split_manifest=split_manifest,
            handoff=handoff,
            evidence_scope=evidence_scope,
            class_weight_ratio=class_weight_ratio,
            dataset_info=dataset_info,
            feature_lineage=feature_lineage,
        )
        results.append(result)
        print(
            f"{config.name}: validation Average Precision="
            f"{result['validation_average_precision']:.6f}, run={result['run_id']}"
        )
        gc.collect()

    decision = select_lightgbm_configuration(results, evidence_scope=evidence_scope)
    summary = {
        "purpose": "Tune the selected LightGBM model with a small predeclared validation-only search.",
        "evidence_scope": evidence_scope,
        "feature_set": FEATURE_SET_NAME,
        "imbalance_strategy": "class_weight_reference",
        "stage_11_handoff": handoff,
        "stage_11_decision": stage_11_summary["decision"],
        "split_manifest": split_manifest,
        "dataset_info": dataset_info,
        "raw_dataset_info": raw_info,
        "feature_lineage": feature_lineage,
        "configuration_results": results,
        "decision": decision,
        "test_split_evaluated": False,
    }
    with mlflow.start_run(run_name=f"lightgbm_tuning_{evidence_scope}_summary") as run:
        mlflow.set_tags(
            {
                "stage": "12_lightgbm_tuning",
                "run_role": f"{evidence_scope}_summary",
                "evidence_scope": evidence_scope,
                "test_split_evaluated": "false",
                "eligible_for_tuning_selection": str(not is_smoke).lower(),
            }
        )
        mlflow.log_params(
            {
                "feature_set": FEATURE_SET_NAME,
                "imbalance_strategy": "class_weight_reference",
                "stage_11_summary_run_id": handoff["summary_run_id"],
                "split_assignment_sha256": split_manifest["split_assignment_sha256"],
                "selected_configuration": decision["selected_configuration"] or "none",
                "recommendation": decision["recommendation"],
            }
        )
        mlflow.log_dict(summary, f"lightgbm_tuning_{evidence_scope}_summary.json")
        mlflow.log_text(pd.DataFrame(results).to_csv(index=False), "lightgbm_tuning_results.csv")
        summary_run_id = run.info.run_id

    print("Stage 12 complete; the test split was not evaluated.")
    print(f"Summary run: {summary_run_id}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    if is_smoke:
        print("This was a smoke run and no LightGBM configuration was selected.")
    else:
        print(f"Selected configuration: {decision['selected_configuration']}.")
        print(f"Progression decision: {decision['recommendation']}.")


# %% Cell-aware executable entry point
if __name__ == "__main__":
    main()
