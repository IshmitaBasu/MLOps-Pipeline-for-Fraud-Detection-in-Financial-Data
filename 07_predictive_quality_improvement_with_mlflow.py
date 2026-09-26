"""Stage 07: training-only behavioural-feature feasibility diagnostics.

This script creates a frozen stratified random split and explores only the
training partition. It does not train candidate models or evaluate the held-out
validation and test partitions.
"""

# %% Imports and project configuration
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from fraud_modeling_utils import PROJECT_DIR, configure_mlflow, dataset_metadata
from predictive_quality_utils import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    build_split_manifest,
    make_stratified_random_partition,
    run_training_diagnostics,
    training_partition,
)


RAW_DATA_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
DEFAULT_EXPERIMENT = "financial-fraud-stage-07-predictive-quality-improvement"
RAW_COLUMNS = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "amount",
    "location",
    "ip_address",
    "device_hash",
    "is_fraud",
]


# %% Command-line interface
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 07 training-only feature-feasibility diagnostics."
    )
    parser.add_argument("--mode", choices=["diagnostics"], default="diagnostics")
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only the first N rows for a smoke test.",
    )
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="Skip the source-file hash; allowed only for sampled smoke tests.",
    )
    return parser.parse_args()


# %% Raw-data loading and validation
def load_diagnostic_columns(path: Path, sample_rows: int | None) -> pd.DataFrame:
    """Load only columns required for split creation and feature diagnostics."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    if sample_rows is not None and sample_rows <= 0:
        raise ValueError("sample_rows must be positive when supplied.")

    data = pd.read_csv(
        path,
        usecols=RAW_COLUMNS,
        nrows=sample_rows,
        dtype={
            "transaction_id": "string",
            "sender_account": "string",
            "receiver_account": "string",
            "amount": "float64",
            "location": "category",
            "ip_address": "string",
            "device_hash": "string",
            "is_fraud": "boolean",
        },
    ).rename(columns={"timestamp": TIME_COLUMN})
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], format="ISO8601")

    if data.empty:
        raise ValueError("The raw dataset contains no rows.")
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("Transaction IDs must be present and unique.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The target contains missing values.")
    if data[TIME_COLUMN].isna().any():
        raise ValueError("The timestamp contains missing or invalid values.")
    if data["amount"].isna().any() or (data["amount"] < 0).any():
        raise ValueError("Transaction amounts must be present and non-negative.")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    return data


# %% MLflow logging for training-only evidence
def run_diagnostics(
    args: argparse.Namespace,
    training_data: pd.DataFrame,
    split_manifest: dict[str, object],
    source_metadata: dict[str, object],
) -> str:
    """Run and log diagnostics without accepting validation or test data."""
    results = run_training_diagnostics(training_data)
    run_name = (
        "stage_07_feature_feasibility_smoke"
        if args.sample_rows is not None
        else "stage_07_feature_feasibility_full"
    )

    tracked_source_files = [
        Path(__file__),
        PROJECT_DIR / "predictive_quality_utils.py",
        PROJECT_DIR / "Markdown files" / "07_predictive_quality_improvement_plan.md",
        PROJECT_DIR
        / "Markdown files"
        / "07_predictive_quality_improvement_implementation_guide.md",
        PROJECT_DIR
        / "Markdown files"
        / "07_predictive_quality_improvement_results.md",
    ]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "07_predictive_quality_improvement",
                "scope": "training_only_feature_feasibility",
                "evidence_scope": "smoke_test"
                if args.sample_rows is not None
                else "full_training_diagnostic",
                "test_split_evaluated": "false",
                "model_trained": "false",
                "eligible_for_model_promotion": "false",
            }
        )
        mlflow.log_params(
            {
                "split_method": "stratified_random",
                "split_seed": args.seed,
                "train_fraction": args.train_fraction,
                "validation_fraction": args.validation_fraction,
                "test_fraction": (
                    round(
                        1 - args.train_fraction - args.validation_fraction,
                        12,
                    )
                ),
                "sample_rows": args.sample_rows
                if args.sample_rows is not None
                else "all",
                **source_metadata,
            }
        )
        mlflow.log_metric("training_rows", len(training_data))
        mlflow.log_metric(
            "training_fraud_rate", float(training_data[TARGET_COLUMN].mean())
        )

        mlflow.log_dict(split_manifest, "split_manifest.json")
        mlflow.log_dict(
            results.feature_feasibility_decision,
            "feature_feasibility_decision.json",
        )
        mlflow.log_text(
            results.entity_repetition_summary.to_csv(index=False),
            "entity_repetition_summary.csv",
        )
        mlflow.log_text(
            results.new_vs_returning_fraud_rates.to_csv(index=False),
            "new_vs_returning_fraud_rates.csv",
        )
        mlflow.log_text(
            results.sender_amount_deviation_summary.to_csv(index=False),
            "sender_amount_deviation_summary.csv",
        )
        for source_file in tracked_source_files:
            if source_file.exists():
                mlflow.log_artifact(str(source_file), artifact_path="source_snapshot")

        print(json.dumps(results.feature_feasibility_decision, indent=2))
        return run.info.run_id


# %% Orchestration
def main() -> None:
    args = parse_args()
    if args.skip_data_hash and args.sample_rows is None:
        raise ValueError("--skip-data-hash is allowed only with --sample-rows.")

    configure_mlflow(args.experiment_name)
    source_metadata = dataset_metadata(
        args.raw_data_path,
        calculate_hash=not args.skip_data_hash,
    )
    data = load_diagnostic_columns(args.raw_data_path, args.sample_rows)
    partition = make_stratified_random_partition(
        data[TARGET_COLUMN],
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    split_manifest = build_split_manifest(
        data,
        partition,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    training_data = training_partition(data, partition)
    del data

    split_rows = split_manifest["splits"]
    print("Running Stage 07 training-only feature-feasibility diagnostics.")
    print(
        "Frozen stratified random split: "
        f"train={split_rows['train']['rows']:,}, "
        f"validation={split_rows['validation']['rows']:,}, "
        f"test={split_rows['test']['rows']:,}."
    )
    print("Validation and test rows will not be analysed in this diagnostic.")
    run_id = run_diagnostics(
        args,
        training_data,
        split_manifest,
        source_metadata,
    )
    print(f"Diagnostic run: {run_id}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {args.experiment_name}")
    if args.sample_rows is not None:
        print("This was a smoke run and is not thesis evidence.")


# %% Cell-aware executable entry point
if __name__ == "__main__":
    main()
