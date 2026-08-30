"""Test a small set of EDA-informed features with the selected Random Forest.

The earlier model comparison kept the original feature set fixed and selected Random Forest as the strongest model family.
This script now does the opposite: it keeps that Random Forest configuration fixed and changes only the feature set.

Feature groups are ranked with validation PR-AUC. The test split is used only for the group selected from the validation results.
"""

# %% Imports
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_modeling_utils import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURE_REPO_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    configure_mlflow,
    fit_evaluate_and_log,
    load_modeling_table,
    make_optional_sample,
    time_aware_split,
    validate_split_fractions,
)

# %% Experiment choices
FROZEN_ORIGINAL_VALIDATION_PR_AUC = 0.04395981671344107

AMOUNT_FEATURES = ["amount_log1p"]
TEMPORAL_FEATURES = ["transaction_hour", "transaction_day_of_week", "transaction_month", "transaction_is_weekend"]
MISSINGNESS_FEATURES = ["time_since_last_transaction_missing"]
ALL_ENGINEERED_FEATURES = AMOUNT_FEATURES + TEMPORAL_FEATURES + MISSINGNESS_FEATURES


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    description: str
    added_features: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return FEATURE_COLUMNS + self.added_features

    @property
    def numeric_features(self) -> list[str]:
        return NUMERIC_FEATURES + self.added_features


FEATURE_GROUPS = {
    "original_v1": FeatureGroup(
        name="original_v1",
        description="The ten original cleaned predictors. This rerun is the control for the experiment.",
        added_features=[],
    ),
    "amount_v2": FeatureGroup(
        name="amount_v2",
        description="Adds log1p(amount) to test whether a less-skewed view of transaction value helps.",
        added_features=AMOUNT_FEATURES,
    ),
    "temporal_v2": FeatureGroup(
        name="temporal_v2",
        description="Adds hour, weekday, month, and weekend fields derived from the current transaction timestamp.",
        added_features=TEMPORAL_FEATURES,
    ),
    "missingness_sensitivity_v2": FeatureGroup(
        name="missingness_sensitivity_v2",
        description="Adds a missing-value flag for time_since_last_transaction as a separate sensitivity check.",
        added_features=MISSINGNESS_FEATURES,
    ),
    "combined_v2": FeatureGroup(
        name="combined_v2",
        description="Adds all candidate features so their combined effect can be compared with the separate groups.",
        added_features=ALL_ENGINEERED_FEATURES,
    ),
}


# %% Command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare controlled feature groups with the Random Forest selected by the original-feature experiment."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the cleaned gold table when --data-source csv is used.",
    )
    parser.add_argument(
        "--data-source",
        choices=("feast", "csv"),
        default="feast",
        help="Use canonical Feast retrieval or an explicit direct-CSV fallback.",
    )
    parser.add_argument(
        "--feature-repo-path",
        type=Path,
        default=DEFAULT_FEATURE_REPO_PATH,
        help="Directory containing feature_store.yaml and the Feast registry.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional MLflow experiment-name override.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=list(FEATURE_GROUPS),
        default=list(FEATURE_GROUPS),
        help="Feature groups to compare. By default all five groups are run.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional stratified sample size for a smoke run. Omit this for the full experiment.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="Skip source hashing only when a quicker local smoke run is needed.",
    )
    return parser.parse_args()


# %% Feature calculation and checks
def add_candidate_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add row-level candidate features to the in-memory table without changing the persisted Feast data."""
    if (table["amount"] < 0).any():
        raise ValueError("amount_log1p cannot be calculated because the table contains a negative amount.")

    timestamps = pd.to_datetime(table["event_timestamp"], errors="raise")
    table["amount_log1p"] = np.log1p(table["amount"])
    table["transaction_hour"] = timestamps.dt.hour.astype("int8")
    table["transaction_day_of_week"] = timestamps.dt.dayofweek.astype("int8")
    table["transaction_month"] = timestamps.dt.month.astype("int8")
    table["transaction_is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype("int8")
    table["time_since_last_transaction_missing"] = table["time_since_last_transaction"].isna().astype("int8")
    return table


def validate_candidate_features(table: pd.DataFrame, expected_rows: int) -> None:
    """Fail early if a feature calculation changed the row set or produced an impossible value."""
    if len(table) != expected_rows:
        raise ValueError(f"Feature engineering changed the row count from {expected_rows:,} to {len(table):,}.")

    missing_columns = sorted(set(ALL_ENGINEERED_FEATURES) - set(table.columns))
    if missing_columns:
        raise ValueError(f"Candidate feature columns were not created: {missing_columns}")

    expected_ranges = {
        "transaction_hour": (0, 23),
        "transaction_day_of_week": (0, 6),
        "transaction_month": (1, 12),
        "transaction_is_weekend": (0, 1),
        "time_since_last_transaction_missing": (0, 1),
    }
    for column, (minimum, maximum) in expected_ranges.items():
        if not table[column].between(minimum, maximum).all():
            raise ValueError(f"{column} contains values outside the expected range {minimum} to {maximum}.")

    if not np.isfinite(table["amount_log1p"]).all():
        raise ValueError("amount_log1p contains a missing or infinite value.")


# %% Fixed model and preprocessing
def build_random_forest_pipeline(group: FeatureGroup, random_state: int) -> Pipeline:
    """Recreate the selected script-04 Random Forest while allowing the numeric feature list to change."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), group.numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def feature_record(group: FeatureGroup) -> dict[str, Any]:
    transformations = {
        "amount_log1p": "log(1 + amount)",
        "transaction_hour": "hour from the current transaction event_timestamp",
        "transaction_day_of_week": "weekday 0-6 from the current transaction event_timestamp",
        "transaction_month": "month 1-12 from the current transaction event_timestamp",
        "transaction_is_weekend": "1 for Saturday or Sunday, otherwise 0",
        "time_since_last_transaction_missing": "1 when time_since_last_transaction is missing, otherwise 0",
    }
    return {
        "feature_group": group.name,
        "group_description": group.description,
        "base_features": FEATURE_COLUMNS,
        "added_features": group.added_features,
        "transformations": {name: transformations[name] for name in group.added_features},
        "target_used_in_feature_creation": False,
        "future_information_used": False,
        "persisted_feast_v1_files_changed": False,
    }


def experiment_context(
    args: argparse.Namespace,
    group: FeatureGroup,
    source_row_count: int,
    experiment_row_count: int,
) -> dict[str, Any]:
    return {
        "workflow_stage": "controlled_feature_engineering",
        "dspm_alignment": "DP07_data_preparation_to_DP10_model_development_and_DP11_model_evaluation",
        "model_name": "random_forest_depth_12",
        "model_family": "tree_ensemble",
        "feature_group": group.name,
        "added_feature_count": len(group.added_features),
        "feature_count": len(group.feature_columns),
        "selection_metric": "validation_pr_auc",
        "frozen_original_validation_pr_auc": FROZEN_ORIGINAL_VALIDATION_PR_AUC,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        "source_dataset_rows": source_row_count,
        "experiment_rows": experiment_row_count,
        "is_smoke_sample": args.sample_rows is not None,
        "random_state": args.random_state,
    }


def source_files(data_source: str) -> list[Path]:
    files = [
        Path(__file__).resolve(),
        PROJECT_DIR / "fraud_modeling_utils.py",
        PROJECT_DIR / "Markdown files" / "05_feature_engineering_experiment_plan.md",
        PROJECT_DIR / "Markdown files" / "05_feature_engineering_implementation_guide.md",
    ]
    if data_source == "feast":
        files.extend(
            [
                PROJECT_DIR / "fraud_feature_store.py",
                PROJECT_DIR / "fraud_feature_definitions.py",
                PROJECT_DIR / "feature_store.yaml",
            ]
        )
    return [path for path in files if path.exists()]


# %% Experiment runner
def main() -> None:
    args = parse_args()
    validate_split_fractions(args.train_fraction, args.validation_fraction)

    modeling_table, dataset_info, feature_lineage = load_modeling_table(
        data_source=args.data_source,
        data_path=args.data_path,
        feature_repo_path=args.feature_repo_path,
        calculate_hash=not args.skip_data_hash,
    )
    source_row_count = len(modeling_table)
    modeling_table = make_optional_sample(modeling_table, args.sample_rows, args.random_state)
    expected_experiment_rows = len(modeling_table)
    add_candidate_features(modeling_table)
    validate_candidate_features(modeling_table, expected_experiment_rows)

    train_df, validation_df, test_df = time_aware_split(modeling_table, args.train_fraction, args.validation_fraction)
    experiment_name = args.experiment_name or (
        "financial-fraud-feast-feature-engineering"
        if args.data_source == "feast"
        else "financial-fraud-feature-engineering"
    )
    tracking_uri = configure_mlflow(experiment_name)
    selected_groups = [FEATURE_GROUPS[name] for name in args.feature_groups]
    tracked_sources = source_files(args.data_source)

    print("Running the controlled feature comparison.")
    print("The Random Forest configuration stays fixed; only the selected columns change.")
    comparison_rows: list[dict[str, Any]] = []

    for group in selected_groups:
        result = fit_evaluate_and_log(
            pipeline=build_random_forest_pipeline(group, args.random_state),
            run_name=f"feature_comparison__{group.name}",
            train_df=train_df,
            validation_df=validation_df,
            test_df=None,
            dataset_info=dataset_info,
            run_context={
                **experiment_context(args, group, source_row_count, len(modeling_table)),
                **feature_lineage,
            },
            source_files=tracked_sources,
            tags={
                "run_role": "feature_group_comparison",
                "feature_group": group.name,
                "selection_data": "validation_only",
                "test_split_used": "false",
                "evidence_scope": "smoke_test" if args.sample_rows is not None else "full_data",
            },
            log_model=False,
            feature_columns=group.feature_columns,
            categorical_features=CATEGORICAL_FEATURES,
            numeric_features=group.numeric_features,
            feature_metadata=feature_record(group),
        )
        comparison_rows.append(
            {
                "feature_group": group.name,
                "description": group.description,
                "feature_count": len(group.feature_columns),
                "added_features": group.added_features,
                "validation_pr_auc": result.validation_pr_auc,
                "validation_precision": result.validation_metrics["precision"],
                "validation_recall": result.validation_metrics["recall"],
                "validation_f1": result.validation_metrics["f1"],
                "validation_roc_auc": result.validation_metrics["roc_auc"],
                "threshold_tuned_on_validation": result.tuned_threshold,
                "run_id": result.run_id,
            }
        )
        print(f"{group.name}: validation PR-AUC={result.validation_pr_auc:.6f}, run={result.run_id}")

    selected_row = max(comparison_rows, key=lambda row: row["validation_pr_auc"])
    selected_group = FEATURE_GROUPS[selected_row["feature_group"]]
    original_row = next((row for row in comparison_rows if row["feature_group"] == "original_v1"), None)
    controlled_reference = (
        original_row["validation_pr_auc"] if original_row is not None else FROZEN_ORIGINAL_VALIDATION_PR_AUC
    )

    print(f"Selected from validation results: {selected_group.name}")
    print("Running one test evaluation for that selected feature group.")
    final_result = fit_evaluate_and_log(
        pipeline=build_random_forest_pipeline(selected_group, args.random_state),
        run_name=f"selected_feature_set__{selected_group.name}__test",
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        dataset_info=dataset_info,
        run_context={
            **experiment_context(args, selected_group, source_row_count, len(modeling_table)),
            **feature_lineage,
            "selected_from_run_id": selected_row["run_id"],
            "selection_basis": "highest_validation_pr_auc",
        },
        source_files=tracked_sources,
        tags={
            "run_role": "selected_feature_set_test_evaluation",
            "feature_group": selected_group.name,
            "selection_data": "validation_only",
            "test_split_used": "selected_group_only",
            "evidence_scope": "smoke_test" if args.sample_rows is not None else "full_data",
        },
        log_model=True,
        feature_columns=selected_group.feature_columns,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=selected_group.numeric_features,
        feature_metadata=feature_record(selected_group),
    )
    if final_result.test_metrics is None:
        raise RuntimeError("The selected feature-set run did not return test metrics.")

    absolute_change = float(selected_row["validation_pr_auc"] - controlled_reference)
    relative_change = float(absolute_change / controlled_reference) if controlled_reference else 0.0
    summary_payload = {
        "purpose": "Compare EDA-informed feature groups while keeping the selected Random Forest fixed.",
        "selection_metric": "validation_pr_auc",
        "test_selection_prohibited": True,
        "evidence_scope": "smoke_test" if args.sample_rows is not None else "full_data",
        "controlled_reference_validation_pr_auc": controlled_reference,
        "frozen_original_validation_pr_auc": FROZEN_ORIGINAL_VALIDATION_PR_AUC,
        "feature_group_results": comparison_rows,
        "selected_feature_group": selected_row,
        "selected_group_test_run_id": final_result.run_id,
        "selected_group_test_metrics": final_result.test_metrics,
        "absolute_validation_pr_auc_change": absolute_change,
        "relative_validation_pr_auc_change": relative_change,
    }

    with mlflow.start_run(run_name="feature_engineering_comparison_summary") as summary_run:
        mlflow.set_tags(
            {
                "run_role": "feature_comparison_summary",
                "selection_metric": "validation_pr_auc",
                "test_selection_prohibited": "true",
                "evidence_scope": "smoke_test" if args.sample_rows is not None else "full_data",
            }
        )
        mlflow.log_dict(summary_payload, "feature_engineering_comparison_summary.json")
        mlflow.log_params(
            {
                "selected_feature_group": selected_group.name,
                "selected_feature_count": len(selected_group.feature_columns),
                "selected_group_test_run_id": final_result.run_id,
                "selection_basis": "highest_validation_pr_auc",
                "comparison_group_count": len(comparison_rows),
                **feature_lineage,
            }
        )
        mlflow.log_metrics(
            {
                "selected_validation_pr_auc": selected_row["validation_pr_auc"],
                "controlled_reference_validation_pr_auc": controlled_reference,
                "absolute_validation_pr_auc_change": absolute_change,
                "relative_validation_pr_auc_change": relative_change,
                "selected_test_pr_auc": final_result.test_metrics["pr_auc"],
            }
        )

    print("Feature experiment complete.")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print(f"Summary run: {summary_run.info.run_id}")
    print(
        f"Selected group: {selected_group.name}; validation PR-AUC={selected_row['validation_pr_auc']:.6f}; "
        f"test PR-AUC={final_result.test_metrics['pr_auc']:.6f}"
    )
    if args.sample_rows is not None:
        print("This was a smoke run. Use the full dataset before treating the result as thesis evidence.")


# %% Script entry point
if __name__ == "__main__":
    main()
