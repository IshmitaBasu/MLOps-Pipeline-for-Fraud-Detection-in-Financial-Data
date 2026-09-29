"""Stage 08: controlled sender-location history feature comparison.

The fixed Random Forest is trained once with ``original_v1`` and once with a
small point-in-time sender-location feature group. Selection uses validation
evidence only. The test partition is counted in the frozen manifest but is not
materialised, scored, or passed to the experiment runner.
"""

# %% Imports and runtime configuration
from __future__ import annotations

import argparse
import gc
import os
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
    TARGET_COLUMN,
    configure_mlflow,
    dataset_metadata,
    fit_evaluate_and_log,
    load_modeling_table,
    make_optional_sample,
    predict_scores,
    validate_split_fractions,
)
from predictive_quality_utils import (
    ID_COLUMN,
    TEST_CODE,
    TIME_COLUMN,
    TRAIN_CODE,
    VALIDATION_CODE,
    build_split_manifest,
    make_stratified_random_partition,
)
from sender_location_history_utils import (
    DEFAULT_ALERT_RATES,
    HISTORY_FEATURES,
    SENDER_COLUMN,
    add_sender_location_history_features,
    candidate_progression_decision,
    fixed_workload_table,
)


# %% Frozen experiment definition
RAW_DATA_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
DEFAULT_EXPERIMENT = "financial-fraud-stage-08-sender-location-history"
FULL_DATA_ROW_COUNT = 5_000_000
FROZEN_FULL_SPLIT_HASH = (
    "e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242"
)
CONTROL_GROUP = "original_v1"
CANDIDATE_GROUP = "sender_location_history_v1"


# %% Command-line interface
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original_v1 with past-only sender-location history using "
            "a fixed Random Forest and validation evidence only."
        )
    )
    parser.add_argument("--mode", choices=["comparison"], default="comparison")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument(
        "--data-source", choices=("feast", "csv"), default="feast"
    )
    parser.add_argument(
        "--feature-repo-path", type=Path, default=DEFAULT_FEATURE_REPO_PATH
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="Skip source hashes; allowed only for a sampled smoke run.",
    )
    return parser.parse_args()


# %% Data-key retrieval and feature construction
def restore_raw_source_order(
    modeling_table: pd.DataFrame,
    raw_data_path: Path,
) -> pd.DataFrame:
    """Place canonical modeling rows in the verified raw transaction-ID order."""
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")
    if modeling_table[ID_COLUMN].duplicated().any():
        raise ValueError("The modeling table contains duplicate transaction IDs.")

    transaction_ids = modeling_table[ID_COLUMN].astype("string")
    if not transaction_ids.str.fullmatch(r"T\d+").all():
        raise ValueError(
            "Canonical transaction IDs must use the verified T<number> format."
        )
    raw_order = transaction_ids.str.slice(1).astype("int64")
    if raw_order.duplicated().any():
        raise ValueError("Numeric transaction-ID suffixes must be unique.")

    ordered = modeling_table.assign(_raw_source_order=raw_order.to_numpy())
    ordered = ordered.sort_values("_raw_source_order", kind="mergesort")
    return ordered.drop(columns="_raw_source_order").reset_index(drop=True)


def attach_sender_account(
    development_data: pd.DataFrame,
    raw_data_path: Path,
    chunk_rows: int = 500_000,
) -> pd.DataFrame:
    """Join sender keys only for development rows while scanning the raw CSV."""
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")
    required_ids = pd.Index(development_data[ID_COLUMN].astype("string"))
    key_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_data_path,
        usecols=[ID_COLUMN, SENDER_COLUMN],
        dtype={ID_COLUMN: "string", SENDER_COLUMN: "string"},
        chunksize=chunk_rows,
    ):
        selected = chunk.loc[chunk[ID_COLUMN].isin(required_ids)]
        if not selected.empty:
            key_parts.append(selected)

    if not key_parts:
        raise ValueError("No sender keys matched the development transactions.")
    sender_keys = pd.concat(key_parts, ignore_index=True)
    if sender_keys[ID_COLUMN].duplicated().any():
        raise ValueError("The raw data contains duplicate development transaction IDs.")

    original_ids = development_data[ID_COLUMN].astype("string").reset_index(drop=True)
    enriched = development_data.merge(
        sender_keys,
        on=ID_COLUMN,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if not enriched[ID_COLUMN].astype("string").reset_index(drop=True).equals(
        original_ids
    ):
        raise ValueError("Joining sender keys changed the development row order.")
    if enriched[SENDER_COLUMN].isna().any():
        missing_count = int(enriched[SENDER_COLUMN].isna().sum())
        raise ValueError(f"Sender keys are missing for {missing_count:,} rows.")
    return enriched


def build_development_frames(
    modeling_table: pd.DataFrame,
    partition: np.ndarray,
    raw_data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature-complete train/validation frames without a test argument."""
    development_mask = partition != TEST_CODE
    development_partition = partition[development_mask]
    development_data = modeling_table.loc[development_mask].reset_index(drop=True)
    development_data = attach_sender_account(development_data, raw_data_path)
    development_data = add_sender_location_history_features(
        development_data,
        development_partition,
    )

    train_df = development_data.loc[
        development_partition == TRAIN_CODE
    ].reset_index(drop=True)
    validation_df = development_data.loc[
        development_partition == VALIDATION_CODE
    ].reset_index(drop=True)
    if train_df.empty or validation_df.empty:
        raise ValueError("Training and validation frames must both be non-empty.")
    return train_df, validation_df


# %% Fixed model and feature metadata
def feature_columns(group_name: str) -> list[str]:
    if group_name == CONTROL_GROUP:
        return list(FEATURE_COLUMNS)
    if group_name == CANDIDATE_GROUP:
        return [*FEATURE_COLUMNS, *HISTORY_FEATURES]
    raise ValueError(f"Unknown feature group: {group_name}")


def numeric_features(group_name: str) -> list[str]:
    if group_name == CONTROL_GROUP:
        return list(NUMERIC_FEATURES)
    if group_name == CANDIDATE_GROUP:
        return [*NUMERIC_FEATURES, *HISTORY_FEATURES]
    raise ValueError(f"Unknown feature group: {group_name}")


def build_fixed_random_forest(group_name: str, random_state: int) -> Pipeline:
    """Recreate the fixed depth-12 Random Forest for both feature groups."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features(group_name),
            ),
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


def feature_metadata(group_name: str) -> dict[str, Any]:
    return {
        "feature_set_name": group_name,
        "base_features": FEATURE_COLUMNS,
        "added_features": HISTORY_FEATURES if group_name == CANDIDATE_GROUP else [],
        "history_source": (
            "strictly_earlier_training_transactions"
            if group_name == CANDIDATE_GROUP
            else "not_applicable"
        ),
        "validation_rows_update_history": False,
        "test_rows_used_in_feature_construction": False,
        "target_used_in_feature_construction": False,
        "definitions": {
            "sender_prior_transaction_count": (
                "number of strictly earlier training transactions for the sender"
            ),
            "sender_location_prior_transaction_count": (
                "number of strictly earlier training transactions for the "
                "sender-location pair"
            ),
            "is_new_location_for_returning_sender": (
                "1 when sender history exists but sender-location history does not"
            ),
        }
        if group_name == CANDIDATE_GROUP
        else {},
    }


# %% MLflow workload evidence
def log_workload_evidence(
    run_id: str,
    group_name: str,
    workload: pd.DataFrame,
) -> None:
    records = workload.astype(object).where(pd.notna(workload), None)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(
            {
                "feature_group": group_name,
                "evaluation_split": "validation",
                "equal_workload_comparison": True,
                "transaction_value_is_a_loss_proxy": True,
                "scenarios": records.to_dict(orient="records"),
            },
            "operational_evaluation/validation_fixed_workload.json",
        )
        mlflow.log_text(
            workload.to_csv(index=False),
            "operational_evaluation/validation_fixed_workload.csv",
        )
        metrics: dict[str, float] = {}
        for row in workload.to_dict(orient="records"):
            rate_name = f"{int(round(row['requested_alert_rate'] * 100))}pct"
            metrics[f"workload_{rate_name}_precision"] = float(row["precision"])
            metrics[f"workload_{rate_name}_fraud_count_recall"] = float(
                row["fraud_count_recall"]
            )
            metrics[f"workload_{rate_name}_fraud_value_recall"] = float(
                row["fraud_value_recall"]
            )
        mlflow.log_metrics(metrics)


def tracked_source_files() -> list[Path]:
    candidates = [
        Path(__file__).resolve(),
        PROJECT_DIR / "sender_location_history_utils.py",
        PROJECT_DIR / "predictive_quality_utils.py",
        PROJECT_DIR / "fraud_modeling_utils.py",
        PROJECT_DIR
        / "Markdown files"
        / "08_sender_location_history_feature_experiment_plan.md",
        PROJECT_DIR
        / "Markdown files"
        / "08_sender_location_history_feature_experiment_implementation_guide.md",
        PROJECT_DIR
        / "Markdown files"
        / "08_sender_location_history_feature_experiment_results.md",
    ]
    return [path for path in candidates if path.exists()]


# %% Validation-only controlled comparison
def run_feature_comparison(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    dataset_info: dict[str, Any],
    feature_lineage: dict[str, Any],
    split_manifest: dict[str, object],
) -> tuple[dict[str, Any], str]:
    """Run both feature groups without accepting or forwarding a test frame."""
    evidence_scope = "smoke_test" if args.sample_rows is not None else "full_data"
    group_results: dict[str, dict[str, Any]] = {}
    workload_tables: dict[str, pd.DataFrame] = {}
    sources = tracked_source_files()

    for group_name in (CONTROL_GROUP, CANDIDATE_GROUP):
        pipeline = build_fixed_random_forest(group_name, args.random_state)
        selected_features = feature_columns(group_name)
        result = fit_evaluate_and_log(
            pipeline=pipeline,
            run_name=f"sender_location_comparison__{group_name}",
            train_df=train_df,
            validation_df=validation_df,
            test_df=None,
            dataset_info=dataset_info,
            run_context={
                **feature_lineage,
                "workflow_stage": "sender_location_history_feature_comparison",
                "experiment_scope": "validation_only",
                "feature_group": group_name,
                "model_name": "random_forest_depth_12",
                "model_configuration_frozen": True,
                "selection_metric": "validation_pr_auc",
                "split_method": "stratified_random",
                "split_seed": args.random_state,
                "split_assignment_sha256": split_manifest[
                    "split_assignment_sha256"
                ],
                "evidence_scope": evidence_scope,
                "test_split_evaluated": False,
            },
            source_files=sources,
            tags={
                "stage": "08_sender_location_history_feature_experiment",
                "run_role": "controlled_feature_comparison",
                "feature_group": group_name,
                "selection_data": "validation_only",
                "test_split_evaluated": "false",
                "evidence_scope": evidence_scope,
                "eligible_for_model_promotion": "false",
            },
            log_model=False,
            feature_columns=selected_features,
            categorical_features=CATEGORICAL_FEATURES,
            numeric_features=numeric_features(group_name),
            feature_metadata=feature_metadata(group_name),
        )

        validation_scores, _, _ = predict_scores(
            pipeline, validation_df[selected_features]
        )
        workload = fixed_workload_table(
            validation_df[TARGET_COLUMN],
            validation_scores,
            validation_df["amount"],
            DEFAULT_ALERT_RATES,
        )
        log_workload_evidence(result.run_id, group_name, workload)
        stored_metrics = mlflow.get_run(result.run_id).data.metrics
        group_results[group_name] = {
            "feature_group": group_name,
            "run_id": result.run_id,
            "feature_count": len(selected_features),
            "added_features": (
                HISTORY_FEATURES if group_name == CANDIDATE_GROUP else []
            ),
            "validation_pr_auc": result.validation_pr_auc,
            "validation_precision_at_max_f1": result.validation_metrics["precision"],
            "validation_recall_at_max_f1": result.validation_metrics["recall"],
            "validation_f1": result.validation_metrics["f1"],
            "validation_roc_auc": result.validation_metrics["roc_auc"],
            "threshold_tuned_on_validation": result.tuned_threshold,
            "training_seconds": stored_metrics.get("training_seconds"),
            "validation_inference_seconds": stored_metrics.get(
                "validation_inference_seconds"
            ),
        }
        workload_tables[group_name] = workload
        print(
            f"{group_name}: validation Average Precision="
            f"{result.validation_pr_auc:.6f}, run={result.run_id}"
        )
        del pipeline, validation_scores
        gc.collect()

    decision = candidate_progression_decision(
        group_results[CONTROL_GROUP]["validation_pr_auc"],
        group_results[CANDIDATE_GROUP]["validation_pr_auc"],
        workload_tables[CONTROL_GROUP],
        workload_tables[CANDIDATE_GROUP],
    )
    decision["decision_eligible"] = args.sample_rows is None
    if args.sample_rows is not None:
        decision["candidate_progresses"] = False
        decision["recommendation"] = "smoke_run_no_feature_decision"
    summary_payload = {
        "purpose": (
            "Test whether past-only sender-location history improves the fixed "
            "Random Forest over original_v1."
        ),
        "evidence_scope": evidence_scope,
        "model_configuration_changed_between_groups": False,
        "test_split_evaluated": False,
        "split_manifest": split_manifest,
        "group_results": group_results,
        "progression_decision": decision,
    }
    combined_workload = pd.concat(
        [
            table.assign(feature_group=group_name)
            for group_name, table in workload_tables.items()
        ],
        ignore_index=True,
    )

    with mlflow.start_run(run_name="sender_location_feature_comparison_summary") as run:
        mlflow.set_tags(
            {
                "stage": "08_sender_location_history_feature_experiment",
                "run_role": "comparison_summary",
                "selection_data": "validation_only",
                "test_split_evaluated": "false",
                "evidence_scope": evidence_scope,
                "eligible_for_model_promotion": "false",
            }
        )
        mlflow.log_params(
            {
                "control_group": CONTROL_GROUP,
                "candidate_group": CANDIDATE_GROUP,
                "model_name": "random_forest_depth_12",
                "split_assignment_sha256": split_manifest[
                    "split_assignment_sha256"
                ],
                "candidate_progresses": (
                    decision["candidate_progresses"]
                ),
                "evidence_scope": evidence_scope,
            }
        )
        mlflow.log_metrics(
            {
                "control_validation_pr_auc": decision[
                    "control_validation_pr_auc"
                ],
                "candidate_validation_pr_auc": decision[
                    "candidate_validation_pr_auc"
                ],
                "absolute_pr_auc_change": decision["absolute_pr_auc_change"],
                "relative_pr_auc_change": decision["relative_pr_auc_change"],
            }
        )
        mlflow.log_dict(summary_payload, "sender_location_feature_comparison.json")
        mlflow.log_text(
            combined_workload.to_csv(index=False),
            "validation_equal_workload_comparison.csv",
        )
        mlflow.log_dict(split_manifest, "split_manifest.json")
        summary_run_id = run.info.run_id

    return summary_payload, summary_run_id


# %% Orchestration
def main() -> None:
    args = parse_args()
    validate_split_fractions(args.train_fraction, args.validation_fraction)
    if args.skip_data_hash and args.sample_rows is None:
        raise ValueError("--skip-data-hash is allowed only with --sample-rows.")
    if (args.train_fraction, args.validation_fraction, args.random_state) != (
        0.70,
        0.15,
        42,
    ):
        raise ValueError(
            "Stage 08 is frozen to a 70/15/15 split with random state 42."
        )

    tracking_uri = configure_mlflow(args.experiment_name)
    modeling_table, feast_metadata, feature_lineage = load_modeling_table(
        data_source=args.data_source,
        data_path=args.data_path,
        feature_repo_path=args.feature_repo_path,
        calculate_hash=not args.skip_data_hash,
    )
    source_rows = len(modeling_table)
    modeling_table = make_optional_sample(
        modeling_table, args.sample_rows, args.random_state
    )
    modeling_table = restore_raw_source_order(
        modeling_table,
        args.raw_data_path,
    )
    partition = make_stratified_random_partition(
        modeling_table[TARGET_COLUMN],
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.random_state,
    )
    split_manifest = build_split_manifest(
        modeling_table,
        partition,
        seed=args.random_state,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    if source_rows == FULL_DATA_ROW_COUNT and args.sample_rows is None:
        observed_hash = split_manifest["split_assignment_sha256"]
        if observed_hash != FROZEN_FULL_SPLIT_HASH:
            raise ValueError(
                "The full-data split does not match the frozen Stage 07 split. "
                f"Expected {FROZEN_FULL_SPLIT_HASH}, observed {observed_hash}."
            )

    raw_metadata = dataset_metadata(
        args.raw_data_path,
        calculate_hash=not args.skip_data_hash,
    )
    dataset_info = {
        **feast_metadata,
        **{f"raw_{key}": value for key, value in raw_metadata.items()},
    }
    split_rows = split_manifest["splits"]
    print("Running the controlled sender-location history feature comparison.")
    print(
        "Frozen stratified random split: "
        f"train={split_rows['train']['rows']:,}, "
        f"validation={split_rows['validation']['rows']:,}, "
        f"test={split_rows['test']['rows']:,}."
    )
    print("The test split will not be materialised or evaluated.")

    train_df, validation_df = build_development_frames(
        modeling_table,
        partition,
        args.raw_data_path,
    )
    del modeling_table, partition
    gc.collect()

    summary, summary_run_id = run_feature_comparison(
        args,
        train_df,
        validation_df,
        dataset_info,
        feature_lineage,
        split_manifest,
    )
    decision = summary["progression_decision"]
    print("Validation-only comparison complete; the test split was not evaluated.")
    print(f"Summary run: {summary_run_id}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    if args.sample_rows is not None:
        print("This was a smoke run and is not thesis evidence.")
    else:
        print(f"Progression decision: {decision['recommendation']}.")


# %% Cell-aware executable entry point
if __name__ == "__main__":
    main()
