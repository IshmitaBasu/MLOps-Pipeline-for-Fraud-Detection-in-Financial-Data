"""Compare model and imbalance choices, then study usable alert thresholds.

Random Forest and Histogram Gradient Boosting are repeated as frozen controls.
The new evidence in this stage comes from comparing a selected external booster,
training-only imbalance strategies, and validation-based alert/cost scenarios.

This screening runner never evaluates the test split.  A separate final step
will be added only after one candidate and operating threshold have been frozen.
"""

# %% Imports
from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mlflow
import pandas as pd

from fraud_modeling_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURE_REPO_PATH,
    FEATURE_COLUMNS,
    PROJECT_DIR,
    TARGET_COLUMN,
    configure_mlflow,
    fit_evaluate_and_log,
    load_modeling_table,
    make_optional_sample,
    predict_scores,
    time_aware_split,
    validate_split_fractions,
)
from model_optimization_utils import (
    DEFAULT_ALERT_RATES,
    DEFAULT_COST_SCENARIOS,
    SUPPORTED_IMBALANCE_STRATEGIES,
    SUPPORTED_MODELS,
    apply_training_imbalance_strategy,
    build_candidate_pipeline,
    build_threshold_summary,
    model_optimization_configurations,
    positive_class_weight,
    relative_validation_improvement,
    select_validation_candidate,
)


# %% Fixed experiment controls
FROZEN_REFERENCE_VALIDATION_PR_AUC = 0.04395981671344107
PROVISIONAL_RELATIVE_IMPROVEMENT_TARGET = 0.05
CONTROL_MODELS = (
    "random_forest_depth_12",
    "hist_gradient_boosting_leaves_31",
)
EXTERNAL_MODELS = ("xgboost", "lightgbm")


# %% Command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-only model optimisation and operational threshold "
            "analysis with the frozen original_v1 feature set."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Cleaned gold CSV used only when --data-source csv is selected.",
    )
    parser.add_argument(
        "--data-source",
        choices=("feast", "csv"),
        default="feast",
        help="Canonical Feast retrieval or an explicit direct-CSV fallback.",
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
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(CONTROL_MODELS),
        help=(
            "Candidates to compare. The default reruns the two frozen controls; "
            "add either xgboost or lightgbm after choosing and installing one."
        ),
    )
    parser.add_argument(
        "--imbalance-strategies",
        nargs="+",
        choices=SUPPORTED_IMBALANCE_STRATEGIES,
        default=["class_weight"],
        help=(
            "Training-only imbalance methods. Multiple choices create a model-by-"
            "strategy comparison matrix."
        ),
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional stratified row count for a technical smoke run.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="Skip source hashing only for a quicker local smoke run.",
    )
    return parser.parse_args()


# %% Experiment safeguards
def validate_experiment_choices(args: argparse.Namespace) -> None:
    validate_split_fractions(args.train_fraction, args.validation_fraction)
    if len(args.models) != len(set(args.models)):
        raise ValueError("Each model should be listed only once.")
    if len(args.imbalance_strategies) != len(set(args.imbalance_strategies)):
        raise ValueError("Each imbalance strategy should be listed only once.")
    selected_external_models = set(args.models).intersection(EXTERNAL_MODELS)
    if len(selected_external_models) > 1:
        raise ValueError("Choose either XGBoost or LightGBM for this stage, not both.")


# %% Reproducibility and run metadata
def tracked_source_files(data_source: str) -> list[Path]:
    files = [
        Path(__file__).resolve(),
        PROJECT_DIR / "model_optimization_utils.py",
        PROJECT_DIR / "fraud_modeling_utils.py",
        PROJECT_DIR
        / "Markdown files"
        / "06_model_optimization_and_operational_evaluation_plan.md",
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


def experiment_context(
    args: argparse.Namespace,
    *,
    model_name: str,
    imbalance_metadata: dict[str, int | float | bool | str],
    source_rows: int,
    experiment_rows: int,
) -> dict[str, Any]:
    configuration = model_optimization_configurations()[model_name]
    return {
        "workflow_stage": "model_optimization_and_operational_evaluation",
        "experiment_scope": "validation_screening_only",
        "configuration_name": configuration.name,
        "model_family": configuration.model_family,
        "preprocessing_family": configuration.preprocessing,
        "feature_set": "original_v1",
        "selection_metric": "validation_pr_auc",
        "reference_validation_pr_auc": FROZEN_REFERENCE_VALIDATION_PR_AUC,
        "provisional_relative_improvement_target": PROVISIONAL_RELATIVE_IMPROVEMENT_TARGET,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        "source_dataset_rows": source_rows,
        "experiment_rows": experiment_rows,
        "is_smoke_sample": args.sample_rows is not None,
        "random_state": args.random_state,
        **imbalance_metadata,
    }


# %% Operational threshold artifacts
def threshold_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a threshold table to JSON records without pandas NaN values."""

    cleaned = table.astype(object).where(pd.notna(table), None)
    return cleaned.to_dict(orient="records")


def log_operational_threshold_evidence(
    *,
    run_id: str,
    threshold_summary: pd.DataFrame,
) -> None:
    """Attach capacity and cost evidence to the already-created model run."""

    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(
            {
                "evidence_split": "validation",
                "cost_values_are_sensitivity_assumptions": True,
                "threshold_scenarios": threshold_records(threshold_summary),
            },
            "operational_evaluation/validation_threshold_summary.json",
        )
        with TemporaryDirectory(prefix="fraud_operational_evaluation_") as temporary_dir:
            csv_path = Path(temporary_dir) / "validation_threshold_summary.csv"
            threshold_summary.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="operational_evaluation")

        logged_metrics: dict[str, float] = {}
        for row in threshold_summary.to_dict(orient="records"):
            selection = str(row["selection"]).replace(".", "_")
            logged_metrics[f"operational_{selection}_actual_alert_rate"] = float(row["alert_rate"])
            logged_metrics[f"operational_{selection}_precision"] = float(row["precision"])
            logged_metrics[f"operational_{selection}_recall"] = float(row["recall"])
            if pd.notna(row.get("expected_cost")):
                logged_metrics[f"operational_{selection}_expected_cost"] = float(row["expected_cost"])
        mlflow.log_metrics(logged_metrics)


# %% Validation-only experiment runner
def main() -> None:
    args = parse_args()
    validate_experiment_choices(args)

    modeling_table, dataset_info, feature_lineage = load_modeling_table(
        data_source=args.data_source,
        data_path=args.data_path,
        feature_repo_path=args.feature_repo_path,
        calculate_hash=not args.skip_data_hash,
    )
    source_rows = len(modeling_table)
    modeling_table = make_optional_sample(modeling_table, args.sample_rows, args.random_state)
    train_df, validation_df, held_out_test_df = time_aware_split(
        modeling_table,
        args.train_fraction,
        args.validation_fraction,
    )

    experiment_name = args.experiment_name or (
        "financial-fraud-feast-model-optimization-operational-evaluation"
        if args.data_source == "feast"
        else "financial-fraud-model-optimization-operational-evaluation"
    )
    tracking_uri = configure_mlflow(experiment_name)
    source_files = tracked_source_files(args.data_source)
    evidence_scope = "smoke_test" if args.sample_rows is not None else "full_data_validation"
    comparison_rows: list[dict[str, Any]] = []

    print("Running validation-only model optimisation and operational evaluation.")
    print(f"The held-out test split contains {len(held_out_test_df):,} rows and will not be evaluated.")

    for model_name in args.models:
        for strategy in args.imbalance_strategies:
            effective_train_df, imbalance_metadata = apply_training_imbalance_strategy(
                train_df,
                strategy,
                random_state=args.random_state,
            )
            uses_class_weight = strategy == "class_weight"
            external_weight = (
                positive_class_weight(effective_train_df[TARGET_COLUMN])
                if uses_class_weight and model_name in EXTERNAL_MODELS
                else None
            )
            pipeline = build_candidate_pipeline(
                model_name,
                random_state=args.random_state,
                use_class_weight=uses_class_weight,
                external_positive_class_weight=external_weight,
            )
            run_name = f"screening__{model_name}__{strategy}"
            result = fit_evaluate_and_log(
                pipeline=pipeline,
                run_name=run_name,
                train_df=effective_train_df,
                validation_df=validation_df,
                test_df=None,
                dataset_info=dataset_info,
                run_context={
                    **experiment_context(
                        args,
                        model_name=model_name,
                        imbalance_metadata=imbalance_metadata,
                        source_rows=source_rows,
                        experiment_rows=len(modeling_table),
                    ),
                    **feature_lineage,
                },
                source_files=source_files,
                tags={
                    "run_role": "control_model" if model_name in CONTROL_MODELS else "external_candidate",
                    "imbalance_strategy": strategy,
                    "selection_data": "validation_only",
                    "test_split_used": "false",
                    "evidence_scope": evidence_scope,
                },
                log_model=False,
            )

            validation_scores, default_threshold, _ = predict_scores(
                pipeline,
                validation_df[FEATURE_COLUMNS],
            )
            thresholds = build_threshold_summary(
                validation_df[TARGET_COLUMN],
                validation_scores,
                default_threshold=default_threshold,
                alert_rates=DEFAULT_ALERT_RATES,
                cost_scenarios=DEFAULT_COST_SCENARIOS,
            )
            log_operational_threshold_evidence(
                run_id=result.run_id,
                threshold_summary=thresholds,
            )

            relative_improvement = relative_validation_improvement(
                result.validation_pr_auc,
                FROZEN_REFERENCE_VALIDATION_PR_AUC,
            )
            comparison_rows.append(
                {
                    "model": model_name,
                    "imbalance_strategy": strategy,
                    "validation_pr_auc": result.validation_pr_auc,
                    "validation_precision_at_max_f1": result.validation_metrics["precision"],
                    "validation_recall_at_max_f1": result.validation_metrics["recall"],
                    "validation_f1": result.validation_metrics["f1"],
                    "validation_roc_auc": result.validation_metrics["roc_auc"],
                    "relative_improvement_over_frozen_reference": relative_improvement,
                    "meets_provisional_improvement_target": (
                        relative_improvement >= PROVISIONAL_RELATIVE_IMPROVEMENT_TARGET
                    ),
                    "threshold_scenarios": threshold_records(thresholds),
                    "run_id": result.run_id,
                }
            )
            print(
                f"{model_name} + {strategy}: validation Average Precision="
                f"{result.validation_pr_auc:.6f}, run={result.run_id}"
            )

    selected = select_validation_candidate(comparison_rows)
    summary_payload = {
        "purpose": (
            "Select a model and imbalance strategy from validation evidence, then "
            "describe its operational threshold trade-offs."
        ),
        "selection_metric": "validation_pr_auc",
        "test_selection_prohibited": True,
        "test_split_evaluated": False,
        "feature_set": "original_v1",
        "evidence_scope": evidence_scope,
        "frozen_reference_validation_pr_auc": FROZEN_REFERENCE_VALIDATION_PR_AUC,
        "provisional_relative_improvement_target": PROVISIONAL_RELATIVE_IMPROVEMENT_TARGET,
        "cost_values_are_sensitivity_assumptions": True,
        "candidate_results": comparison_rows,
        "selected_validation_candidate": selected,
        "next_decision": (
            "Review the validation winner and its capacity/cost table before defining "
            "a tuning search or authorising one final test evaluation."
        ),
    }

    with mlflow.start_run(run_name="model_optimization_operational_evaluation_summary") as summary_run:
        mlflow.set_tags(
            {
                "run_role": "validation_comparison_summary",
                "selection_metric": "validation_pr_auc",
                "test_selection_prohibited": "true",
                "test_split_used": "false",
                "evidence_scope": evidence_scope,
            }
        )
        mlflow.log_dict(summary_payload, "model_optimization_operational_evaluation_summary.json")
        mlflow.log_params(
            {
                "selected_model": selected["model"],
                "selected_imbalance_strategy": selected["imbalance_strategy"],
                "selected_run_id": selected["run_id"],
                "selection_basis": "highest_validation_pr_auc",
                "candidate_count": len(comparison_rows),
                "feature_set": "original_v1",
                **feature_lineage,
            }
        )
        mlflow.log_metrics(
            {
                "selected_validation_pr_auc": selected["validation_pr_auc"],
                "selected_relative_improvement": selected[
                    "relative_improvement_over_frozen_reference"
                ],
                "frozen_reference_validation_pr_auc": FROZEN_REFERENCE_VALIDATION_PR_AUC,
            }
        )

    print("Validation screening complete; the test split was not evaluated.")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print(f"Summary run: {summary_run.info.run_id}")
    print(
        "Selected from validation evidence: "
        f"{selected['model']} + {selected['imbalance_strategy']} "
        f"(Average Precision={selected['validation_pr_auc']:.6f})."
    )
    if args.sample_rows is not None:
        print("This was a smoke run and is not thesis evidence.")


# %% Script entry point
if __name__ == "__main__":
    main()
