"""Compare model and imbalance choices, then study usable alert thresholds.

Random Forest and Histogram Gradient Boosting are repeated as frozen controls.
The new evidence in this stage comes from comparing a selected external booster,
training-only imbalance strategies, and validation-based alert/cost scenarios.

This runner never evaluates the test split. A separate final step
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
    RandomForestTuningConfiguration,
    SUPPORTED_IMBALANCE_STRATEGIES,
    SUPPORTED_MODELS,
    apply_training_imbalance_strategy,
    build_candidate_pipeline,
    build_random_forest_tuning_pipeline,
    build_threshold_summary,
    model_optimization_configurations,
    positive_class_weight,
    random_forest_tuning_configurations,
    relative_validation_improvement,
    select_validation_candidate,
)


# %% Fixed experiment controls
FROZEN_REFERENCE_VALIDATION_PR_AUC = 0.04395981671344107
FIVE_TO_ONE_REFERENCE_VALIDATION_PR_AUC = 0.044226770836638986
PROVISIONAL_RELATIVE_IMPROVEMENT_TARGET = 0.05
PROVISIONAL_ACCEPTANCE_VALIDATION_PR_AUC = (
    FROZEN_REFERENCE_VALIDATION_PR_AUC
    * (1.0 + PROVISIONAL_RELATIVE_IMPROVEMENT_TARGET)
)
TUNING_IMBALANCE_STRATEGY = "undersample_5_to_1"
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
        "--mode",
        choices=("screening", "tuning"),
        default="screening",
        help=(
            "Run the existing model/imbalance screening workflow or the fixed "
            "Random Forest 5:1 tuning search."
        ),
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
        default=None,
        help=(
            "Candidates to compare. The default reruns the two frozen controls; "
            "add either xgboost or lightgbm after choosing and installing one."
        ),
    )
    parser.add_argument(
        "--imbalance-strategies",
        nargs="+",
        choices=SUPPORTED_IMBALANCE_STRATEGIES,
        default=None,
        help=(
            "Training-only imbalance methods. Multiple choices create a model-by-"
            "strategy comparison matrix."
        ),
    )
    parser.add_argument(
        "--tuning-configurations",
        nargs="+",
        choices=tuple(random_forest_tuning_configurations()),
        default=None,
        help=(
            "Optional subset of the nine predeclared tuning configurations. "
            "Valid only with --mode tuning; the default runs all nine."
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
    if args.mode == "tuning":
        if args.models is not None or args.imbalance_strategies is not None:
            raise ValueError(
                "Tuning always uses Random Forest with 5:1 undersampling; do not "
                "pass --models or --imbalance-strategies with --mode tuning."
            )
        selected_tuning = args.tuning_configurations or list(
            random_forest_tuning_configurations()
        )
        if len(selected_tuning) != len(set(selected_tuning)):
            raise ValueError("Each tuning configuration should be listed only once.")
        return

    if args.tuning_configurations is not None:
        raise ValueError("--tuning-configurations is valid only with --mode tuning.")
    models = args.models or list(CONTROL_MODELS)
    strategies = args.imbalance_strategies or ["class_weight"]
    if len(models) != len(set(models)):
        raise ValueError("Each model should be listed only once.")
    if len(strategies) != len(set(strategies)):
        raise ValueError("Each imbalance strategy should be listed only once.")
    selected_external_models = set(models).intersection(EXTERNAL_MODELS)
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
        PROJECT_DIR
        / "Markdown files"
        / "06_model_optimization_and_operational_evaluation_implementation_guide.md",
        PROJECT_DIR
        / "Markdown files"
        / "06_model_optimization_and_operational_evaluation_results.md",
        PROJECT_DIR
        / "tests"
        / "test_model_optimization_and_operational_evaluation.py",
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


def tuning_context(
    args: argparse.Namespace,
    *,
    configuration: RandomForestTuningConfiguration,
    imbalance_metadata: dict[str, int | float | bool | str],
    source_rows: int,
    experiment_rows: int,
) -> dict[str, Any]:
    context = experiment_context(
        args,
        model_name="random_forest_depth_12",
        imbalance_metadata=imbalance_metadata,
        source_rows=source_rows,
        experiment_rows=experiment_rows,
    )
    return {
        **context,
        "workflow_stage": "random_forest_5_to_1_tuning",
        "experiment_scope": "validation_tuning_only",
        "configuration_name": configuration.name,
        "configuration_description": configuration.description,
        "tuning_reference_validation_pr_auc": FIVE_TO_ONE_REFERENCE_VALIDATION_PR_AUC,
        "provisional_acceptance_validation_pr_auc": PROVISIONAL_ACCEPTANCE_VALIDATION_PR_AUC,
        **{
            f"tuning.{parameter_name}": parameter_value
            for parameter_name, parameter_value in configuration.parameters.items()
        },
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


# %% Fixed Random Forest tuning workflow
def run_tuning_experiments(
    *,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    dataset_info: dict[str, Any],
    feature_lineage: dict[str, Any],
    source_files: list[Path],
    source_rows: int,
    experiment_rows: int,
    evidence_scope: str,
) -> tuple[dict[str, Any], str]:
    """Tune Random Forest with 5:1 training data without accepting a test frame."""

    effective_train_df, imbalance_metadata = apply_training_imbalance_strategy(
        train_df,
        TUNING_IMBALANCE_STRATEGY,
        random_state=args.random_state,
    )
    all_configurations = random_forest_tuning_configurations()
    selected_names = args.tuning_configurations or list(all_configurations)
    comparison_rows: list[dict[str, Any]] = []
    is_smoke_run = args.sample_rows is not None

    print("Running the fixed Random Forest 5:1 tuning search on validation data only.")
    print(
        f"Effective training rows after 5:1 undersampling: "
        f"{len(effective_train_df):,}."
    )

    for configuration_name in selected_names:
        configuration = all_configurations[configuration_name]
        pipeline = build_random_forest_tuning_pipeline(
            configuration,
            random_state=args.random_state,
        )
        result = fit_evaluate_and_log(
            pipeline=pipeline,
            run_name=f"tuning__{configuration.name}__5_to_1",
            train_df=effective_train_df,
            validation_df=validation_df,
            test_df=None,
            dataset_info=dataset_info,
            run_context={
                **tuning_context(
                    args,
                    configuration=configuration,
                    imbalance_metadata=imbalance_metadata,
                    source_rows=source_rows,
                    experiment_rows=experiment_rows,
                ),
                **feature_lineage,
            },
            source_files=source_files,
            tags={
                "run_role": "tuning_reference"
                if configuration.name == "rf_reference"
                else "tuning_candidate",
                "model_family": "random_forest",
                "imbalance_strategy": TUNING_IMBALANCE_STRATEGY,
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

        improvement_over_tuning_reference = (
            None
            if is_smoke_run
            else relative_validation_improvement(
                result.validation_pr_auc,
                FIVE_TO_ONE_REFERENCE_VALIDATION_PR_AUC,
            )
        )
        improvement_over_frozen_reference = (
            None
            if is_smoke_run
            else relative_validation_improvement(
                result.validation_pr_auc,
                FROZEN_REFERENCE_VALIDATION_PR_AUC,
            )
        )
        comparison_rows.append(
            {
                "model": configuration.name,
                "model_family": "random_forest",
                "imbalance_strategy": TUNING_IMBALANCE_STRATEGY,
                "description": configuration.description,
                "parameters": dict(configuration.parameters),
                "validation_pr_auc": result.validation_pr_auc,
                "validation_precision_at_max_f1": result.validation_metrics["precision"],
                "validation_recall_at_max_f1": result.validation_metrics["recall"],
                "validation_f1": result.validation_metrics["f1"],
                "validation_roc_auc": result.validation_metrics["roc_auc"],
                "relative_improvement_over_5_to_1_reference": (
                    improvement_over_tuning_reference
                ),
                "relative_improvement_over_frozen_reference": (
                    improvement_over_frozen_reference
                ),
                "meets_provisional_acceptance_target": (
                    None
                    if is_smoke_run
                    else result.validation_pr_auc
                    >= PROVISIONAL_ACCEPTANCE_VALIDATION_PR_AUC
                ),
                "reference_comparison_valid": not is_smoke_run,
                "threshold_scenarios": threshold_records(thresholds),
                "run_id": result.run_id,
            }
        )
        print(
            f"{configuration.name}: validation Average Precision="
            f"{result.validation_pr_auc:.6f}, run={result.run_id}"
        )

    highest_scoring_candidate = select_validation_candidate(comparison_rows)
    decision_status = (
        "technical_smoke_no_selection"
        if is_smoke_run
        else "selected_by_full_validation_pr_auc"
    )
    candidate_record_key = (
        "highest_scoring_smoke_configuration"
        if is_smoke_run
        else "selected_validation_candidate"
    )
    summary_payload = {
        "purpose": (
            "Tune only the selected Random Forest and 5:1 undersampling "
            "combination using validation evidence."
        ),
        "search_type": "fixed_predeclared_configuration_list",
        "selection_metric": "validation_pr_auc",
        "test_selection_prohibited": True,
        "test_split_evaluated": False,
        "feature_set": "original_v1",
        "imbalance_strategy": TUNING_IMBALANCE_STRATEGY,
        "evidence_scope": evidence_scope,
        "decision_status": decision_status,
        "tuning_reference_validation_pr_auc": FIVE_TO_ONE_REFERENCE_VALIDATION_PR_AUC,
        "frozen_reference_validation_pr_auc": FROZEN_REFERENCE_VALIDATION_PR_AUC,
        "provisional_acceptance_validation_pr_auc": PROVISIONAL_ACCEPTANCE_VALIDATION_PR_AUC,
        "candidate_results": comparison_rows,
        candidate_record_key: highest_scoring_candidate,
        "next_decision": (
            "Run the unchanged predeclared search on full data; smoke scores do not select a configuration."
            if is_smoke_run
            else "Freeze one configuration and operating threshold before any final test evaluation."
        ),
    }

    with mlflow.start_run(run_name="random_forest_5_to_1_tuning_summary") as summary_run:
        mlflow.set_tags(
            {
                "run_role": "validation_tuning_summary",
                "selection_metric": "validation_pr_auc",
                "test_selection_prohibited": "true",
                "test_split_used": "false",
                "evidence_scope": evidence_scope,
            }
        )
        mlflow.log_dict(summary_payload, "random_forest_5_to_1_tuning_summary.json")
        mlflow.log_params(
            {
                "reported_configuration": highest_scoring_candidate["model"],
                "reported_run_id": highest_scoring_candidate["run_id"],
                "decision_status": decision_status,
                "selection_basis": (
                    "none_smoke_test_only"
                    if is_smoke_run
                    else "highest_validation_pr_auc"
                ),
                "candidate_count": len(comparison_rows),
                "model_family": "random_forest",
                "imbalance_strategy": TUNING_IMBALANCE_STRATEGY,
                "feature_set": "original_v1",
                **feature_lineage,
            }
        )
        summary_metrics = {
            "highest_candidate_validation_pr_auc": highest_scoring_candidate[
                "validation_pr_auc"
            ],
            "tuning_reference_validation_pr_auc": FIVE_TO_ONE_REFERENCE_VALIDATION_PR_AUC,
            "provisional_acceptance_validation_pr_auc": (
                PROVISIONAL_ACCEPTANCE_VALIDATION_PR_AUC
            ),
        }
        if not is_smoke_run:
            summary_metrics.update(
                {
                    "selected_relative_improvement_over_5_to_1_reference": (
                        highest_scoring_candidate[
                            "relative_improvement_over_5_to_1_reference"
                        ]
                    ),
                    "selected_relative_improvement_over_frozen_reference": (
                        highest_scoring_candidate[
                            "relative_improvement_over_frozen_reference"
                        ]
                    ),
                }
            )
        mlflow.log_metrics(summary_metrics)

    print("Validation-only tuning complete; the test split was not evaluated.")
    print(f"Tuning summary run: {summary_run.info.run_id}")
    if is_smoke_run:
        print(
            "Highest smoke score (not a selection): "
            f"{highest_scoring_candidate['model']} "
            f"(Average Precision={highest_scoring_candidate['validation_pr_auc']:.6f})."
        )
        print("This was a tuning smoke run and is not thesis evidence.")
    else:
        print(
            "Selected tuning configuration: "
            f"{highest_scoring_candidate['model']} "
            f"(Average Precision={highest_scoring_candidate['validation_pr_auc']:.6f})."
        )
    return highest_scoring_candidate, summary_run.info.run_id


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

    print("Running validation-only model optimisation and operational evaluation.")
    print(f"The held-out test split contains {len(held_out_test_df):,} rows and will not be evaluated.")

    if args.mode == "tuning":
        run_tuning_experiments(
            args=args,
            train_df=train_df,
            validation_df=validation_df,
            dataset_info=dataset_info,
            feature_lineage=feature_lineage,
            source_files=source_files,
            source_rows=source_rows,
            experiment_rows=len(modeling_table),
            evidence_scope=evidence_scope,
        )
        print(f"Tracking URI: {tracking_uri}")
        print(f"Experiment: {experiment_name}")
        return

    selected_models = args.models or list(CONTROL_MODELS)
    selected_strategies = args.imbalance_strategies or ["class_weight"]
    comparison_rows: list[dict[str, Any]] = []
    for model_name in selected_models:
        for strategy in selected_strategies:
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
