"""Compare a small, EDA-justified set of fraud models with MLflow.

This script keeps the original cleaned feature set fixed and compares a compact
set of candidate models required for the thesis: a non-skill reference, an
interpretable linear model, and tree-based nonlinear models. KNN is available as
an explicit optional feasibility check, but it is not part of the default
full-data comparison because the dataset is large and mixed-type.
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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fraud_modeling_utils import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    configure_mlflow,
    dataset_metadata,
    fit_evaluate_and_log,
    load_gold_table,
    make_optional_sample,
    stratified_sample,
    time_aware_split,
    validate_split_fractions,
)


# %% Candidate model registry
CORE_MODELS = (
    "dummy_prior",
    "logistic_sgd",
    "random_forest_depth_12",
    "hist_gradient_boosting_leaves_31",
)
OPTIONAL_MODELS = (
    "decision_tree_depth_10",
    "knn_neighbors_31",
)
ALL_MODELS = CORE_MODELS + OPTIONAL_MODELS

MODEL_RATIONALE = {
    "dummy_prior": (
        "Non-skill reference; shows the minimum benchmark before learning "
        "relationships between predictors and fraud."
    ),
    "logistic_sgd": (
        "Interpretable linear fraud-risk baseline; useful because EDA does not "
        "show strong individual numeric correlations."
    ),
    "decision_tree_depth_10": (
        "Simple tree-based nonlinear comparison; tests whether threshold-like "
        "rules and interactions visible in EDA improve prediction."
    ),
    "random_forest_depth_12": (
        "Robust tree ensemble for nonlinear effects and feature interactions in "
        "mixed tabular fraud data."
    ),
    "hist_gradient_boosting_leaves_31": (
        "Stronger tabular benchmark for nonlinear patterns, using compact "
        "ordinal categorical preprocessing."
    ),
    "knn_neighbors_31": (
        "Optional distance-based feasibility check; sampled by default because "
        "full-data KNN is costly and mixed one-hot distances may be weak."
    ),
}


# %% Configuration container
@dataclass(frozen=True)
class ModelConfiguration:
    name: str
    model_family: str
    preprocessing: str
    parameters: dict[str, Any]


# %% Command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EDA-justified fraud candidate models with MLflow."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--experiment-name",
        default="financial-fraud-original-feature-candidate-models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(CORE_MODELS),
        help=(
            "Candidate models to compare. Defaults to dummy, logistic, random "
            "forest, and histogram gradient boosting."
        ),
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional stratified smoke sample applied before chronological splitting.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--knn-training-rows",
        type=int,
        default=50_000,
        help="Maximum training rows for the optional KNN feasibility run.",
    )
    parser.add_argument(
        "--knn-evaluation-rows",
        type=int,
        default=25_000,
        help="Maximum validation/test rows for the optional KNN feasibility run.",
    )
    parser.add_argument("--skip-data-hash", action="store_true")
    return parser.parse_args()


# %% Preprocessing factories
def scaled_one_hot_preprocessing() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
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


def unscaled_one_hot_preprocessing() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
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


def ordinal_categorical_preprocessing() -> ColumnTransformer:
    """Create a compact dense matrix for histogram boosting."""
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# %% Candidate configurations
def configurations() -> dict[str, ModelConfiguration]:
    return {
        "dummy_prior": ModelConfiguration(
            "dummy_prior",
            "non_skill",
            "scaled_one_hot",
            {},
        ),
        "logistic_sgd": ModelConfiguration(
            "logistic_sgd",
            "linear_logistic",
            "scaled_one_hot",
            {"alpha": 1e-4},
        ),
        "decision_tree_depth_10": ModelConfiguration(
            "decision_tree_depth_10",
            "single_tree",
            "unscaled_one_hot",
            {"max_depth": 10, "min_samples_leaf": 100},
        ),
        "random_forest_depth_12": ModelConfiguration(
            "random_forest_depth_12",
            "tree_ensemble",
            "unscaled_one_hot",
            {
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_leaf": 100,
                "max_features": "sqrt",
            },
        ),
        "hist_gradient_boosting_leaves_31": ModelConfiguration(
            "hist_gradient_boosting_leaves_31",
            "gradient_boosted_trees",
            "ordinal_categorical",
            {
                "learning_rate": 0.1,
                "max_iter": 150,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 100,
                "l2_regularization": 0.0,
            },
        ),
        "knn_neighbors_31": ModelConfiguration(
            "knn_neighbors_31",
            "distance_based",
            "scaled_one_hot",
            {"n_neighbors": 31, "weights": "distance"},
        ),
    }


# %% Model pipeline factory
def build_pipeline(config: ModelConfiguration, random_state: int) -> Pipeline:
    if config.preprocessing == "scaled_one_hot":
        preprocessing = scaled_one_hot_preprocessing()
    elif config.preprocessing == "unscaled_one_hot":
        preprocessing = unscaled_one_hot_preprocessing()
    elif config.preprocessing == "ordinal_categorical":
        preprocessing = ordinal_categorical_preprocessing()
    else:
        raise ValueError(f"Unknown preprocessing: {config.preprocessing}")

    if config.name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior")
    elif config.name == "logistic_sgd":
        estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            **config.parameters,
        )
    elif config.name == "decision_tree_depth_10":
        estimator = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
            **config.parameters,
        )
    elif config.name == "random_forest_depth_12":
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=random_state,
            **config.parameters,
        )
    elif config.name == "hist_gradient_boosting_leaves_31":
        categorical_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(
            CATEGORICAL_FEATURES
        )
        estimator = HistGradientBoostingClassifier(
            categorical_features=categorical_mask,
            class_weight="balanced",
            early_stopping=False,
            random_state=random_state,
            **config.parameters,
        )
    elif config.name == "knn_neighbors_31":
        estimator = KNeighborsClassifier(**config.parameters)
    else:
        raise ValueError(f"Unknown model configuration: {config.name}")

    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


# %% Experiment sampling and run metadata helpers
def experiment_frames(
    args: argparse.Namespace,
    config: ModelConfiguration,
    train_df,
    validation_df,
    test_df,
):
    if config.name != "knn_neighbors_31":
        return train_df, validation_df, test_df, "full_data"

    sampled_train = stratified_sample(
        train_df,
        min(args.knn_training_rows, len(train_df)),
        args.random_state,
    )
    sampled_validation = stratified_sample(
        validation_df,
        min(args.knn_evaluation_rows, len(validation_df)),
        args.random_state + 100,
    )
    sampled_test = stratified_sample(
        test_df,
        min(args.knn_evaluation_rows, len(test_df)),
        args.random_state + 200,
    )
    return sampled_train, sampled_validation, sampled_test, "sampled_feasibility"


def run_context(
    args: argparse.Namespace,
    config: ModelConfiguration,
    source_row_count: int,
    experiment_row_count: int,
    comparison_scope: str,
) -> dict[str, Any]:
    return {
        "workflow_stage": "candidate_model_comparison",
        "configuration_name": config.name,
        "model_family": config.model_family,
        "preprocessing_family": config.preprocessing,
        "model_rationale": MODEL_RATIONALE[config.name],
        "feature_set": "original_cleaned_features_only",
        "selection_metric": "validation_pr_auc",
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        "source_dataset_rows": source_row_count,
        "experiment_rows": experiment_row_count,
        "is_smoke_sample": args.sample_rows is not None,
        "comparison_scope": comparison_scope,
        "random_state": args.random_state,
    }


# %% Candidate-model comparison runner
def main() -> None:
    args = parse_args()
    validate_split_fractions(args.train_fraction, args.validation_fraction)
    if "knn_neighbors_31" in args.models and (
        args.knn_training_rows < 1_000 or args.knn_evaluation_rows < 1_000
    ):
        raise ValueError("KNN sample limits must each be at least 1,000 rows.")

    data_info = dataset_metadata(
        args.data_path, calculate_hash=not args.skip_data_hash
    )
    df = load_gold_table(args.data_path)
    source_row_count = len(df)
    df = make_optional_sample(df, args.sample_rows, args.random_state)
    train_df, validation_df, test_df = time_aware_split(
        df, args.train_fraction, args.validation_fraction
    )
    tracking_uri = configure_mlflow(args.experiment_name)
    source_files = [Path(__file__).resolve(), PROJECT_DIR / "fraud_modeling_utils.py"]

    config_by_name = configurations()
    selected_configs = [config_by_name[model_name] for model_name in args.models]

    print("Running EDA-justified candidate model comparison...")
    results = []
    for config in selected_configs:
        candidate_train, candidate_validation, candidate_test, scope = experiment_frames(
            args, config, train_df, validation_df, test_df
        )
        result = fit_evaluate_and_log(
            pipeline=build_pipeline(config, args.random_state),
            run_name=f"candidate__{config.name}",
            train_df=candidate_train,
            validation_df=candidate_validation,
            test_df=candidate_test,
            dataset_info=data_info,
            run_context=run_context(
                args,
                config,
                source_row_count,
                len(df),
                scope,
            ),
            source_files=source_files,
            tags={
                "test_usage": "final_candidate_evaluation",
                "run_role": "candidate_model",
                "comparison_scope": scope,
            },
            log_model=True,
        )
        results.append((config, result, scope))
        print(
            f"{config.name}: validation PR-AUC={result.validation_pr_auc:.6f}, "
            f"test PR-AUC={result.test_metrics['pr_auc']:.6f}, scope={scope}"
        )

    comparison_summary = []
    for config, result, scope in results:
        comparison_summary.append(
            {
                "model": config.name,
                "model_family": config.model_family,
                "rationale": MODEL_RATIONALE[config.name],
                "validation_pr_auc": result.validation_pr_auc,
                "test_pr_auc": result.test_metrics["pr_auc"],
                "test_precision": result.test_metrics["precision"],
                "test_recall": result.test_metrics["recall"],
                "test_f1": result.test_metrics["f1"],
                "test_roc_auc": result.test_metrics["roc_auc"],
                "selected_threshold": result.tuned_threshold,
                "run_id": result.run_id,
                "comparison_scope": scope,
            }
        )

    full_data_candidates = [
        row for row in comparison_summary if row["comparison_scope"] == "full_data"
    ]
    selection_pool = full_data_candidates or comparison_summary
    selection_scope = "full_data" if full_data_candidates else "sampled_feasibility"
    selected_overall = max(
        selection_pool,
        key=lambda row: row["validation_pr_auc"],
    )

    with mlflow.start_run(run_name="candidate_model_comparison_summary") as summary_run:
        mlflow.set_tags(
            {
                "run_role": "comparison_summary",
                "selection_metric": "validation_pr_auc",
                "test_selection_prohibited": "true",
            }
        )
        mlflow.log_dict(
            {
                "selection_metric": "validation_pr_auc",
                "model_candidates": comparison_summary,
                "model_rationale": MODEL_RATIONALE,
                "overall_selected_model": selected_overall,
            },
            "candidate_model_comparison_summary.json",
        )
        mlflow.log_params(
            {
                "overall_selected_model": selected_overall["model"],
                "overall_selected_model_family": selected_overall["model_family"],
                "overall_selection_basis": "validation_pr_auc_only",
                "overall_selection_scope": selection_scope,
                "default_models": ",".join(CORE_MODELS),
            }
        )
        mlflow.log_metric(
            "overall_selected_validation_pr_auc",
            selected_overall["validation_pr_auc"],
        )
        mlflow.log_metric(
            "overall_selected_test_pr_auc",
            selected_overall["test_pr_auc"],
        )

    print("MLflow candidate model comparison complete.")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Comparison summary run: {summary_run.info.run_id}")
    print(
        "Selected by validation PR-AUC: "
        f"{selected_overall['model']} "
        f"(validation={selected_overall['validation_pr_auc']:.6f}, "
        f"test={selected_overall['test_pr_auc']:.6f}, "
        f"scope={selection_scope})"
    )


# %% Script entry point
if __name__ == "__main__":
    main()
