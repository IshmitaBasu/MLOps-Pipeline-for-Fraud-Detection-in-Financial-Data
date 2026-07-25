"""Train non-skill and logistic fraud baselines and track them with MLflow.

This script deliberately uses only the original cleaned feature set. It creates
two reference points: a prior-only DummyClassifier and a scalable regularized
logistic model. The compact candidate-model comparison lives in
05_candidate_model_comparison_with_mlflow.py.
"""

# %% Imports
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_modeling_utils import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    TARGET_COLUMN,
    configure_mlflow,
    dataset_metadata,
    fit_evaluate_and_log,
    load_gold_table,
    make_optional_sample,
    time_aware_split,
    validate_split_fractions,
)


# %% Baseline model registry
BASELINE_MODELS = ("dummy_prior", "logistic_sgd")


# %% Command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train original-feature fraud baselines and log them with MLflow."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the cleaned gold table.",
    )
    parser.add_argument(
        "--experiment-name",
        default="financial-fraud-original-feature-baseline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=list(BASELINE_MODELS),
        help="Baseline runs to execute. By default both baselines are run.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional stratified sample size for a smoke run. Omit for full data.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for smoke sampling and models that use randomness.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Oldest time-ordered share used for training.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Next time-ordered share used for validation.",
    )
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="Skip SHA-256 calculation only when a faster local smoke run is needed.",
    )
    return parser.parse_args()


# %% Shared preprocessing
def make_preprocessing() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


# %% Baseline pipeline factory
def make_baseline_pipeline(model_name: str, random_state: int) -> Pipeline:
    if model_name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior")
    elif model_name == "logistic_sgd":
        estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    return Pipeline(
        steps=[("preprocessing", make_preprocessing()), ("model", estimator)]
    )


# %% Baseline experiment runner
def main() -> None:
    args = parse_args()
    validate_split_fractions(args.train_fraction, args.validation_fraction)

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
    results = []
    for model_name in args.models:
        model_family = "non_skill" if model_name == "dummy_prior" else "linear_logistic"
        pipeline = make_baseline_pipeline(model_name, args.random_state)
        result = fit_evaluate_and_log(
            pipeline=pipeline,
            run_name=f"{model_name}_original_features",
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            dataset_info=data_info,
            run_context={
                "workflow_stage": "original_feature_baseline",
                "model_name": model_name,
                "model_family": model_family,
                "feature_set": "original_cleaned_features_only",
                "train_fraction": args.train_fraction,
                "validation_fraction": args.validation_fraction,
                "test_fraction": 1.0
                - args.train_fraction
                - args.validation_fraction,
                "source_dataset_rows": source_row_count,
                "experiment_rows": len(df),
                "is_smoke_sample": args.sample_rows is not None,
                "random_state": args.random_state,
            },
            source_files=source_files,
            tags={
                "baseline_type": "non_skill" if model_name == "dummy_prior" else "predictive",
                "test_usage": "final_baseline_evaluation",
            },
        )
        results.append((model_name, result))

    print("MLflow baseline runs complete.")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Features: {len(FEATURE_COLUMNS)} original cleaned predictors")
    for model_name, result in results:
        print(
            f"{model_name}: run={result.run_id}, "
            f"validation PR-AUC={result.validation_pr_auc:.6f}, "
            f"threshold={result.tuned_threshold:.6g}, "
            f"test PR-AUC={result.test_metrics['pr_auc']:.6f}"
        )


# %% Script entry point
if __name__ == "__main__":
    main()
