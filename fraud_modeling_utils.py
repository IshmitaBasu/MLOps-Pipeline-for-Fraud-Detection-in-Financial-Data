"""Shared, leakage-aware utilities for tracked fraud-model experiments."""

# %% Imports
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# %% Project paths and feature schema
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "gold_financial_fraud_detection_table.csv"
DEFAULT_FEATURE_REPO_PATH = PROJECT_DIR
DEFAULT_TRACKING_DB = PROJECT_DIR / "mlflow_tracking.db"
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "mlartifacts"

CATEGORICAL_FEATURES = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_used",
    "payment_channel",
]

NUMERIC_FEATURES = [
    "amount",
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]

LINEAGE_COLUMNS = ["transaction_id", "event_timestamp"]
TARGET_COLUMN = "is_fraud"
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = LINEAGE_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]


# %% Shared result container
@dataclass
class RunResult:
    run_id: str
    validation_pr_auc: float
    tuned_threshold: float
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int] | None


# %% Data loading and validation helpers
def validate_split_fractions(train_fraction: float, validation_fraction: float) -> None:
    test_fraction = 1.0 - train_fraction - validation_fraction
    if train_fraction <= 0 or validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive.")


def load_gold_table(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Gold table not found at {data_path}. " "Run 02_data_pipeline_preprocessing.py first.")

    header_columns = set(pd.read_csv(data_path, nrows=0).columns)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - header_columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df = pd.read_csv(data_path, usecols=REQUIRED_COLUMNS)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="raise", format="mixed")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")
    return df


def make_optional_sample(df: pd.DataFrame, sample_rows: int | None, random_state: int) -> pd.DataFrame:
    if sample_rows is None or sample_rows >= len(df):
        return df
    if sample_rows < 1_000:
        raise ValueError("Use at least 1,000 rows for a smoke sample.")
    return stratified_sample(df, sample_rows, random_state)


def stratified_sample(df: pd.DataFrame, sample_rows: int, random_state: int) -> pd.DataFrame:
    """Return a reproducible class-stratified sample without altering source rows."""
    if sample_rows >= len(df):
        return df.copy()
    if sample_rows < 2:
        raise ValueError("A stratified sample must contain at least two rows.")

    sample_parts: list[pd.DataFrame] = []
    allocated = 0
    grouped = list(df.groupby(TARGET_COLUMN, sort=True))
    for group_index, (target_value, group) in enumerate(grouped):
        if group_index == len(grouped) - 1:
            target_rows = sample_rows - allocated
        else:
            target_rows = max(1, int(round(sample_rows * len(group) / len(df))))
            target_rows = min(target_rows, sample_rows - allocated - 1)
        target_rows = min(target_rows, len(group))
        sample_parts.append(
            group.sample(
                n=target_rows,
                random_state=random_state + int(target_value),
                replace=False,
            )
        )
        allocated += target_rows

    sampled = pd.concat(sample_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def time_aware_split(
    df: pd.DataFrame, train_fraction: float, validation_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("event_timestamp", kind="mergesort").reset_index(drop=True)
    train_end = int(len(df_sorted) * train_fraction)
    validation_end = int(len(df_sorted) * (train_fraction + validation_fraction))

    train_df = df_sorted.iloc[:train_end].copy()
    validation_df = df_sorted.iloc[train_end:validation_end].copy()
    test_df = df_sorted.iloc[validation_end:].copy()

    for split_name, split_df in {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():
        if split_df.empty:
            raise ValueError(f"The {split_name} split is empty.")
        if split_df[TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"The {split_name} split contains only one target class.")

    return train_df, validation_df, test_df


# %% MLflow setup and reproducibility metadata
def configure_mlflow(experiment_name: str) -> str:
    tracking_uri = f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}"
    artifact_location = DEFAULT_ARTIFACT_DIR.as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )

    mlflow.set_experiment(experiment_name)
    return tracking_uri


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_metadata(data_path: Path, calculate_hash: bool = True) -> dict[str, Any]:
    stat = data_path.stat()
    return {
        "dataset_name": data_path.name,
        "dataset_absolute_path": str(data_path.resolve()),
        "dataset_size_bytes": int(stat.st_size),
        "dataset_modified_time_ns": int(stat.st_mtime_ns),
        "dataset_sha256": sha256_file(data_path) if calculate_hash else "not_calculated",
    }


def load_modeling_table(
    *,
    data_source: str,
    data_path: Path,
    feature_repo_path: Path,
    calculate_hash: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str | int | float | bool]]:
    """Load the canonical Feast table or an explicitly requested CSV fallback."""
    if data_source == "feast":
        from fraud_feature_store import load_feast_training_table

        return load_feast_training_table(
            feature_repo_path,
            validate_hashes=calculate_hash,
        )
    if data_source == "csv":
        table = load_gold_table(data_path)
        return (
            table,
            dataset_metadata(data_path, calculate_hash=calculate_hash),
            {
                "data_interface": "direct_csv_fallback",
                "feature_store": "not_used",
                "feature_version": "not_applicable",
                "feature_hash_validation_enabled": calculate_hash,
            },
        )
    raise ValueError(f"Unknown data source: {data_source}")


def split_summary(split_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(split_df)),
        "fraud_count": int(split_df[TARGET_COLUMN].sum()),
        "fraud_rate": float(split_df[TARGET_COLUMN].mean()),
        "start_timestamp": split_df["event_timestamp"].min().isoformat(),
        "end_timestamp": split_df["event_timestamp"].max().isoformat(),
    }


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for package in [
        "mlflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "scipy",
        "matplotlib",
        "cloudpickle",
    ]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def scalar_pipeline_params(pipeline: Pipeline) -> dict[str, str | int | float | bool]:
    logged: dict[str, str | int | float | bool] = {}
    for key, value in pipeline.get_params(deep=True).items():
        if value is None:
            logged[f"pipeline.{key}"] = "None"
        elif isinstance(value, (str, int, float, bool)):
            logged[f"pipeline.{key}"] = value
    return logged


# %% Scoring and metric helpers
def predict_scores(model: Pipeline, features: pd.DataFrame) -> tuple[np.ndarray, float, str]:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1]), 0.5, "probability"
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features)), 0.0, "decision_function"
    raise TypeError("The fitted model must provide predict_proba or decision_function.")


def best_f1_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return float(y_score[0]), 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def classification_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def log_metrics(prefix: str, metrics: dict[str, float | int]) -> None:
    mlflow.log_metrics({f"{prefix}_{key}": value for key, value in metrics.items()})


# %% Evaluation artifact helpers
def _downsample_curve(*arrays: np.ndarray, maximum_points: int = 2_000) -> list[np.ndarray]:
    if len(arrays[0]) <= maximum_points:
        return list(arrays)
    indices = np.linspace(0, len(arrays[0]) - 1, maximum_points, dtype=int)
    return [array[indices] for array in arrays]


def save_evaluation_plots(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path,
    prefix: str,
) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    precision_plot, recall_plot = _downsample_curve(precision, recall)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_plot, precision_plot)
    ax.axhline(float(y_true.mean()), color="grey", linestyle="--", label="Fraud prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{prefix}: precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fpr_plot, tpr_plot = _downsample_curve(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_plot, tpr_plot)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{prefix}: ROC curve")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=150)
    plt.close(fig)

    predicted = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(f"{prefix}: confusion matrix at threshold {threshold:.6g}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    if thresholds.size:
        f1_values = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(thresholds),
            where=(precision[:-1] + recall[:-1]) > 0,
        )
        threshold_plot, precision_threshold, recall_threshold, f1_plot = _downsample_curve(
            thresholds, precision[:-1], recall[:-1], f1_values
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(threshold_plot, precision_threshold, label="Precision")
        ax.plot(threshold_plot, recall_threshold, label="Recall")
        ax.plot(threshold_plot, f1_plot, label="F1")
        ax.axvline(threshold, color="black", linestyle="--", label="Selected threshold")
        ax.set(xlabel="Decision threshold", ylabel="Metric", title=f"{prefix}: threshold trade-off")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_threshold_tradeoff.png", dpi=150)
        plt.close(fig)


def save_model_explanation(pipeline: Pipeline, output_dir: Path) -> str | None:
    estimator = pipeline.named_steps["model"]
    preprocessing = pipeline.named_steps["preprocessing"]
    try:
        feature_names = preprocessing.get_feature_names_out()
    except (AttributeError, ValueError):
        return None

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        elif values.ndim != 1:
            return None
        column_name = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        column_name = "feature_importance"
    else:
        return None

    if len(feature_names) != len(values):
        return None
    explanation = pd.DataFrame({"feature": feature_names, column_name: values})
    explanation["absolute_value"] = explanation[column_name].abs()
    explanation = explanation.sort_values("absolute_value", ascending=False)
    artifact_name = "model_coefficients.csv" if column_name == "coefficient" else "feature_importances.csv"
    explanation.to_csv(output_dir / artifact_name, index=False)
    return artifact_name


# %% Common experiment logging
def log_common_metadata(
    *,
    pipeline: Pipeline,
    dataset_info: dict[str, Any],
    run_context: dict[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    source_files: Iterable[Path],
    feature_columns: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
    feature_metadata: dict[str, Any] | None,
) -> None:
    combined_params = {**dataset_info, **run_context, **scalar_pipeline_params(pipeline)}
    mlflow.log_params(combined_params)
    mlflow.log_dict(environment_metadata(), "reproducibility/environment.json")
    feature_record: dict[str, Any] = {
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "excluded_columns": LINEAGE_COLUMNS + [TARGET_COLUMN],
    }
    if feature_metadata:
        feature_record.update(feature_metadata)
    elif feature_columns == FEATURE_COLUMNS:
        feature_record["postponed_feature_experiments"] = [
            "amount_log1p",
            "timestamp_hour_weekday_month",
            "time_since_last_transaction_missing_indicator",
        ]
    mlflow.log_dict(feature_record, "reproducibility/feature_set.json")
    summaries: dict[str, Any] = {
        "train": split_summary(train_df),
        "validation": split_summary(validation_df),
    }
    if test_df is not None:
        summaries["test"] = split_summary(test_df)
    mlflow.log_dict(summaries, "reproducibility/split_summary.json")
    for source_file in source_files:
        mlflow.log_artifact(str(source_file), artifact_path="reproducibility/source")


# %% Fit, evaluate, and log one MLflow run
def fit_evaluate_and_log(
    *,
    pipeline: Pipeline,
    run_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    dataset_info: dict[str, Any],
    run_context: dict[str, Any],
    source_files: Iterable[Path],
    tags: dict[str, str] | None = None,
    log_model: bool = True,
    feature_columns: list[str] | None = None,
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
    feature_metadata: dict[str, Any] | None = None,
) -> RunResult:
    """Fit a pipeline inside an MLflow run and log reproducible evaluation evidence."""
    selected_features = list(FEATURE_COLUMNS if feature_columns is None else feature_columns)
    selected_categorical = list(CATEGORICAL_FEATURES if categorical_features is None else categorical_features)
    selected_numeric = list(NUMERIC_FEATURES if numeric_features is None else numeric_features)
    declared_features = selected_categorical + selected_numeric

    if len(selected_features) != len(set(selected_features)):
        raise ValueError("The feature list contains duplicate column names.")
    if set(declared_features) != set(selected_features):
        raise ValueError("Categorical and numeric feature declarations must match the selected feature columns.")

    x_train = train_df[selected_features]
    y_train = train_df[TARGET_COLUMN]
    x_validation = validation_df[selected_features]
    y_validation = validation_df[TARGET_COLUMN]

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        log_common_metadata(
            pipeline=pipeline,
            dataset_info=dataset_info,
            run_context=run_context,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            source_files=source_files,
            feature_columns=selected_features,
            categorical_features=selected_categorical,
            numeric_features=selected_numeric,
            feature_metadata=feature_metadata,
        )

        training_start = perf_counter()
        pipeline.fit(x_train, y_train)
        training_seconds = perf_counter() - training_start

        validation_start = perf_counter()
        validation_scores, default_threshold, score_type = predict_scores(pipeline, x_validation)
        validation_inference_seconds = perf_counter() - validation_start
        tuned_threshold, validation_best_f1 = best_f1_threshold(y_validation, validation_scores)
        validation_default = classification_metrics(y_validation, validation_scores, default_threshold)
        validation_tuned = classification_metrics(y_validation, validation_scores, tuned_threshold)

        mlflow.log_params(
            {
                "score_type": score_type,
                "threshold_default": default_threshold,
                "threshold_tuned_on_validation": tuned_threshold,
            }
        )
        mlflow.log_metrics(
            {
                "training_seconds": training_seconds,
                "validation_inference_seconds": validation_inference_seconds,
                "validation_inference_rows_per_second": (
                    len(validation_df) / validation_inference_seconds if validation_inference_seconds > 0 else 0.0
                ),
                "validation_best_f1_from_threshold_search": validation_best_f1,
            }
        )
        log_metrics("validation_default", validation_default)
        log_metrics("validation_tuned", validation_tuned)

        test_metrics: dict[str, float | int] | None = None
        test_scores: np.ndarray | None = None
        if test_df is not None:
            test_start = perf_counter()
            test_scores, _, _ = predict_scores(pipeline, test_df[selected_features])
            test_inference_seconds = perf_counter() - test_start
            test_metrics = classification_metrics(test_df[TARGET_COLUMN], test_scores, tuned_threshold)
            mlflow.log_metrics(
                {
                    "test_inference_seconds": test_inference_seconds,
                    "test_inference_rows_per_second": (
                        len(test_df) / test_inference_seconds if test_inference_seconds > 0 else 0.0
                    ),
                }
            )
            log_metrics("test_tuned", test_metrics)

        metrics_summary: dict[str, Any] = {
            "validation_default_threshold": validation_default,
            "validation_tuned_threshold": validation_tuned,
        }
        if test_metrics is not None:
            metrics_summary["test_tuned_threshold"] = test_metrics
        mlflow.log_dict(metrics_summary, "evaluation/metrics_summary.json")

        with TemporaryDirectory(prefix="fraud_model_evaluation_") as temporary_dir:
            output_dir = Path(temporary_dir)
            save_evaluation_plots(
                y_validation,
                validation_scores,
                tuned_threshold,
                output_dir,
                "validation",
            )
            if test_df is not None and test_scores is not None:
                save_evaluation_plots(
                    test_df[TARGET_COLUMN],
                    test_scores,
                    tuned_threshold,
                    output_dir,
                    "test",
                )
            save_model_explanation(pipeline, output_dir)
            for artifact in output_dir.iterdir():
                mlflow.log_artifact(str(artifact), artifact_path="evaluation")

        if log_model:
            input_example = x_train.head(5).copy()
            if hasattr(pipeline, "predict_proba"):
                example_output = pipeline.predict_proba(input_example)
            else:
                example_output = pipeline.decision_function(input_example)
            signature = infer_signature(input_example, example_output)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )

        return RunResult(
            run_id=run.info.run_id,
            validation_pr_auc=float(validation_default["pr_auc"]),
            tuned_threshold=tuned_threshold,
            validation_metrics=validation_tuned,
            test_metrics=test_metrics,
        )


# %% Local summary writer
def write_json(path: Path, payload: Any) -> None:
    """Small helper used for local summaries produced by comparison scripts."""
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
