"""Shared validation-only fitting and evaluation helpers.

The functions in this module cannot receive a test frame. This keeps model
comparison and tuning runners consistent while preserving the closed test set.
"""

# %% Imports
from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fraud_modeling_utils import best_f1_threshold, classification_metrics, predict_scores
from sender_location_history_utils import fixed_workload_table


# %% Shared class-weight calculation
def positive_class_weight(labels: pd.Series | np.ndarray) -> float:
    """Return the legitimate-to-fraud ratio from training labels only."""

    values = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("Training labels must be a one-dimensional binary array.")
    fraud = int(values.sum())
    legitimate = int(len(values) - fraud)
    if fraud == 0 or legitimate == 0:
        raise ValueError("Both training classes are required.")
    return legitimate / fraud


# %% Shared validation-only model evaluation
def fit_and_evaluate_validation(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    amount_column: str = "amount",
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Fit on training rows and return validation evidence and workloads."""

    x_train = train_df[list(feature_columns)]
    y_train = train_df[target_column]
    x_validation = validation_df[list(feature_columns)]
    y_validation = validation_df[target_column]

    fit_start = perf_counter()
    pipeline.fit(x_train, y_train)
    training_seconds = perf_counter() - fit_start

    inference_start = perf_counter()
    validation_scores, _, score_type = predict_scores(pipeline, x_validation)
    inference_seconds = perf_counter() - inference_start
    metrics, workloads = evaluate_validation_scores(y_validation, validation_scores, validation_df[amount_column])
    try:
        transformed_dimensions = int(pipeline.named_steps["preprocessing"].get_feature_names_out().size)
    except (AttributeError, ValueError):
        transformed_dimensions = -1

    metrics.update(
        {
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "transformed_feature_dimensions": transformed_dimensions,
            "training_seconds": float(training_seconds),
            "validation_inference_seconds": float(inference_seconds),
        }
    )
    return metrics, workloads, score_type


# %% Shared evaluation of validation scores
def evaluate_validation_scores(
    y_validation: pd.Series | np.ndarray, validation_scores: np.ndarray, amounts: pd.Series | np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate already computed validation scores without fitting a model."""

    workloads = fixed_workload_table(y_validation, validation_scores, amounts)
    if len(np.unique(y_validation)) != 2:
        raise ValueError("Validation labels must contain both binary classes.")
    threshold, _ = best_f1_threshold(y_validation, validation_scores)
    threshold_metrics = classification_metrics(y_validation, validation_scores, threshold)
    workload_five = workloads.loc[np.isclose(workloads["requested_alert_rate"], 0.05)].iloc[0]
    metrics = {
        "validation_average_precision": float(average_precision_score(y_validation, validation_scores)),
        "validation_roc_auc": float(roc_auc_score(y_validation, validation_scores)),
        "validation_precision_at_max_f1": float(threshold_metrics["precision"]),
        "validation_recall_at_max_f1": float(threshold_metrics["recall"]),
        "validation_f1": float(threshold_metrics["f1"]),
        "validation_threshold_at_max_f1": float(threshold),
        "validation_accuracy_at_max_f1": float(threshold_metrics["accuracy"]),
        "validation_balanced_accuracy_at_max_f1": float(threshold_metrics["balanced_accuracy"]),
        "validation_specificity_at_max_f1": float(threshold_metrics["specificity"]),
        "validation_true_negatives_at_max_f1": int(threshold_metrics["true_negatives"]),
        "validation_false_positives_at_max_f1": int(threshold_metrics["false_positives"]),
        "validation_false_negatives_at_max_f1": int(threshold_metrics["false_negatives"]),
        "validation_true_positives_at_max_f1": int(threshold_metrics["true_positives"]),
        "validation_alert_rate_at_max_f1": float(
            (threshold_metrics["true_positives"] + threshold_metrics["false_positives"]) / len(y_validation)
        ),
        "workload_5pct_fraud_count_recall": float(workload_five["fraud_count_recall"]),
        "workload_5pct_fraud_value_recall": float(workload_five["fraud_value_recall"]),
    }
    return metrics, workloads
