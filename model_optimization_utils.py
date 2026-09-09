"""Reusable building blocks for model optimisation and operational evaluation.

The module deliberately does not load data, start MLflow runs, fit models, or
inspect the test split.  Those orchestration tasks belong in
``06_model_optimization_and_operational_evaluation_with_mlflow.py``. Keeping
this file focused on
pure construction and calculation makes the experiment rules easier to test.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from fraud_modeling_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    best_f1_threshold,
    classification_metrics,
)


DEFAULT_RANDOM_STATE = 42
DEFAULT_ALERT_RATES = (0.01, 0.05, 0.10)
SUPPORTED_MODELS = (
    "random_forest_depth_12",
    "hist_gradient_boosting_leaves_31",
    "xgboost",
    "lightgbm",
)
SUPPORTED_IMBALANCE_STRATEGIES = (
    "class_weight",
    "undersample_10_to_1",
    "undersample_5_to_1",
)


@dataclass(frozen=True)
class ModelOptimizationConfiguration:
    """One documented, untuned candidate used in the family comparison."""

    name: str
    model_family: str
    preprocessing: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class CostScenario:
    """Relative cost assumptions for threshold sensitivity analysis."""

    name: str
    false_positive_cost: float = 1.0
    false_negative_cost: float = 10.0

    def __post_init__(self) -> None:
        if self.false_positive_cost < 0 or self.false_negative_cost < 0:
            raise ValueError("False-positive and false-negative costs must be non-negative.")


DEFAULT_COST_SCENARIOS = (
    CostScenario("fn_cost_10", false_positive_cost=1.0, false_negative_cost=10.0),
    CostScenario("fn_cost_50", false_positive_cost=1.0, false_negative_cost=50.0),
    CostScenario("fn_cost_100", false_positive_cost=1.0, false_negative_cost=100.0),
)


def model_optimization_configurations() -> dict[str, ModelOptimizationConfiguration]:
    """Return deterministic starting configurations, before model tuning."""

    return {
        "random_forest_depth_12": ModelOptimizationConfiguration(
            name="random_forest_depth_12",
            model_family="random_forest",
            preprocessing="one_hot",
            parameters={
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_leaf": 100,
                "max_features": "sqrt",
            },
        ),
        "hist_gradient_boosting_leaves_31": ModelOptimizationConfiguration(
            name="hist_gradient_boosting_leaves_31",
            model_family="hist_gradient_boosting",
            preprocessing="ordinal",
            parameters={
                "learning_rate": 0.1,
                "max_iter": 150,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 100,
                "l2_regularization": 0.0,
            },
        ),
        "xgboost": ModelOptimizationConfiguration(
            name="xgboost",
            model_family="external_gradient_boosting",
            preprocessing="one_hot",
            parameters={
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        ),
        "lightgbm": ModelOptimizationConfiguration(
            name="lightgbm",
            model_family="external_gradient_boosting",
            preprocessing="one_hot",
            parameters={
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 100,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        ),
    }


def one_hot_preprocessing() -> ColumnTransformer:
    """Impute numeric values and one-hot encode categorical predictors."""

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


def ordinal_preprocessing() -> ColumnTransformer:
    """Create the compact dense representation used by histogram boosting."""

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


def positive_class_weight(y_train: Sequence[int] | pd.Series | np.ndarray) -> float:
    """Return the legitimate-to-fraud ratio used by weighted boosters."""

    labels = _validated_binary_labels(y_train)
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    return negative_count / positive_count


def build_candidate_pipeline(
    model: str | ModelOptimizationConfiguration,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    use_class_weight: bool = True,
    external_positive_class_weight: float | None = None,
) -> Pipeline:
    """Build one optimisation candidate without fitting or importing unused libraries.

    XGBoost and LightGBM are optional.  Their modules are imported only when the
    corresponding candidate is requested, allowing the project to remain usable
    until one library has been selected and pinned.
    """

    if isinstance(model, str):
        try:
            config = model_optimization_configurations()[model]
        except KeyError as error:
            raise ValueError(f"Unknown model {model!r}. Choose from {SUPPORTED_MODELS}.") from error
    else:
        config = model

    if config.preprocessing == "one_hot":
        preprocessing = one_hot_preprocessing()
    elif config.preprocessing == "ordinal":
        preprocessing = ordinal_preprocessing()
    else:
        raise ValueError(f"Unknown preprocessing family: {config.preprocessing!r}")

    parameters = dict(config.parameters)
    if config.name == "random_forest_depth_12":
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample" if use_class_weight else None,
            random_state=random_state,
            **parameters,
        )
    elif config.name == "hist_gradient_boosting_leaves_31":
        categorical_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
        estimator = HistGradientBoostingClassifier(
            categorical_features=categorical_mask,
            class_weight="balanced" if use_class_weight else None,
            early_stopping=False,
            random_state=random_state,
            **parameters,
        )
    elif config.name == "xgboost":
        xgboost = _import_optional_model_library("xgboost", "XGBoost")
        estimator = xgboost.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=_external_weight(use_class_weight, external_positive_class_weight),
            random_state=random_state,
            n_jobs=-1,
            **parameters,
        )
    elif config.name == "lightgbm":
        lightgbm = _import_optional_model_library("lightgbm", "LightGBM")
        estimator = lightgbm.LGBMClassifier(
            objective="binary",
            scale_pos_weight=_external_weight(use_class_weight, external_positive_class_weight),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            **parameters,
        )
    else:
        raise ValueError(f"Unknown model configuration: {config.name!r}")

    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def _external_weight(use_class_weight: bool, supplied_weight: float | None) -> float:
    if not use_class_weight:
        return 1.0
    if supplied_weight is None or supplied_weight <= 0:
        raise ValueError(
            "A positive external_positive_class_weight is required for weighted "
            "XGBoost or LightGBM. Calculate it from the training labels only."
        )
    return float(supplied_weight)


def _import_optional_model_library(module_name: str, display_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{display_name} is not installed. Select the external booster first, "
            "then pin and install it from requirements.txt."
        ) from error


def undersample_training_frame(
    train_df: pd.DataFrame,
    majority_to_minority_ratio: float,
    *,
    target_column: str = TARGET_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, int | float | bool | str]]:
    """Randomly undersample legitimate rows while retaining every fraud row.

    The function is intentionally named for, and accepts, a training frame.
    Validation and test frames should never be passed through it.
    """

    if majority_to_minority_ratio <= 0:
        raise ValueError("majority_to_minority_ratio must be positive.")
    if target_column not in train_df.columns:
        raise ValueError(f"Training frame does not contain target column {target_column!r}.")

    labels = _validated_binary_labels(train_df[target_column])
    minority = train_df.loc[labels == 1]
    majority = train_df.loc[labels == 0]
    requested_majority_count = int(np.floor(len(minority) * majority_to_minority_ratio))
    retained_majority_count = min(len(majority), requested_majority_count)
    retained_majority = majority.sample(
        n=retained_majority_count,
        replace=False,
        random_state=random_state,
    )
    sampled = pd.concat([retained_majority, minority], axis=0)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    metadata: dict[str, int | float | bool | str] = {
        "strategy": f"random_undersampling_{majority_to_minority_ratio:g}_to_1",
        "sampling_applied": retained_majority_count < len(majority),
        "majority_to_minority_ratio_requested": float(majority_to_minority_ratio),
        "original_rows": int(len(train_df)),
        "original_majority_rows": int(len(majority)),
        "original_minority_rows": int(len(minority)),
        "effective_rows": int(len(sampled)),
        "effective_majority_rows": int(retained_majority_count),
        "effective_minority_rows": int(len(minority)),
        "effective_fraud_rate": float(sampled[target_column].mean()),
    }
    return sampled, metadata


def apply_training_imbalance_strategy(
    train_df: pd.DataFrame,
    strategy: str,
    *,
    target_column: str = TARGET_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, int | float | bool | str]]:
    """Apply one named imbalance strategy to the training split only."""

    if strategy not in SUPPORTED_IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy {strategy!r}. "
            f"Choose from {SUPPORTED_IMBALANCE_STRATEGIES}."
        )
    if target_column not in train_df.columns:
        raise ValueError(f"Training frame does not contain target column {target_column!r}.")
    if strategy == "undersample_10_to_1":
        return undersample_training_frame(
            train_df,
            10.0,
            target_column=target_column,
            random_state=random_state,
        )
    if strategy == "undersample_5_to_1":
        return undersample_training_frame(
            train_df,
            5.0,
            target_column=target_column,
            random_state=random_state,
        )

    labels = _validated_binary_labels(train_df[target_column])
    copied = train_df.copy()
    metadata: dict[str, int | float | bool | str] = {
        "strategy": "class_weight",
        "sampling_applied": False,
        "original_rows": int(len(copied)),
        "original_majority_rows": int((labels == 0).sum()),
        "original_minority_rows": int((labels == 1).sum()),
        "effective_rows": int(len(copied)),
        "effective_majority_rows": int((labels == 0).sum()),
        "effective_minority_rows": int((labels == 1).sum()),
        "effective_fraud_rate": float(labels.mean()),
    }
    return copied, metadata


def threshold_tradeoff_table(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    y_score: Sequence[float] | pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Calculate confusion counts and alert metrics at every score boundary."""

    labels, scores = _validated_labels_and_scores(y_true, y_score)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_true_positives = np.cumsum(sorted_labels)
    cumulative_false_positives = np.cumsum(1 - sorted_labels)
    boundary_indices = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )

    total_positives = int(labels.sum())
    total_negatives = int(len(labels) - total_positives)
    true_positives = cumulative_true_positives[boundary_indices].astype(int)
    false_positives = cumulative_false_positives[boundary_indices].astype(int)
    alert_counts = boundary_indices + 1

    no_alert_threshold = np.nextafter(float(sorted_scores[0]), np.inf)
    thresholds = np.r_[no_alert_threshold, sorted_scores[boundary_indices]]
    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]
    alert_counts = np.r_[0, alert_counts]
    false_negatives = total_positives - true_positives
    true_negatives = total_negatives - false_positives

    precision = np.divide(
        true_positives,
        alert_counts,
        out=np.zeros(len(thresholds), dtype=float),
        where=alert_counts > 0,
    )
    recall = true_positives / total_positives
    specificity = true_negatives / total_negatives
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(thresholds), dtype=float),
        where=(precision + recall) > 0,
    )

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "true_negatives": true_negatives.astype(int),
            "false_positives": false_positives.astype(int),
            "false_negatives": false_negatives.astype(int),
            "true_positives": true_positives.astype(int),
            "alert_count": alert_counts.astype(int),
            "alert_rate": alert_counts / len(labels),
            "alerts_per_10k": alert_counts / len(labels) * 10_000,
            "fraud_captured_per_10k": true_positives / len(labels) * 10_000,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        }
    )


def select_alert_rate_threshold(
    tradeoffs: pd.DataFrame,
    target_alert_rate: float,
) -> dict[str, int | float]:
    """Select the closest observed alert rate that does not exceed capacity."""

    if not 0 <= target_alert_rate <= 1:
        raise ValueError("target_alert_rate must be between 0 and 1.")
    _validate_tradeoff_columns(tradeoffs)
    feasible = tradeoffs.loc[tradeoffs["alert_rate"] <= target_alert_rate]
    selected = feasible.sort_values(
        ["alert_rate", "recall", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    result = selected.to_dict()
    result["target_alert_rate"] = float(target_alert_rate)
    return result


def expected_classification_cost(
    false_positives: int,
    false_negatives: int,
    scenario: CostScenario,
) -> float:
    """Calculate the total relative error cost for one scenario."""

    if false_positives < 0 or false_negatives < 0:
        raise ValueError("Confusion-matrix counts must be non-negative.")
    return float(
        false_positives * scenario.false_positive_cost
        + false_negatives * scenario.false_negative_cost
    )


def select_cost_optimal_threshold(
    tradeoffs: pd.DataFrame,
    scenario: CostScenario,
) -> dict[str, int | float | str]:
    """Select the threshold with the lowest cost, preferring fewer alerts on ties."""

    _validate_tradeoff_columns(tradeoffs)
    candidates = tradeoffs.copy()
    candidates["expected_cost"] = (
        candidates["false_positives"] * scenario.false_positive_cost
        + candidates["false_negatives"] * scenario.false_negative_cost
    )
    selected = candidates.sort_values(
        ["expected_cost", "alert_rate", "threshold"],
        ascending=[True, True, False],
        kind="mergesort",
    ).iloc[0]
    result = selected.to_dict()
    result.update(
        {
            "cost_scenario": scenario.name,
            "false_positive_cost": float(scenario.false_positive_cost),
            "false_negative_cost": float(scenario.false_negative_cost),
            "mean_cost_per_transaction": float(
                selected["expected_cost"] / _evaluation_row_count(tradeoffs)
            ),
        }
    )
    return result


def build_threshold_summary(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    y_score: Sequence[float] | pd.Series | np.ndarray,
    *,
    default_threshold: float = 0.5,
    alert_rates: Iterable[float] = DEFAULT_ALERT_RATES,
    cost_scenarios: Iterable[CostScenario] = DEFAULT_COST_SCENARIOS,
) -> pd.DataFrame:
    """Summarise default, F1, capacity, and cost-based validation thresholds."""

    labels, scores = _validated_labels_and_scores(y_true, y_score)
    tradeoffs = threshold_tradeoff_table(labels, scores)
    f1_threshold, _ = best_f1_threshold(pd.Series(labels), scores)

    selections: list[tuple[str, float, dict[str, Any]]] = [
        ("default", float(default_threshold), {}),
        ("max_f1", float(f1_threshold), {}),
    ]
    for alert_rate in alert_rates:
        selected = select_alert_rate_threshold(tradeoffs, float(alert_rate))
        selections.append(
            (
                f"alert_capacity_{float(alert_rate):g}",
                float(selected["threshold"]),
                {"target_alert_rate": float(alert_rate)},
            )
        )
    for scenario in cost_scenarios:
        selected = select_cost_optimal_threshold(tradeoffs, scenario)
        selections.append(
            (
                f"cost_{scenario.name}",
                float(selected["threshold"]),
                {
                    "cost_scenario": scenario.name,
                    "false_positive_cost": scenario.false_positive_cost,
                    "false_negative_cost": scenario.false_negative_cost,
                },
            )
        )

    rows: list[dict[str, Any]] = []
    for selection_name, threshold, context in selections:
        metrics = classification_metrics(pd.Series(labels), scores, threshold)
        alert_count = int(metrics["true_positives"] + metrics["false_positives"])
        row: dict[str, Any] = {
            "selection": selection_name,
            "threshold": threshold,
            **metrics,
            "alert_count": alert_count,
            "alert_rate": alert_count / len(labels),
            "alerts_per_10k": alert_count / len(labels) * 10_000,
            "fraud_captured_per_10k": metrics["true_positives"] / len(labels) * 10_000,
            **context,
        }
        if "cost_scenario" in context:
            row["expected_cost"] = expected_classification_cost(
                int(metrics["false_positives"]),
                int(metrics["false_negatives"]),
                CostScenario(
                    str(context["cost_scenario"]),
                    float(context["false_positive_cost"]),
                    float(context["false_negative_cost"]),
                ),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_validation_candidate(
    results: Iterable[Mapping[str, Any]] | pd.DataFrame,
    *,
    metric: str = "validation_pr_auc",
) -> dict[str, Any]:
    """Choose a candidate using a validation metric and never a test metric."""

    if not metric.startswith("validation_") or "test" in metric.lower():
        raise ValueError("Candidate selection must use a validation metric, never a test metric.")
    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(list(results))
    if frame.empty:
        raise ValueError("At least one candidate result is required.")
    if metric not in frame.columns:
        raise ValueError(f"Candidate results do not contain selection metric {metric!r}.")
    numeric_metric = pd.to_numeric(frame[metric], errors="coerce")
    if numeric_metric.isna().any() or not np.isfinite(numeric_metric).all():
        raise ValueError(f"Selection metric {metric!r} must contain finite numeric values.")

    ranked = frame.assign(_selection_metric=numeric_metric)
    sort_columns = ["_selection_metric"]
    ascending = [False]
    if "training_seconds" in ranked.columns:
        sort_columns.append("training_seconds")
        ascending.append(True)
    if "model" in ranked.columns:
        sort_columns.append("model")
        ascending.append(True)
    selected = ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").iloc[0]
    return selected.drop(labels="_selection_metric").to_dict()


def relative_validation_improvement(candidate_score: float, reference_score: float) -> float:
    """Return relative validation improvement as a fraction, not a percentage."""

    if not np.isfinite(candidate_score) or not np.isfinite(reference_score):
        raise ValueError("Validation scores must be finite.")
    if reference_score <= 0:
        raise ValueError("reference_score must be positive.")
    return float((candidate_score - reference_score) / reference_score)


def _validated_binary_labels(
    values: Sequence[int] | pd.Series | np.ndarray,
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Binary labels must be a non-empty one-dimensional sequence.")
    if pd.isna(labels).any():
        raise ValueError("Binary labels must not contain missing values.")
    unique_values = set(np.unique(labels).tolist())
    if unique_values != {0, 1}:
        raise ValueError("Binary labels must contain both classes encoded as 0 and 1.")
    return labels.astype(int, copy=False)


def _validated_labels_and_scores(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    y_score: Sequence[float] | pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _validated_binary_labels(y_true)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Scores must be one-dimensional and match the number of labels.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")
    return labels, scores


def _validate_tradeoff_columns(tradeoffs: pd.DataFrame) -> None:
    required = {
        "threshold",
        "false_positives",
        "false_negatives",
        "alert_count",
        "alert_rate",
        "recall",
    }
    missing = sorted(required - set(tradeoffs.columns))
    if missing:
        raise ValueError(f"Threshold trade-off table is missing columns: {missing}")
    if tradeoffs.empty:
        raise ValueError("Threshold trade-off table must not be empty.")


def _evaluation_row_count(tradeoffs: pd.DataFrame) -> int:
    """Recover evaluation row count from a complete threshold trade-off table."""

    final_row = tradeoffs.sort_values("alert_count", ascending=False).iloc[0]
    row_count = int(final_row["alert_count"])
    if row_count <= 0:
        raise ValueError("Could not recover a positive evaluation row count.")
    return row_count
