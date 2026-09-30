"""Frozen configuration and safeguards for the Stage 11 model comparison.

This module contains no data loading, model fitting, or MLflow side effects.
Keeping the experiment rules separate makes them testable before validation
scores are produced.
"""

# %% Imports and constants
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from fraud_modeling_utils import CATEGORICAL_FEATURES
from oversampling_experiment_utils import MODEL_NUMERIC_FEATURES

DEFAULT_RANDOM_STATE: Final = 42
FROZEN_FULL_SPLIT_HASH: Final = "e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242"
ALLOWED_STAGE_10_STRATEGIES: Final = (
    "class_weight_reference",
    "undersample_5_to_1",
    "random_oversample_10_to_1",
    "random_oversample_5_to_1",
    "smotenc_10_to_1",
    "smotenc_5_to_1",
)
REFERENCE_MODEL: Final = "random_forest_reference"
COMPARISON_ALERT_RATE: Final = 0.05
AVERAGE_PRECISION_TIE_TOLERANCE: Final = 0.0001


@dataclass(frozen=True)
class ModelFamilyConfig:
    """One predeclared Stage 11 model configuration."""

    name: str
    family: str
    preprocessing: str
    parameters: Mapping[str, Any]
    full_data_candidate: bool = True


# %% Frozen candidate configurations
def frozen_model_family_configs() -> tuple[ModelFamilyConfig, ...]:
    """Return model configurations in their fixed execution order."""

    return (
        ModelFamilyConfig(
            name=REFERENCE_MODEL,
            family="random_forest",
            preprocessing="one_hot",
            parameters={"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 100, "max_features": "sqrt"},
        ),
        ModelFamilyConfig(
            name="hist_gradient_boosting",
            family="hist_gradient_boosting",
            preprocessing="ordinal",
            parameters={
                "max_iter": 150,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 100,
                "l2_regularization": 0.0,
            },
        ),
        ModelFamilyConfig(
            name="lightgbm",
            family="lightgbm",
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
        ModelFamilyConfig(
            name="xgboost",
            family="xgboost",
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
        ModelFamilyConfig(
            name="catboost",
            family="catboost",
            preprocessing="ordinal",
            parameters={"iterations": 300, "learning_rate": 0.05, "depth": 6, "verbose": False},
        ),
        ModelFamilyConfig(
            name="linear_svm",
            family="sgd_hinge",
            preprocessing="scaled_one_hot",
            parameters={"loss": "hinge", "max_iter": 1000, "tol": 0.001},
        ),
        ModelFamilyConfig(
            name="knn_sampled_feasibility",
            family="knn",
            preprocessing="scaled_one_hot",
            parameters={"n_neighbors": 25, "weights": "distance", "metric": "euclidean"},
            full_data_candidate=False,
        ),
    )


# %% Train-fitted preprocessing and estimator construction
def _one_hot_preprocessing(*, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), MODEL_NUMERIC_FEATURES),
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


def _ordinal_preprocessing() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), MODEL_NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"Required Stage 11 dependency {name!r} is not installed.") from error


def build_model_pipeline(
    config: ModelFamilyConfig, *, positive_class_weight: float, random_state: int = DEFAULT_RANDOM_STATE
) -> Pipeline:
    """Build one class-weighted candidate without fitting it."""

    if positive_class_weight <= 0 or not np.isfinite(positive_class_weight):
        raise ValueError("positive_class_weight must be finite and greater than zero.")
    parameters = dict(config.parameters)
    if config.family == "random_forest":
        preprocessing = _one_hot_preprocessing()
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample", random_state=random_state, n_jobs=-1, **parameters
        )
    elif config.family == "hist_gradient_boosting":
        preprocessing = _ordinal_preprocessing()
        mask = [False] * len(MODEL_NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
        estimator = HistGradientBoostingClassifier(
            class_weight="balanced",
            categorical_features=mask,
            early_stopping=False,
            random_state=random_state,
            **parameters,
        )
    elif config.family == "lightgbm":
        preprocessing = _one_hot_preprocessing()
        lightgbm = _optional_module("lightgbm")
        estimator = lightgbm.LGBMClassifier(
            objective="binary",
            scale_pos_weight=positive_class_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            **parameters,
        )
    elif config.family == "xgboost":
        preprocessing = _one_hot_preprocessing()
        xgboost = _optional_module("xgboost")
        estimator = xgboost.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=positive_class_weight,
            random_state=random_state,
            n_jobs=-1,
            **parameters,
        )
    elif config.family == "catboost":
        preprocessing = _ordinal_preprocessing()
        catboost = _optional_module("catboost")
        estimator = catboost.CatBoostClassifier(
            auto_class_weights="Balanced",
            random_seed=random_state,
            thread_count=-1,
            allow_writing_files=False,
            **parameters,
        )
    elif config.family == "sgd_hinge":
        preprocessing = _one_hot_preprocessing(scale_numeric=True)
        estimator = SGDClassifier(class_weight="balanced", random_state=random_state, **parameters)
    elif config.family == "knn":
        preprocessing = _one_hot_preprocessing(scale_numeric=True)
        estimator = KNeighborsClassifier(n_jobs=-1, **parameters)
    else:
        raise ValueError(f"Unsupported Stage 11 model family: {config.family!r}.")
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


# %% Stage 10 handoff validation
def validate_stage_10_handoff(summary: Mapping[str, Any]) -> dict[str, str]:
    """Accept only a complete full-data Stage 10 validation decision."""

    if summary.get("test_split_evaluated") is not False:
        raise ValueError("The Stage 10 handoff must confirm that the test split was not evaluated.")
    prohibited = [key for key in summary if str(key).lower().startswith("test_") and key != "test_split_evaluated"]
    if prohibited:
        raise ValueError(f"Stage 10 handoff contains prohibited test evidence: {prohibited}")
    if summary.get("evidence_scope") != "full_data":
        raise ValueError("Stage 11 requires a full-data Stage 10 decision, not smoke evidence.")
    decision = summary.get("decision", summary)
    if not isinstance(decision, Mapping):
        raise ValueError("The Stage 10 decision payload is missing.")
    if decision.get("decision_eligible") is not True:
        raise ValueError("The Stage 10 summary is not eligible to select a strategy.")
    strategy = decision.get("selected_strategy")
    if strategy not in ALLOWED_STAGE_10_STRATEGIES:
        raise ValueError(f"Unknown or missing Stage 10 strategy: {strategy!r}.")
    split_manifest = summary.get("split_manifest", {})
    if split_manifest is not None and not isinstance(split_manifest, Mapping):
        raise ValueError("The Stage 10 split manifest has an invalid format.")
    split_hash = (
        split_manifest.get("split_assignment_sha256")
        or summary.get("split_assignment_sha256")
        or summary.get("frozen_full_split_hash")
    )
    if split_hash != FROZEN_FULL_SPLIT_HASH:
        raise ValueError("The Stage 10 handoff does not match the frozen split hash.")
    run_id = summary.get("summary_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("The Stage 10 summary run ID is required.")
    return {"selected_strategy": str(strategy), "split_assignment_sha256": str(split_hash), "summary_run_id": run_id}


# %% Deterministic sampled KNN feasibility population
def stratified_sample_indices(
    labels: Sequence[int] | pd.Series | np.ndarray, sample_size: int, *, random_state: int = DEFAULT_RANDOM_STATE
) -> np.ndarray:
    """Return stable row positions while preserving the source fraud rate."""

    values = np.asarray(labels)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Labels must be a non-empty one-dimensional sequence.")
    if not np.isin(values, [0, 1]).all() or len(np.unique(values)) != 2:
        raise ValueError("Labels must contain both binary classes.")
    if sample_size <= 0 or sample_size > len(values):
        raise ValueError("sample_size must be between 1 and the number of rows.")
    if sample_size == len(values):
        return np.arange(len(values), dtype=np.int64)
    positions = np.arange(len(values), dtype=np.int64)
    sampled, _ = train_test_split(positions, train_size=sample_size, stratify=values, random_state=random_state)
    return np.sort(sampled.astype(np.int64, copy=False))


# %% Validation-only model progression decision
def _metric(result: Mapping[str, Any], name: str) -> float:
    if name not in result:
        raise ValueError(f"Model result is missing required metric {name!r}.")
    value = float(result[name])
    if not np.isfinite(value):
        raise ValueError(f"Model metric {name!r} must be finite.")
    return value


def select_model_family(results: Sequence[Mapping[str, Any]], *, evidence_scope: str) -> dict[str, Any]:
    """Apply the predeclared replacement rule using validation evidence only."""

    if not results:
        raise ValueError("At least one model result is required.")
    for result in results:
        prohibited = [key for key in result if str(key).lower().startswith("test_")]
        if prohibited:
            raise ValueError(f"Model selection received prohibited test evidence: {prohibited}")
    if evidence_scope != "full_data":
        return {
            "decision_eligible": False,
            "selected_model": None,
            "recommendation": "smoke_run_no_model_decision",
            "candidate_decisions": [],
        }

    eligible = [result for result in results if result.get("full_data_candidate") is True]
    references = [result for result in eligible if result.get("model") == REFERENCE_MODEL]
    if len(references) != 1:
        raise ValueError("Exactly one eligible Random Forest reference result is required.")
    reference = references[0]
    reference_ap = _metric(reference, "validation_average_precision")
    reference_count = _metric(reference, "workload_5pct_fraud_count_recall")
    reference_value = _metric(reference, "workload_5pct_fraud_value_recall")

    passed: list[Mapping[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for candidate in eligible:
        if candidate is reference:
            continue
        ap = _metric(candidate, "validation_average_precision")
        count_recall = _metric(candidate, "workload_5pct_fraud_count_recall")
        value_recall = _metric(candidate, "workload_5pct_fraud_value_recall")
        feasible = candidate.get("feasible") is True
        passes = bool(
            feasible
            and ap > reference_ap
            and count_recall >= reference_count
            and value_recall >= reference_value
            and (count_recall > reference_count or value_recall > reference_value)
        )
        decisions.append({"model": candidate.get("model"), "feasible": feasible, "passes_progression_rule": passes})
        if passes:
            passed.append(candidate)

    selected: Mapping[str, Any] = reference
    for candidate in passed:
        selected_ap = _metric(selected, "validation_average_precision")
        candidate_ap = _metric(candidate, "validation_average_precision")
        if candidate_ap > selected_ap + AVERAGE_PRECISION_TIE_TOLERANCE:
            selected = candidate
            continue
        if abs(candidate_ap - selected_ap) <= AVERAGE_PRECISION_TIE_TOLERANCE:
            selected_value = _metric(selected, "workload_5pct_fraud_value_recall")
            candidate_value = _metric(candidate, "workload_5pct_fraud_value_recall")
            if candidate_value > selected_value:
                selected = candidate
            elif candidate_value == selected_value and _metric(candidate, "validation_inference_seconds") < _metric(
                selected, "validation_inference_seconds"
            ):
                selected = candidate

    selected_name = str(selected["model"])
    return {
        "decision_eligible": True,
        "selected_model": selected_name,
        "reference_model": REFERENCE_MODEL,
        "reference_retained": selected_name == REFERENCE_MODEL,
        "recommendation": (
            "retain_random_forest_reference"
            if selected_name == REFERENCE_MODEL
            else "progress_selected_model_to_limited_tuning"
        ),
        "candidate_decisions": decisions,
    }
