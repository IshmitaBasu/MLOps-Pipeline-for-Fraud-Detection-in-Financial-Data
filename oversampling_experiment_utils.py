"""Shared safeguards and resampling helpers for the Stage 10 experiment."""

# %% Imports and frozen definitions
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from fraud_modeling_utils import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from sender_location_history_utils import HISTORY_FEATURES, build_development_frames, restore_raw_source_order

RANDOM_STATE: Final = 42
FEATURE_SET_NAME: Final = "original_v1_plus_fraud_behaviour_features_v2"
MODEL_FEATURES: Final = [*FEATURE_COLUMNS, *HISTORY_FEATURES]
MODEL_NUMERIC_FEATURES: Final = [*NUMERIC_FEATURES, *HISTORY_FEATURES]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    sampler_family: str
    sampling_strategy: float | None
    class_weight: str | None
    role: str


@dataclass(frozen=True)
class ResamplingResult:
    features: np.ndarray
    labels: np.ndarray
    diagnostics: dict[str, Any]
    sample_indices: np.ndarray | None = None


def frozen_strategies() -> tuple[StrategyConfig, ...]:
    """Return the predeclared comparison in its fixed execution order."""
    return (
        StrategyConfig("class_weight_reference", "none", None, "balanced_subsample", "main_reference"),
        StrategyConfig("undersample_5_to_1", "random_under_sampler", 0.20, None, "sensitivity_reference"),
        StrategyConfig("random_oversample_10_to_1", "random_over_sampler", 0.10, None, "oversampling_candidate"),
        StrategyConfig("random_oversample_5_to_1", "random_over_sampler", 0.20, None, "oversampling_candidate"),
        StrategyConfig("smotenc_10_to_1", "smotenc", 0.10, None, "conditional_synthetic_candidate"),
        StrategyConfig("smotenc_5_to_1", "smotenc", 0.20, None, "conditional_synthetic_candidate"),
    )


# %% Common train-fitted representation
def build_pre_resampling_transformer() -> ColumnTransformer:
    """Impute numeric fields and ordinal-code categories before resampling."""
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), MODEL_NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )


def fit_train_representation(
    train_df: pd.DataFrame, validation_df: pd.DataFrame
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """Fit preprocessing on training rows and transform untouched validation rows."""
    required = set(MODEL_FEATURES)
    for name, frame in (("training", train_df), ("validation", validation_df)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} feature columns: {sorted(missing)}")
    transformer = build_pre_resampling_transformer()
    train_array = np.asarray(transformer.fit_transform(train_df[MODEL_FEATURES]))
    validation_array = np.asarray(transformer.transform(validation_df[MODEL_FEATURES]))
    if not np.isfinite(train_array).all() or not np.isfinite(validation_array).all():
        raise ValueError("Preprocessed feature arrays must contain only finite values.")
    return transformer, train_array, validation_array


def categorical_indices() -> list[int]:
    start = len(MODEL_NUMERIC_FEATURES)
    return list(range(start, start + len(CATEGORICAL_FEATURES)))


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("Resampling labels must be a one-dimensional binary array.")
    return {"legitimate": int((values == 0).sum()), "fraud": int((values == 1).sum())}


def _validate_requested_ratio(labels: np.ndarray, requested_ratio: float | None, strategy_name: str) -> None:
    if requested_ratio is None:
        return
    counts = _class_counts(labels)
    observed = counts["fraud"] / counts["legitimate"]
    tolerance = 1 / max(1, counts["legitimate"])
    if not np.isclose(observed, requested_ratio, atol=tolerance, rtol=0):
        raise ValueError(f"{strategy_name} produced ratio {observed:.8f}, expected " f"{requested_ratio:.8f}.")


def _validate_smotenc_categories(original: np.ndarray, generated: np.ndarray) -> None:
    for index in categorical_indices():
        allowed = np.unique(original[:, index])
        if not np.isin(generated[:, index], allowed).all():
            raise ValueError("SMOTENC produced a categorical value absent from training.")


# %% Training-only resampling
def resample_training_data(
    train_features: np.ndarray,
    train_labels: Sequence[int] | np.ndarray | pd.Series,
    strategy: StrategyConfig,
    random_state: int = RANDOM_STATE,
) -> ResamplingResult:
    """Apply one frozen imbalance strategy to training arrays only."""
    features = np.asarray(train_features)
    labels = np.asarray(train_labels, dtype=np.int8)
    if features.ndim != 2 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError("Training features and labels must have equal non-zero rows.")
    if not np.isfinite(features).all():
        raise ValueError("Training features must be finite before resampling.")
    before = _class_counts(labels)
    sampler: Any | None = None
    if strategy.sampler_family == "none":
        resampled_features = features
        resampled_labels = labels
    elif strategy.sampler_family == "random_under_sampler":
        sampler = RandomUnderSampler(sampling_strategy=strategy.sampling_strategy, random_state=random_state)
        resampled_features, resampled_labels = sampler.fit_resample(features, labels)
    elif strategy.sampler_family == "random_over_sampler":
        sampler = RandomOverSampler(sampling_strategy=strategy.sampling_strategy, random_state=random_state)
        resampled_features, resampled_labels = sampler.fit_resample(features, labels)
    elif strategy.sampler_family == "smotenc":
        sampler = SMOTENC(
            categorical_features=categorical_indices(),
            sampling_strategy=strategy.sampling_strategy,
            random_state=random_state,
            k_neighbors=5,
        )
        resampled_features, resampled_labels = sampler.fit_resample(features, labels)
        _validate_smotenc_categories(features, np.asarray(resampled_features))
    else:
        raise ValueError(f"Unknown sampler family: {strategy.sampler_family}")

    resampled_features = np.asarray(resampled_features)
    resampled_labels = np.asarray(resampled_labels, dtype=np.int8)
    if not np.isfinite(resampled_features).all():
        raise ValueError(f"{strategy.name} produced non-finite feature values.")
    _validate_requested_ratio(resampled_labels, strategy.sampling_strategy, strategy.name)
    after = _class_counts(resampled_labels)
    sample_indices = getattr(sampler, "sample_indices_", None)
    if strategy.sampler_family == "random_over_sampler":
        if sample_indices is None:
            raise ValueError("Random oversampling did not expose source row indices.")
        if not np.array_equal(resampled_features, features[sample_indices]):
            raise ValueError("Random oversampling created rows instead of duplicating rows.")

    diagnostics = {
        "strategy": strategy.name,
        "sampler_family": strategy.sampler_family,
        "sampling_strategy": strategy.sampling_strategy,
        "random_state": random_state,
        "rows_before": int(len(labels)),
        "rows_after": int(len(resampled_labels)),
        "class_counts_before": before,
        "class_counts_after": after,
        "categorical_outputs_valid": True,
        "finite_outputs": True,
    }
    return ResamplingResult(
        features=resampled_features,
        labels=resampled_labels,
        diagnostics=diagnostics,
        sample_indices=(np.asarray(sample_indices, dtype=np.int64) if sample_indices is not None else None),
    )


# %% Fixed post-resampling model
def build_fixed_random_forest(
    strategy: StrategyConfig, random_state: int = RANDOM_STATE, categorical_levels: Sequence[np.ndarray] | None = None
) -> Pipeline:
    """One-hot encode resampled categories and fit the fixed Stage 08 forest."""
    numeric_indices = list(range(len(MODEL_NUMERIC_FEATURES)))
    encoder = OneHotEncoder(
        categories=(list(categorical_levels) if categorical_levels is not None else "auto"), handle_unknown="ignore"
    )
    preprocessing = ColumnTransformer(
        transformers=[("numeric", "passthrough", numeric_indices), ("categorical", encoder, categorical_indices())]
    )
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight=strategy.class_weight,
        random_state=random_state,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def categorical_levels_from_training(train_features: np.ndarray) -> list[np.ndarray]:
    """Freeze the category vocabulary before a sampler removes or adds rows."""
    features = np.asarray(train_features)
    return [np.unique(features[:, index]) for index in categorical_indices()]


# %% Validation integrity and strategy selection
def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Hash values, index, columns, and dtypes for mutation safeguards."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("|".join(map(str, frame.dtypes)).encode("utf-8"))
    return digest.hexdigest()


def validate_full_run_identity(*, source_rows: int, observed_split_hash: str, expected_split_hash: str) -> None:
    """Stop a full run unless its dataset size and frozen assignment match."""
    if source_rows != 5_000_000:
        raise ValueError("The Stage 10 full run requires exactly 5,000,000 source rows; " f"observed {source_rows:,}.")
    if observed_split_hash != expected_split_hash:
        raise ValueError(
            "The Stage 10 full run does not match the frozen Stage 07 split. "
            f"Expected {expected_split_hash}, observed {observed_split_hash}."
        )


def _five_percent_metrics(result: Mapping[str, Any]) -> tuple[float, float]:
    return (float(result["workload_5pct_fraud_count_recall"]), float(result["workload_5pct_fraud_value_recall"]))


def select_validation_strategy(results: Sequence[Mapping[str, Any]], *, evidence_scope: str) -> dict[str, Any]:
    """Apply the frozen rule without accepting any test-derived field."""
    if not results:
        raise ValueError("At least one strategy result is required.")
    for result in results:
        if any(str(key).lower().startswith("test") for key in result):
            raise ValueError("Strategy selection cannot consume test evidence.")
    reference_rows = [result for result in results if result["strategy"] == "class_weight_reference"]
    if len(reference_rows) != 1:
        raise ValueError("Exactly one class-weight reference result is required.")
    if evidence_scope != "full_data":
        return {
            "evidence_scope": evidence_scope,
            "decision_eligible": False,
            "selected_strategy": None,
            "recommendation": "smoke_run_no_strategy_decision",
            "test_split_evaluated": False,
        }

    reference = reference_rows[0]
    reference_count, reference_value = _five_percent_metrics(reference)
    eligible: list[Mapping[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    for candidate in results:
        if candidate is reference:
            continue
        candidate_count, candidate_value = _five_percent_metrics(candidate)
        passes = bool(
            float(candidate["validation_pr_auc"]) > float(reference["validation_pr_auc"])
            and candidate_count >= reference_count
            and candidate_value >= reference_value
            and (candidate_count > reference_count or candidate_value > reference_value)
        )
        assessments.append({"strategy": candidate["strategy"], "passes_progression_rule": passes})
        if passes:
            eligible.append(candidate)

    if not eligible:
        selected = reference
        recommendation = "retain_class_weight_reference"
    else:
        highest_pr_auc = max(float(row["validation_pr_auc"]) for row in eligible)
        contenders = [row for row in eligible if highest_pr_auc - float(row["validation_pr_auc"]) < 0.0001]
        contenders.sort(
            key=lambda row: (float(row["workload_5pct_fraud_value_recall"]), -float(row["training_seconds"])),
            reverse=True,
        )
        selected = contenders[0]
        recommendation = "use_selected_strategy_for_model_family_comparison"
    return {
        "evidence_scope": evidence_scope,
        "decision_eligible": True,
        "selected_strategy": selected["strategy"],
        "recommendation": recommendation,
        "candidate_assessments": assessments,
        "test_split_evaluated": False,
    }
