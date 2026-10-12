"""Training-only helpers for the Stage 19 multivariate-signal diagnostic."""

# %% Imports and fixed diagnostic contract
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_modeling_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from predictive_quality_utils import ID_COLUMN, TARGET_COLUMN, TIME_COLUMN, TRAIN_CODE, VALIDATION_CODE
from sender_location_history_utils import HISTORY_FEATURES, add_sender_location_history_features, fixed_workload_table
from validation_evaluation_utils import positive_class_weight

RANDOM_STATE: Final = 19_042
AUDIT_FRACTION: Final = 0.20
ALERT_RATE: Final = 0.05
MISSING_CATEGORY: Final = "<MISSING>"
TIME_STATUS_FEATURE: Final = "time_since_last_transaction_status"
SUPPLIED_RISK_SCORES: Final = ["spending_deviation_score", "velocity_score", "geo_anomaly_score"]
CORE_NUMERIC_FEATURES: Final = ["amount", "time_since_last_transaction", *HISTORY_FEATURES]
CORE_CATEGORICAL_FEATURES: Final = [*CATEGORICAL_FEATURES, TIME_STATUS_FEATURE]
INTERACTION_FEATURES: Final = [
    "location_x_device",
    "location_x_payment_channel",
    "transaction_type_x_payment_channel",
    "transaction_type_x_time_status",
]
PROHIBITED_PREDICTORS: Final = {TARGET_COLUMN, "fraud_type", ID_COLUMN, TIME_COLUMN, "sender_account"}


@dataclass(frozen=True)
class FeatureSet:
    """One predeclared feature set; the model configuration stays unchanged."""

    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    purpose: str

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.numeric, *self.categorical)


def frozen_feature_sets() -> tuple[FeatureSet, ...]:
    """Return the three ordered comparisons used by the diagnostic."""

    return (
        FeatureSet(
            "core_fields",
            tuple(CORE_NUMERIC_FEATURES),
            tuple(CORE_CATEGORICAL_FEATURES),
            "Observed transaction, category and past-only history fields; excludes supplied risk scores.",
        ),
        FeatureSet(
            "all_supplied_main_fields",
            tuple([*CORE_NUMERIC_FEATURES, *SUPPLIED_RISK_SCORES]),
            tuple(CORE_CATEGORICAL_FEATURES),
            "Adds the three supplied risk-score fields whose generation is not documented.",
        ),
        FeatureSet(
            "all_supplied_plus_interactions",
            tuple([*CORE_NUMERIC_FEATURES, *SUPPLIED_RISK_SCORES]),
            tuple([*CORE_CATEGORICAL_FEATURES, *INTERACTION_FEATURES]),
            "Adds four predeclared, label-free combinations to test incremental joint signal.",
        ),
    )


# %% Label-free feature preparation and training-internal split
def add_multivariate_features(training: pd.DataFrame) -> pd.DataFrame:
    """Add only deterministic fields that do not read the fraud label."""

    required = {*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, *HISTORY_FEATURES}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise ValueError(f"Missing Stage 19 input columns: {missing}")
    result = training.copy()
    time_values = pd.to_numeric(result["time_since_last_transaction"], errors="coerce")
    result[TIME_STATUS_FEATURE] = np.where(np.isfinite(time_values), "observed", "missing")

    def clean(column: str) -> pd.Series:
        return result[column].astype("string").fillna(MISSING_CATEGORY)

    result["location_x_device"] = clean("location") + " | " + clean("device_used")
    result["location_x_payment_channel"] = clean("location") + " | " + clean("payment_channel")
    result["transaction_type_x_payment_channel"] = (
        clean("transaction_type") + " | " + clean("payment_channel")
    )
    result["transaction_type_x_time_status"] = clean("transaction_type") + " | " + clean(TIME_STATUS_FEATURE)
    return result


def validate_feature_sets(feature_sets: Sequence[FeatureSet]) -> None:
    """Reject target, identifier and post-outcome fields before any fit."""

    if not feature_sets:
        raise ValueError("At least one feature set is required.")
    names = [item.name for item in feature_sets]
    if len(names) != len(set(names)):
        raise ValueError("Feature-set names must be unique.")
    for item in feature_sets:
        duplicate_columns = len(item.columns) != len(set(item.columns))
        prohibited = sorted(PROHIBITED_PREDICTORS.intersection(item.columns))
        if duplicate_columns or prohibited:
            raise ValueError(f"Unsafe feature set {item.name!r}; prohibited={prohibited}, duplicates={duplicate_columns}.")


def internal_development_audit_split(
    training: pd.DataFrame, *, audit_fraction: float = AUDIT_FRACTION, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create a reproducible stratified split entirely inside frozen training."""

    required = {ID_COLUMN, TARGET_COLUMN, TIME_COLUMN}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise ValueError(f"Missing split columns: {missing}")
    if not 0 < audit_fraction < 0.5:
        raise ValueError("audit_fraction must be greater than zero and below 0.5.")
    if training[ID_COLUMN].isna().any() or training[ID_COLUMN].duplicated().any():
        raise ValueError("Training transaction IDs must be present and unique.")
    labels = training[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("The training target must contain both binary classes.")
    positions = np.arange(len(training), dtype=np.int64)
    development_positions, audit_positions = train_test_split(
        positions, test_size=audit_fraction, random_state=random_state, shuffle=True, stratify=labels
    )
    development = training.iloc[np.sort(development_positions)].reset_index(drop=True)
    audit = training.iloc[np.sort(audit_positions)].reset_index(drop=True)
    overlap = set(development[ID_COLUMN]).intersection(audit[ID_COLUMN])
    if overlap or len(development) + len(audit) != len(training):
        raise RuntimeError("Internal development/audit split is not disjoint and exhaustive.")

    def id_hash(frame: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        for value in sorted(frame[ID_COLUMN].astype(str)):
            digest.update(value.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    return development, audit, {
        "scope": "frozen_training_partition_only",
        "random_state": random_state,
        "audit_fraction": audit_fraction,
        "development_rows": int(len(development)),
        "audit_rows": int(len(audit)),
        "development_fraud_rows": int(development[TARGET_COLUMN].sum()),
        "audit_fraud_rows": int(audit[TARGET_COLUMN].sum()),
        "development_id_sha256": id_hash(development),
        "audit_id_sha256": id_hash(audit),
        "validation_rows_used": 0,
        "test_rows_used": 0,
    }


def attach_development_only_history(
    development: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build past-only counts while preventing audit events from entering history."""

    if development.empty or audit.empty:
        raise ValueError("Development and audit subsets must both be non-empty.")
    combined = pd.concat([development, audit], ignore_index=True)
    partition = np.concatenate(
        [
            np.full(len(development), TRAIN_CODE, dtype="uint8"),
            np.full(len(audit), VALIDATION_CODE, dtype="uint8"),
        ]
    )
    enriched = add_sender_location_history_features(combined, partition)
    development_enriched = enriched.iloc[: len(development)].reset_index(drop=True)
    audit_enriched = enriched.iloc[len(development) :].reset_index(drop=True)
    if not development_enriched[ID_COLUMN].equals(development[ID_COLUMN].reset_index(drop=True)):
        raise RuntimeError("History construction changed development row identity or order.")
    if not audit_enriched[ID_COLUMN].equals(audit[ID_COLUMN].reset_index(drop=True)):
        raise RuntimeError("History construction changed audit row identity or order.")
    return development_enriched, audit_enriched


# %% Fixed model and aggregate evaluation
def _optional_lightgbm() -> Any:
    try:
        return importlib.import_module("lightgbm")
    except ImportError as error:
        raise RuntimeError("Stage 19 requires the pinned lightgbm dependency.") from error


def build_diagnostic_pipeline(
    feature_set: FeatureSet, *, class_weight: float, random_state: int = RANDOM_STATE
) -> Pipeline:
    """Build a fixed, deliberately untuned LightGBM diagnostic pipeline."""

    if class_weight <= 0 or not np.isfinite(class_weight):
        raise ValueError("class_weight must be finite and positive.")
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), list(feature_set.numeric)),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(feature_set.categorical),
            ),
        ]
    )
    lightgbm = _optional_lightgbm()
    model = lightgbm.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def score_metrics(labels: Sequence[int], scores: Sequence[float], amounts: Sequence[float]) -> dict[str, float | int]:
    """Return rank and fixed-workload metrics without selecting a threshold."""

    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_score) or len(np.unique(y_true)) != 2:
        raise ValueError("Metric inputs must have equal rows and both binary classes.")
    if not np.isfinite(y_score).all():
        raise ValueError("Diagnostic scores must be finite.")
    workload = fixed_workload_table(y_true, y_score, amounts, alert_rates=(ALERT_RATE,)).iloc[0]
    return {
        "rows": int(len(y_true)),
        "fraud_rows": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision_at_5pct": float(workload["precision"]),
        "fraud_count_recall_at_5pct": float(workload["fraud_count_recall"]),
        "fraud_value_recall_at_5pct": float(workload["fraud_value_recall"]),
    }


def fit_and_score(
    development: pd.DataFrame,
    audit: pd.DataFrame,
    feature_set: FeatureSet,
    *,
    shuffle_development_labels: bool = False,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    """Fit on the internal development subset and score the untouched training-audit subset."""

    missing = sorted(set(feature_set.columns).difference(development.columns).union(set(feature_set.columns).difference(audit.columns)))
    if missing:
        raise ValueError(f"Missing diagnostic model fields: {missing}")
    original_labels = development[TARGET_COLUMN].to_numpy(dtype=np.int8)
    fit_labels = original_labels.copy()
    if shuffle_development_labels:
        fit_labels = np.random.default_rng(random_state).permutation(fit_labels)
        if len(fit_labels) > 1 and np.array_equal(fit_labels, original_labels):
            fit_labels = np.roll(fit_labels, 1)
    pipeline = build_diagnostic_pipeline(
        feature_set, class_weight=positive_class_weight(fit_labels), random_state=random_state
    )
    started = perf_counter()
    pipeline.fit(development[list(feature_set.columns)], fit_labels)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    scores = pipeline.predict_proba(audit[list(feature_set.columns)])[:, 1]
    inference_seconds = perf_counter() - started
    metrics = score_metrics(audit[TARGET_COLUMN], scores, audit["amount"])
    metrics.update(
        {
            "candidate": feature_set.name + ("_shuffled_label_control" if shuffle_development_labels else ""),
            "feature_count_before_encoding": len(feature_set.columns),
            "fit_seconds": float(fit_seconds),
            "audit_inference_seconds": float(inference_seconds),
            "development_labels_shuffled": bool(shuffle_development_labels),
            "eligible_for_model_selection": False,
        }
    )
    return pipeline, scores, metrics


def baseline_results(audit: pd.DataFrame) -> list[dict[str, Any]]:
    """Evaluate prevalence and the strongest Stage 17 marginal pattern."""

    labels = audit[TARGET_COLUMN].to_numpy(dtype=np.int8)
    prevalence_scores = np.repeat(labels.mean(), len(labels))
    observed_scores = pd.to_numeric(audit["time_since_last_transaction"], errors="coerce").notna().astype(float)
    rows = []
    for name, scores in (("prevalence_baseline", prevalence_scores), ("time_observed_marginal_pattern", observed_scores)):
        result = score_metrics(labels, scores, audit["amount"])
        for metric in ("precision_at_5pct", "fraud_count_recall_at_5pct", "fraud_value_recall_at_5pct"):
            result[metric] = float("nan")
        result.update(
            {
                "candidate": name,
                "feature_count_before_encoding": 0 if name == "prevalence_baseline" else 1,
                "fit_seconds": 0.0,
                "audit_inference_seconds": 0.0,
                "development_labels_shuffled": False,
                "fixed_workload_ties_make_ranking_ambiguous": True,
                "eligible_for_model_selection": False,
            }
        )
        rows.append(result)
    return rows


# %% Stability and interpretation
def audit_slice_metrics(audit: pd.DataFrame, scores: Sequence[float], candidate: str) -> pd.DataFrame:
    """Measure AP/AUC in fixed time quarters and ID-hash halves of the training audit subset."""

    if len(audit) != len(scores):
        raise ValueError("Audit rows and scores must have equal length.")
    frame = audit[[ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, "amount"]].copy()
    frame["_score"] = np.asarray(scores, dtype=float)
    timestamp_rank = frame[TIME_COLUMN].rank(method="first")
    frame["_time_slice"] = pd.qcut(timestamp_rank, 4, labels=["time_q1", "time_q2", "time_q3", "time_q4"])
    frame["_hash_slice"] = np.where(
        pd.util.hash_pandas_object(frame[ID_COLUMN].astype("string"), index=False).to_numpy(dtype=np.uint64) % 2 == 0,
        "hash_a",
        "hash_b",
    )
    rows = []
    for column in ("_time_slice", "_hash_slice"):
        for slice_name, part in frame.groupby(column, observed=True, sort=True):
            if part[TARGET_COLUMN].nunique() < 2:
                metrics = {
                    "rows": int(len(part)),
                    "fraud_rows": int(part[TARGET_COLUMN].sum()),
                    "fraud_rate": float(part[TARGET_COLUMN].mean()),
                    "average_precision": float("nan"),
                    "roc_auc": float("nan"),
                    "precision_at_5pct": float("nan"),
                    "fraud_count_recall_at_5pct": float("nan"),
                    "fraud_value_recall_at_5pct": float("nan"),
                }
            else:
                metrics = score_metrics(part[TARGET_COLUMN], part["_score"], part["amount"])
            rows.append({"candidate": candidate, "slice": str(slice_name), **metrics})
    return pd.DataFrame(rows)


def feature_importance_table(pipeline: Pipeline, candidate: str) -> pd.DataFrame:
    """Return aggregate-safe transformed-feature gain importances."""

    preprocessing = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    names = preprocessing.get_feature_names_out()
    importances = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    if len(names) != len(importances):
        raise RuntimeError("Transformed feature names and importances do not align.")
    total = float(importances.sum())
    return pd.DataFrame(
        {
            "candidate": candidate,
            "transformed_feature": names,
            "gain_importance": importances,
            "gain_share": importances / total if total > 0 else np.zeros(len(importances)),
        }
    ).sort_values(["gain_importance", "transformed_feature"], ascending=[False, True], ignore_index=True)


def interpretation_summary(results: pd.DataFrame, slices: pd.DataFrame) -> dict[str, Any]:
    """Describe evidence size without selecting or replacing a production model."""

    indexed = results.set_index("candidate")
    required = {
        "prevalence_baseline",
        "time_observed_marginal_pattern",
        "core_fields",
        "all_supplied_main_fields",
        "all_supplied_plus_interactions",
        "all_supplied_plus_interactions_shuffled_label_control",
    }
    missing = sorted(required.difference(indexed.index))
    if missing:
        raise ValueError(f"Missing interpretation candidates: {missing}")
    interaction_lift = float(
        indexed.at["all_supplied_plus_interactions", "average_precision"]
        - indexed.at["all_supplied_main_fields", "average_precision"]
    )
    multivariate_lift = float(
        indexed.at["all_supplied_main_fields", "average_precision"]
        - indexed.at["time_observed_marginal_pattern", "average_precision"]
    )
    shuffled_gap = float(
        indexed.at["all_supplied_plus_interactions", "average_precision"]
        - indexed.at["all_supplied_plus_interactions_shuffled_label_control", "average_precision"]
    )
    interaction_slices = slices.loc[slices["candidate"] == "all_supplied_plus_interactions", ["slice", "average_precision"]]
    main_slices = slices.loc[slices["candidate"] == "all_supplied_main_fields", ["slice", "average_precision"]]
    paired = interaction_slices.merge(main_slices, on="slice", suffixes=("_interaction", "_main"))
    stable_positive_slices = int((paired["average_precision_interaction"] > paired["average_precision_main"]).sum())
    if interaction_lift < 0.001:
        interaction_reading = "no_clear_incremental_interaction_signal"
    elif interaction_lift < 0.005:
        interaction_reading = "small_incremental_interaction_signal"
    else:
        interaction_reading = "larger_exploratory_interaction_signal_requires_new_validation"
    return {
        "multivariate_ap_lift_over_time_missingness_pattern": multivariate_lift,
        "interaction_ap_lift_over_all_main_fields": interaction_lift,
        "interaction_positive_stability_slices": stable_positive_slices,
        "interaction_total_stability_slices": int(len(paired)),
        "interaction_reading": interaction_reading,
        "true_label_ap_gap_over_shuffled_control": shuffled_gap,
        "thresholds_are_descriptive_not_selection_rules": True,
        "model_or_operating_rule_changed": False,
        "registered_model_changed": False,
        "api_changed": False,
        "validation_analysed": False,
        "test_analysed": False,
    }
