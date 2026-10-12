"""Versioned batch-monitoring profiles, comparisons and human-gated recommendations."""

# %% Imports and immutable monitoring schema
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

from fraud_modeling_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from oversampling_experiment_utils import MODEL_FEATURES

SCORE_COLUMN = "score"
TARGET_COLUMN = "is_fraud"
AMOUNT_COLUMN = "amount"
CATEGORICAL_MONITORING_FEATURES = tuple(CATEGORICAL_FEATURES)
NUMERIC_MONITORING_FEATURES = tuple(name for name in MODEL_FEATURES if name not in CATEGORICAL_FEATURES)
PROFILE_VERSION = "research_monitoring_reference_v1"


@dataclass(frozen=True)
class MonitoringThresholds:
    """Research heuristics; these values are not universal production standards."""

    psi_warning: float = 0.10
    psi_critical: float = 0.25
    missing_rate_change_warning: float = 0.05
    alert_rate_ratio_warning: float = 1.50
    minimum_rows: int = 100
    minimum_labelled_rows: int = 100
    minimum_positive_labels: int = 30
    minimum_fraud_value_recall: float = 0.80
    false_positive_review_cost: float = 1.0
    missed_fraud_amount_multiplier: float = 1.0


# %% Validation, profiling and stable population comparisons
def _json_number(value: Any) -> float | int | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _normalise_probabilities(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    values = np.clip(values, 0.0, None) + epsilon
    return values / values.sum()


def population_stability_index(reference: list[float], current: list[float]) -> float:
    """Return PSI with small smoothing for bins absent from either population."""

    expected = _normalise_probabilities(np.asarray(reference))
    observed = _normalise_probabilities(np.asarray(current))
    if expected.shape != observed.shape:
        raise ValueError("PSI distributions must use the same bins.")
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _require_columns(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{context} contains no rows.")


def validate_scored_batch(frame: pd.DataFrame, *, require_score: bool = True) -> dict[str, Any]:
    required = list(MODEL_FEATURES) + ([SCORE_COLUMN] if require_score else [])
    _require_columns(frame, required, "Monitoring batch")
    quality: dict[str, Any] = {
        "rows": int(len(frame)),
        "duplicate_transaction_ids": None,
        "missing_rates": {name: float(frame[name].isna().mean()) for name in MODEL_FEATURES},
        "invalid_numeric_values": {},
        "score_out_of_range_rows": 0,
    }
    if "transaction_id" in frame:
        quality["duplicate_transaction_ids"] = int(frame["transaction_id"].duplicated().sum())
    for name in NUMERIC_MONITORING_FEATURES:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        invalid = frame[name].notna() & (numeric.isna() | ~np.isfinite(numeric))
        quality["invalid_numeric_values"][name] = int(invalid.sum())
    if require_score:
        score = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
        invalid_score = score.isna() | ~np.isfinite(score) | (score < 0) | (score > 1)
        quality["score_out_of_range_rows"] = int(invalid_score.sum())
    return quality


def _numeric_profile(series: pd.Series, quantiles: int = 10) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = numeric.dropna().to_numpy(dtype="float64")
    if finite.size == 0:
        edges: list[float] = []
        counts: list[int] = []
    else:
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, quantiles + 1))).astype(float).tolist()
        counts = (
            [int(finite.size)]
            if len(edges) == 1
            else np.histogram(finite, bins=np.asarray(edges))[0].astype(int).tolist()
        )
    total = len(series)
    bin_counts = counts + [int(numeric.isna().sum())]
    return {
        "kind": "numeric",
        "edges": edges,
        "counts": bin_counts,
        "proportions": (np.asarray(bin_counts, dtype="float64") / total).tolist(),
        "missing_rate": float(numeric.isna().mean()),
        "minimum": _json_number(np.min(finite)) if finite.size else None,
        "maximum": _json_number(np.max(finite)) if finite.size else None,
    }


def _categorical_profile(series: pd.Series, maximum_categories: int = 50) -> dict[str, Any]:
    clean = series.astype("string").fillna("__MISSING__")
    top = clean.value_counts().head(maximum_categories).index.astype(str).tolist()
    grouped = clean.where(clean.isin(top), "__OTHER__")
    categories = [*top, "__OTHER__"]
    counts = grouped.value_counts().reindex(categories, fill_value=0).astype(int).tolist()
    return {
        "kind": "categorical",
        "categories": categories,
        "counts": counts,
        "proportions": (np.asarray(counts, dtype="float64") / len(series)).tolist(),
        "missing_rate": float(series.isna().mean()),
    }


def build_reference_profile(
    frame: pd.DataFrame, *, threshold: float, model_run_id: str, state_version: str, thresholds: MonitoringThresholds
) -> dict[str, Any]:
    quality = validate_scored_batch(frame)
    if (
        quality["score_out_of_range_rows"]
        or any(quality["invalid_numeric_values"].values())
        or quality["duplicate_transaction_ids"] not in (None, 0)
    ):
        raise ValueError("A monitoring reference must pass numeric, score and identifier quality checks.")
    features: dict[str, Any] = {}
    for name in CATEGORICAL_MONITORING_FEATURES:
        features[name] = _categorical_profile(frame[name])
    for name in NUMERIC_MONITORING_FEATURES:
        features[name] = _numeric_profile(frame[name])
    features[SCORE_COLUMN] = _numeric_profile(frame[SCORE_COLUMN])
    score = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    return {
        "profile_version": PROFILE_VERSION,
        "model_run_id": model_run_id,
        "state_version": state_version,
        "decision_threshold": float(threshold),
        "rows": int(len(frame)),
        "alert_rate": float((score >= threshold).mean()),
        "features": features,
        "quality": quality,
        "thresholds": asdict(thresholds),
        "source_scope": "explicit_reference_batch",
    }


def _current_proportions(series: pd.Series, specification: dict[str, Any]) -> tuple[list[float], float]:
    if specification["kind"] == "categorical":
        clean = series.astype("string").fillna("__MISSING__")
        known = specification["categories"][:-1]
        grouped = clean.where(clean.isin(known), "__OTHER__")
        counts = grouped.value_counts().reindex(specification["categories"], fill_value=0).to_numpy(dtype="float64")
        return (counts / len(series)).tolist(), float(series.isna().mean())
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = numeric.dropna().to_numpy(dtype="float64")
    edges = np.asarray(specification["edges"], dtype="float64")
    if len(edges) == 0:
        counts: list[int] = []
    elif len(edges) == 1:
        counts = [int(finite.size)]
    else:
        clipped = np.clip(finite, edges[0], edges[-1])
        counts = np.histogram(clipped, bins=edges)[0].astype(int).tolist()
    counts.append(int(numeric.isna().sum()))
    return (np.asarray(counts, dtype="float64") / len(series)).tolist(), float(numeric.isna().mean())


def compare_with_reference(frame: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    if profile.get("profile_version") != PROFILE_VERSION:
        raise ValueError("Unsupported monitoring reference profile.")
    validate_scored_batch(frame)
    rows = []
    for name, reference in profile["features"].items():
        current, missing_rate = _current_proportions(frame[name], reference)
        rows.append(
            {
                "feature": name,
                "kind": reference["kind"],
                "psi": population_stability_index(reference["proportions"], current),
                "reference_missing_rate": float(reference["missing_rate"]),
                "current_missing_rate": missing_rate,
                "missing_rate_change": missing_rate - float(reference["missing_rate"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["psi", "feature"], ascending=[False, True]).reset_index(drop=True)


# %% Delayed-label evidence, cost assumptions and human-gated decision
def labelled_performance(frame: pd.DataFrame, threshold: float, thresholds: MonitoringThresholds) -> dict[str, Any]:
    if TARGET_COLUMN not in frame:
        return {"status": "unavailable_no_labels", "labelled_rows": 0}
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    labelled = frame.loc[labels.isin([0, 1])].copy()
    if len(labelled) < thresholds.minimum_labelled_rows:
        return {"status": "insufficient_labels", "labelled_rows": int(len(labelled))}
    y_true = labelled[TARGET_COLUMN].astype(int).to_numpy()
    score = labelled[SCORE_COLUMN].astype(float).to_numpy()
    positive_labels = int(y_true.sum())
    if positive_labels < thresholds.minimum_positive_labels:
        return {
            "status": "insufficient_positive_labels",
            "labelled_rows": int(len(labelled)),
            "positive_labels": positive_labels,
        }
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "status": "available",
        "labelled_rows": int(len(labelled)),
        "positive_labels": positive_labels,
        "average_precision": float(average_precision_score(y_true, score)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    if AMOUNT_COLUMN in labelled:
        amount = pd.to_numeric(labelled[AMOUNT_COLUMN], errors="coerce").fillna(0).clip(lower=0).to_numpy()
        fraud_amount = float(amount[y_true == 1].sum())
        missed_amount = float(amount[(y_true == 1) & (prediction == 0)].sum())
        captured_amount = fraud_amount - missed_amount
        result.update(
            {
                "fraud_amount": fraud_amount,
                "captured_fraud_amount": captured_amount,
                "missed_fraud_amount": missed_amount,
                "fraud_value_recall": None if fraud_amount == 0 else captured_amount / fraud_amount,
                "assumption_based_cost": (
                    fp * thresholds.false_positive_review_cost
                    + missed_amount * thresholds.missed_fraud_amount_multiplier
                ),
                "cost_interpretation": "Scenario value only: amount is exposure, not verified institutional loss.",
            }
        )
    return result


def monitoring_recommendation(
    profile: dict[str, Any],
    frame: pd.DataFrame,
    drift: pd.DataFrame,
    quality: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    thresholds = MonitoringThresholds(**profile["thresholds"])
    reasons: list[str] = []
    critical_reasons: list[str] = []
    if len(frame) < thresholds.minimum_rows:
        reasons.append("batch_below_minimum_rows")
    if quality["score_out_of_range_rows"] or any(quality["invalid_numeric_values"].values()):
        critical_reasons.append("invalid_numeric_or_score_values")
    if quality["duplicate_transaction_ids"] not in (None, 0):
        critical_reasons.append("duplicate_transaction_ids")
    critical_features = drift.loc[drift.psi >= thresholds.psi_critical, "feature"].tolist()
    warning_features = drift.loc[
        (drift.psi >= thresholds.psi_warning) & (drift.psi < thresholds.psi_critical), "feature"
    ].tolist()
    if critical_features:
        critical_reasons.append("critical_population_drift")
    if warning_features:
        reasons.append("population_drift_warning")
    missing_changes = drift.loc[
        drift.missing_rate_change.abs() >= thresholds.missing_rate_change_warning, "feature"
    ].tolist()
    if missing_changes:
        reasons.append("missingness_change")
    current_score = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    current_alert_rate = float((current_score >= profile["decision_threshold"]).mean())
    reference_alert_rate = float(profile["alert_rate"])
    alert_ratio = None if reference_alert_rate == 0 else current_alert_rate / reference_alert_rate
    if alert_ratio is not None and (
        alert_ratio >= thresholds.alert_rate_ratio_warning or alert_ratio <= 1 / thresholds.alert_rate_ratio_warning
    ):
        reasons.append("alert_rate_change")
    if performance.get("status") == "available" and performance.get("fraud_value_recall") is not None:
        if performance["fraud_value_recall"] < thresholds.minimum_fraud_value_recall:
            critical_reasons.append("fraud_value_recall_below_research_floor")
    if critical_reasons:
        decision = "investigate_before_retraining_decision"
        severity = "critical"
    elif reasons:
        decision = "investigate"
        severity = "warning"
    else:
        decision = "continue_monitoring"
        severity = "normal"
    return {
        "severity": severity,
        "recommendation": decision,
        "critical_reasons": sorted(set(critical_reasons)),
        "warning_reasons": sorted(set(reasons)),
        "critical_features": critical_features,
        "warning_features": warning_features,
        "missingness_features": missing_changes,
        "reference_alert_rate": reference_alert_rate,
        "current_alert_rate": current_alert_rate,
        "alert_rate_ratio": alert_ratio,
        "automatic_retraining_triggered": False,
        "human_approval_required": decision != "continue_monitoring",
    }


def load_prediction_ledger(path: Path) -> pd.DataFrame:
    """Read successful local API records without modifying the private ledger."""

    if not path.exists():
        raise FileNotFoundError(f"Prediction ledger not found: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute("SELECT transaction_id, features_json, response_json FROM predictions").fetchall()
    finally:
        connection.close()
    records = []
    for transaction_id, features_json, response_json in rows:
        features, response = json.loads(features_json), json.loads(response_json)
        records.append({"transaction_id": transaction_id, **features, SCORE_COLUMN: response[SCORE_COLUMN]})
    return pd.DataFrame(records)
