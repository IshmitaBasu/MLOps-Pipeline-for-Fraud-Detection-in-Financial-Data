"""Frozen final-evaluation protocol and training-only history for test queries."""

# %% Imports and fixed protocol
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fraud_modeling_utils import classification_metrics
from lightgbm_tuning_utils import frozen_lightgbm_configs
from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH
from oversampling_experiment_utils import FEATURE_SET_NAME
from predictive_quality_utils import ID_COLUMN, TIME_COLUMN
from sender_location_history_utils import (
    HISTORY_FEATURES,
    LOCATION_COLUMN,
    SENDER_COLUMN,
    _strictly_earlier_training_count,
    fixed_workload_table,
    validate_history_features,
)

FROZEN_THRESHOLD = 0.5226022799524157
STAGE_13_SUMMARY_RUN_ID = "5920c17849574c11b4854c3c5290bc41"
FROZEN_ALERT_RATES = (0.01, 0.05, 0.10)


def reserve_final_evaluation(receipt: Path, reservation: Mapping[str, Any]) -> None:
    """Reserve canonical scoring once; never replace an existing receipt."""

    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(dict(reservation), handle, indent=2)


# %% Handoff and protocol frozen before any test scoring
def freeze_final_protocol(summary: Mapping[str, Any], summary_run_id: str) -> dict[str, Any]:
    """Reject any handoff inconsistent with the approved validation reference."""

    if summary_run_id != STAGE_13_SUMMARY_RUN_ID or summary.get("evidence_scope") != "full_data":
        raise ValueError("The recorded full-data Stage 13 summary is required.")
    if summary.get("test_split_evaluated") is not False:
        raise ValueError("The handoff must confirm no revised test evaluation occurred.")
    if any(str(key).startswith("test_") and key != "test_split_evaluated" for key in summary):
        raise ValueError("The handoff contains prohibited test evidence.")
    decision = summary.get("decision", {})
    if not isinstance(decision, Mapping) or decision.get("decision_eligible") is not True:
        raise ValueError("The Stage 13 decision is missing or ineligible.")
    if decision.get("selected_configuration") != "lgbm_reference":
        raise ValueError("The final protocol expects the retained LightGBM reference.")
    if summary.get("feature_set") != FEATURE_SET_NAME or summary.get("imbalance_strategy") != "class_weight_reference":
        raise ValueError("The validation features or class-weight treatment changed.")
    manifest = summary.get("split_manifest", {})
    if not isinstance(manifest, Mapping) or manifest.get("split_assignment_sha256") != FROZEN_FULL_SPLIT_HASH:
        raise ValueError("The frozen split hash does not match Stage 13.")
    references = [row for row in summary.get("candidate_results", []) if row.get("configuration") == "lgbm_reference"]
    if len(references) != 1 or references[0].get("validation_threshold_at_max_f1") != FROZEN_THRESHOLD:
        raise ValueError("The recorded validation threshold does not match the frozen rule.")
    return {
        "stage_13_summary_run_id": summary_run_id,
        "validation_reference_run_id": references[0]["run_id"],
        "model": "lightgbm",
        "configuration": "lgbm_reference",
        "model_parameters": dict(frozen_lightgbm_configs()[0].parameters),
        "feature_set": FEATURE_SET_NAME,
        "imbalance_strategy": "class_weight_reference",
        "split_assignment_sha256": FROZEN_FULL_SPLIT_HASH,
        "random_state": 42,
        "fit_population": "original_training_partition_only",
        "history_population": "strictly_earlier_training_transactions_only",
        "primary_rule": "validation_max_f1_fixed_score_threshold",
        "threshold": FROZEN_THRESHOLD,
        "secondary_rules": "fixed_batch_review_workloads",
        "alert_rates": list(FROZEN_ALERT_RATES),
        "batch_tie_break": "descending_score_then_original_numeric_transaction_id_order",
        "test_threshold_optimisation": False,
        "probabilities_calibrated": False,
    }


# %% Test queries never update the training history
def add_final_history_features(data: pd.DataFrame, training_mask: np.ndarray) -> pd.DataFrame:
    """Calculate history from training events, with test events as read-only queries.

    Only identifiers, time, sender, location, and the training membership mask
    enter this calculation. Labels are never consulted.
    """

    mask = np.asarray(training_mask)
    if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(data):
        raise ValueError("A matching one-dimensional boolean training mask is required.")
    if not mask.any() or mask.all():
        raise ValueError("Both training and query rows are required.")
    required = [ID_COLUMN, TIME_COLUMN, SENDER_COLUMN, LOCATION_COLUMN]
    if not set(required).issubset(data.columns):
        raise ValueError("Final history keys are missing.")
    if not data.index.is_unique or data[ID_COLUMN].duplicated().any():
        raise ValueError("Final history rows must have unique indices and transaction IDs.")
    ordered = data[required].copy()
    ordered["_is_training_event"] = mask.astype("int8")
    ordered = ordered.sort_values([TIME_COLUMN, ID_COLUMN], kind="mergesort", na_position="first")
    sender = _strictly_earlier_training_count(ordered, [SENDER_COLUMN])
    pair = _strictly_earlier_training_count(ordered, [SENDER_COLUMN, LOCATION_COLUMN])
    result = data.copy()
    result[HISTORY_FEATURES[0]] = sender.reindex(result.index).astype("int32")
    result[HISTORY_FEATURES[1]] = pair.reindex(result.index).astype("int32")
    result[HISTORY_FEATURES[2]] = ((result[HISTORY_FEATURES[0]] > 0) & (result[HISTORY_FEATURES[1]] == 0)).astype(
        "int8"
    )
    validate_history_features(result)
    return result


# %% Final metrics at frozen rules only
def evaluate_frozen_test(
    labels: pd.Series | np.ndarray, scores: np.ndarray, amounts: pd.Series | np.ndarray, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute final metrics without selecting or searching test thresholds."""

    if protocol.get("threshold") != FROZEN_THRESHOLD or tuple(protocol.get("alert_rates", [])) != FROZEN_ALERT_RATES:
        raise ValueError("The test evaluation rules must match the frozen validation protocol.")
    if protocol.get("test_threshold_optimisation") is not False:
        raise ValueError("Test threshold optimisation is prohibited.")
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    workloads = fixed_workload_table(labels, scores, amounts, alert_rates=FROZEN_ALERT_RATES)
    if len(np.unique(labels)) != 2 or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Both classes and valid probability-like scores are required.")
    metrics = classification_metrics(labels, scores, FROZEN_THRESHOLD)
    alerted = scores >= FROZEN_THRESHOLD
    total_value = float(amounts[labels == 1].sum())
    captured_value = float(amounts[(labels == 1) & alerted].sum())
    metrics.update(
        {
            "threshold": FROZEN_THRESHOLD,
            "rows": len(labels),
            "fraud_rows": int(labels.sum()),
            "alert_count": int(alerted.sum()),
            "alert_rate": float(alerted.mean()),
            "fraudulent_value_captured": captured_value,
            "missed_fraud_value": float(amounts[(labels == 1) & ~alerted].sum()),
            "fraud_value_recall": captured_value / total_value if total_value else 0.0,
        }
    )
    workloads["false_negatives"] = int(labels.sum()) - workloads["fraud_cases_captured"]
    workloads["true_negatives"] = len(labels) - int(labels.sum()) - workloads["legitimate_alerts"]
    workloads["missed_fraud_value"] = total_value - workloads["fraudulent_value_captured"]
    workloads["f1"] = 2 * workloads["fraud_cases_captured"] / (workloads["alert_count"] + int(labels.sum()))
    workloads["accuracy"] = (workloads["fraud_cases_captured"] + workloads["true_negatives"]) / len(labels)
    workloads["balanced_accuracy"] = 0.5 * (
        workloads["fraud_count_recall"] + workloads["true_negatives"] / (len(labels) - int(labels.sum()))
    )
    return metrics, workloads
