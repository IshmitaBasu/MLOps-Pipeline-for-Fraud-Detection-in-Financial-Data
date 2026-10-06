"""Fixed voting candidates, Stage 12 handoff, and validation threshold reports."""

# %% Imports and frozen definitions
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier

from fraud_modeling_utils import best_f1_threshold, classification_metrics
from lightgbm_tuning_utils import frozen_lightgbm_configs, select_lightgbm_configuration
from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH, build_model_pipeline, frozen_model_family_configs
from oversampling_experiment_utils import FEATURE_SET_NAME

REFERENCE_MODEL = "lgbm_reference"
CANDIDATES = {
    REFERENCE_MODEL: ("lightgbm",),
    "soft_vote_lgbm_catboost": ("lightgbm", "catboost"),
    "soft_vote_lgbm_catboost_rf": ("lightgbm", "catboost", "random_forest_reference"),
}
FIXED_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# %% Frozen Stage 12 handoff
def validate_stage_12_handoff(summary: Mapping[str, Any]) -> dict[str, str]:
    """Require completed tuning evidence for the retained LightGBM reference."""

    if summary.get("test_split_evaluated") is not False:
        raise ValueError("The Stage 12 handoff must confirm that the test split was not evaluated.")
    if any(str(key).startswith("test_") and key != "test_split_evaluated" for key in summary):
        raise ValueError("The Stage 12 handoff contains prohibited test evidence.")
    if summary.get("evidence_scope") != "full_data":
        raise ValueError("Stage 13 requires full-data Stage 12 evidence.")
    if summary.get("feature_set") != FEATURE_SET_NAME or summary.get("imbalance_strategy") != "class_weight_reference":
        raise ValueError("Stage 12 features or imbalance treatment differ from the frozen Stage 13 inputs.")
    decision = summary.get("decision", {})
    if not isinstance(decision, Mapping) or decision.get("decision_eligible") is not True:
        raise ValueError("The Stage 12 decision is missing or ineligible.")
    if decision.get("selected_configuration") != REFERENCE_MODEL:
        raise ValueError("Stage 13 expects the retained Stage 12 LightGBM reference.")
    manifest = summary.get("split_manifest", {})
    if not isinstance(manifest, Mapping) or manifest.get("split_assignment_sha256") != FROZEN_FULL_SPLIT_HASH:
        raise ValueError("The Stage 12 handoff does not match the frozen split hash.")
    run_id = summary.get("summary_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("The Stage 12 summary run ID is required.")
    return {
        "summary_run_id": run_id,
        "selected_configuration": REFERENCE_MODEL,
        "split_assignment_sha256": FROZEN_FULL_SPLIT_HASH,
    }


# %% Candidate construction
def build_voting_candidate(name: str, *, positive_class_weight: float) -> Any:
    """Build actual sklearn soft voting with complete train-fitted pipelines."""

    if name not in CANDIDATES:
        raise ValueError(f"Unknown voting candidate: {name!r}.")
    configs = {config.name: config for config in frozen_model_family_configs()}
    configs["lightgbm"] = frozen_lightgbm_configs()[0].as_model_family_config()
    estimators = [
        (member, build_model_pipeline(configs[member], positive_class_weight=positive_class_weight))
        for member in CANDIDATES[name]
    ]
    if name == REFERENCE_MODEL:
        return estimators[0][1]
    # Fit members sequentially to avoid simultaneous copies of the full dataset.
    return VotingClassifier(estimators=estimators, voting="soft", weights=None, n_jobs=1)


# %% Threshold diagnostics
def validation_threshold_table(
    labels: pd.Series | np.ndarray, scores: np.ndarray, amounts: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Report a fixed grid and maximum-F1 rule, with actual confusion and value counts."""

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or amounts.ndim != 1:
        raise ValueError("Validation arrays must be one-dimensional.")
    if not (len(labels) == len(scores) == len(amounts)) or not len(labels):
        raise ValueError("Validation arrays must have equal non-zero lengths.")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Validation labels must contain both binary classes.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Voting scores must be finite and between zero and one.")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Transaction amounts must be finite and non-negative.")
    best_threshold, _ = best_f1_threshold(labels, scores)
    rules = [("fixed_score", threshold) for threshold in FIXED_THRESHOLDS]
    rules.append(("validation_max_f1", best_threshold))
    total_fraud_value = float(amounts[labels == 1].sum())
    rows = []
    for rule, threshold in rules:
        metrics = classification_metrics(labels, scores, threshold)
        alerted = scores >= threshold
        captured_value = float(amounts[(labels == 1) & alerted].sum())
        rows.append(
            {
                "rule": rule,
                "threshold": float(threshold),
                **metrics,
                "alert_count": int(alerted.sum()),
                "alert_rate": float(alerted.mean()),
                "fraudulent_value_captured": captured_value,
                "missed_fraud_value": float(amounts[(labels == 1) & ~alerted].sum()),
                "fraud_value_recall": captured_value / total_fraud_value if total_fraud_value else 0.0,
            }
        )
    return pd.DataFrame(rows)


# %% Validation-only model decision
def select_voting_candidate(results: Sequence[Mapping[str, Any]], *, evidence_scope: str) -> dict[str, Any]:
    """Reuse the established AP and equal-workload progression rule."""

    names = [row.get("configuration") for row in results]
    if len(names) != len(CANDIDATES) or set(names) != set(CANDIDATES):
        raise ValueError("All three predeclared voting comparison results are required exactly once.")
    decision = select_lightgbm_configuration(results, evidence_scope=evidence_scope)
    decision["recommendation"] = (
        "smoke_run_no_model_decision"
        if not decision["decision_eligible"]
        else "retain_lightgbm_reference" if decision["reference_retained"] else "progress_soft_voting_candidate"
    )
    decision["operating_rule_frozen"] = False
    return decision
