"""Frozen Stage 12 LightGBM configurations and selection safeguards."""

# %% Imports and constants
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np

from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH, ModelFamilyConfig

REFERENCE_CONFIGURATION: Final = "lgbm_reference"
AVERAGE_PRECISION_TIE_TOLERANCE: Final = 0.0001


@dataclass(frozen=True)
class LightGBMTuningConfig:
    """One deliberately limited LightGBM configuration."""

    name: str
    parameters: Mapping[str, Any]
    change_from_reference: str

    def as_model_family_config(self) -> ModelFamilyConfig:
        return ModelFamilyConfig(name=self.name, family="lightgbm", preprocessing="one_hot", parameters=self.parameters)


# %% Predeclared tuning configurations
def frozen_lightgbm_configs() -> tuple[LightGBMTuningConfig, ...]:
    """Return the small fixed search in execution order."""

    reference = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }

    def changed(**updates: Any) -> dict[str, Any]:
        return {**reference, **updates}

    return (
        LightGBMTuningConfig(REFERENCE_CONFIGURATION, reference, "Unchanged Stage 11 LightGBM reference."),
        LightGBMTuningConfig("lgbm_trees_500", changed(n_estimators=500), "Increase boosting rounds to 500."),
        LightGBMTuningConfig(
            "lgbm_slow_500", changed(n_estimators=500, learning_rate=0.03), "Use more, smaller boosting steps."
        ),
        LightGBMTuningConfig("lgbm_leaves_15", changed(num_leaves=15), "Reduce tree complexity."),
        LightGBMTuningConfig("lgbm_leaves_63", changed(num_leaves=63), "Increase tree complexity."),
        LightGBMTuningConfig("lgbm_min_child_50", changed(min_child_samples=50), "Allow smaller leaves."),
        LightGBMTuningConfig("lgbm_min_child_250", changed(min_child_samples=250), "Require larger leaves."),
        LightGBMTuningConfig(
            "lgbm_regularized", changed(reg_alpha=1.0, reg_lambda=5.0), "Increase L1 and L2 regularisation."
        ),
    )


# %% Stage 11 handoff validation
def validate_stage_11_handoff(summary: Mapping[str, Any]) -> dict[str, str]:
    """Accept only the complete full-data decision that selected LightGBM."""

    if summary.get("test_split_evaluated") is not False:
        raise ValueError("The Stage 11 handoff must confirm that the test split was not evaluated.")
    prohibited = [key for key in summary if str(key).lower().startswith("test_") and key != "test_split_evaluated"]
    if prohibited:
        raise ValueError(f"Stage 11 handoff contains prohibited test evidence: {prohibited}")
    if summary.get("evidence_scope") != "full_data":
        raise ValueError("Stage 12 requires full-data Stage 11 evidence.")
    decision = summary.get("decision")
    if not isinstance(decision, Mapping) or decision.get("decision_eligible") is not True:
        raise ValueError("The Stage 11 decision is missing or ineligible.")
    if decision.get("selected_model") != "lightgbm":
        raise ValueError("Stage 12 is frozen to the LightGBM model selected in Stage 11.")
    split_manifest = summary.get("split_manifest")
    if not isinstance(split_manifest, Mapping):
        raise ValueError("The Stage 11 split manifest is missing.")
    split_hash = split_manifest.get("split_assignment_sha256")
    if split_hash != FROZEN_FULL_SPLIT_HASH:
        raise ValueError("The Stage 11 handoff does not match the frozen split hash.")
    run_id = summary.get("summary_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("The Stage 11 summary run ID is required.")
    return {"selected_model": "lightgbm", "split_assignment_sha256": str(split_hash), "summary_run_id": run_id}


# %% Validation-only selection rule
def _metric(result: Mapping[str, Any], name: str) -> float:
    if name not in result:
        raise ValueError(f"Tuning result is missing required metric {name!r}.")
    value = float(result[name])
    if not np.isfinite(value):
        raise ValueError(f"Tuning metric {name!r} must be finite.")
    return value


def select_lightgbm_configuration(results: Sequence[Mapping[str, Any]], *, evidence_scope: str) -> dict[str, Any]:
    """Choose a full-data candidate only when it improves the reference safely."""

    if not results:
        raise ValueError("At least one tuning result is required.")
    for result in results:
        prohibited = [key for key in result if str(key).lower().startswith("test_")]
        if prohibited:
            raise ValueError(f"Tuning selection received prohibited test evidence: {prohibited}")
    if evidence_scope != "full_data":
        return {
            "decision_eligible": False,
            "selected_configuration": None,
            "recommendation": "smoke_run_no_tuning_decision",
            "candidate_decisions": [],
        }

    references = [result for result in results if result.get("configuration") == REFERENCE_CONFIGURATION]
    if len(references) != 1:
        raise ValueError("Exactly one LightGBM reference result is required.")
    reference = references[0]
    reference_ap = _metric(reference, "validation_average_precision")
    reference_count = _metric(reference, "workload_5pct_fraud_count_recall")
    reference_value = _metric(reference, "workload_5pct_fraud_value_recall")

    passed: list[Mapping[str, Any]] = []
    candidate_decisions: list[dict[str, Any]] = []
    for candidate in results:
        if candidate is reference:
            continue
        ap = _metric(candidate, "validation_average_precision")
        count_recall = _metric(candidate, "workload_5pct_fraud_count_recall")
        value_recall = _metric(candidate, "workload_5pct_fraud_value_recall")
        passes = bool(
            ap > reference_ap
            and count_recall >= reference_count
            and value_recall >= reference_value
            and (count_recall > reference_count or value_recall > reference_value)
        )
        candidate_decisions.append({"configuration": candidate.get("configuration"), "passes_progression_rule": passes})
        if passes:
            passed.append(candidate)

    selected: Mapping[str, Any] = reference
    for candidate in passed:
        candidate_ap = _metric(candidate, "validation_average_precision")
        selected_ap = _metric(selected, "validation_average_precision")
        if candidate_ap > selected_ap + AVERAGE_PRECISION_TIE_TOLERANCE:
            selected = candidate
            continue
        if abs(candidate_ap - selected_ap) <= AVERAGE_PRECISION_TIE_TOLERANCE:
            candidate_value = _metric(candidate, "workload_5pct_fraud_value_recall")
            selected_value = _metric(selected, "workload_5pct_fraud_value_recall")
            if candidate_value > selected_value or (
                candidate_value == selected_value
                and _metric(candidate, "validation_inference_seconds")
                < _metric(selected, "validation_inference_seconds")
            ):
                selected = candidate

    selected_name = str(selected["configuration"])
    return {
        "decision_eligible": True,
        "selected_configuration": selected_name,
        "reference_configuration": REFERENCE_CONFIGURATION,
        "reference_retained": selected_name == REFERENCE_CONFIGURATION,
        "recommendation": (
            "retain_stage_11_lightgbm_reference"
            if selected_name == REFERENCE_CONFIGURATION
            else "retain_tuned_lightgbm_configuration"
        ),
        "candidate_decisions": candidate_decisions,
    }
