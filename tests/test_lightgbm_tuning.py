"""Tests for the frozen Stage 12 LightGBM tuning rules."""

# %% Imports
from __future__ import annotations

import unittest
from pathlib import Path

from lightgbm_tuning_utils import (
    REFERENCE_CONFIGURATION,
    frozen_lightgbm_configs,
    select_lightgbm_configuration,
    validate_stage_11_handoff,
)
from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH


# %% Configuration and handoff tests
class ConfigurationTests(unittest.TestCase):
    def test_search_is_small_fixed_and_lightgbm_only(self) -> None:
        configs = frozen_lightgbm_configs()
        self.assertEqual(configs[0].name, REFERENCE_CONFIGURATION)
        self.assertEqual(len(configs), 8)
        self.assertEqual(len({config.name for config in configs}), len(configs))
        self.assertTrue(all(config.as_model_family_config().family == "lightgbm" for config in configs))

    def test_runner_has_cells_and_no_test_frame_parameter(self) -> None:
        source = Path("12_lightgbm_tuning_with_mlflow.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("# %%"), 3)
        self.assertNotIn("test_df:", source)
        self.assertIn("The test split will not be materialised or evaluated.", source)

    def test_stage_11_handoff_requires_selected_lightgbm_and_frozen_split(self) -> None:
        valid = {
            "evidence_scope": "full_data",
            "decision": {"decision_eligible": True, "selected_model": "lightgbm"},
            "split_manifest": {"split_assignment_sha256": FROZEN_FULL_SPLIT_HASH},
            "summary_run_id": "stage11-run",
            "test_split_evaluated": False,
        }
        handoff = validate_stage_11_handoff(valid)
        self.assertEqual(handoff["selected_model"], "lightgbm")

        invalid = {**valid, "decision": {"decision_eligible": True, "selected_model": "catboost"}}
        with self.assertRaisesRegex(ValueError, "LightGBM"):
            validate_stage_11_handoff(invalid)
        invalid = {**valid, "split_manifest": {"split_assignment_sha256": "changed"}}
        with self.assertRaisesRegex(ValueError, "frozen split"):
            validate_stage_11_handoff(invalid)

    def test_stage_11_handoff_rejects_test_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "test evidence"):
            validate_stage_11_handoff(
                {
                    "evidence_scope": "full_data",
                    "decision": {"decision_eligible": True, "selected_model": "lightgbm"},
                    "split_manifest": {"split_assignment_sha256": FROZEN_FULL_SPLIT_HASH},
                    "summary_run_id": "stage11-run",
                    "test_split_evaluated": False,
                    "test_average_precision": 1.0,
                }
            )


# %% Selection tests
class SelectionTests(unittest.TestCase):
    @staticmethod
    def result(name: str, ap: float, count: float, value: float, seconds: float) -> dict[str, float | str]:
        return {
            "configuration": name,
            "validation_average_precision": ap,
            "workload_5pct_fraud_count_recall": count,
            "workload_5pct_fraud_value_recall": value,
            "validation_inference_seconds": seconds,
        }

    def test_smoke_cannot_select_configuration(self) -> None:
        decision = select_lightgbm_configuration(
            [self.result(REFERENCE_CONFIGURATION, 0.04, 0.05, 0.06, 1.0)], evidence_scope="smoke_test"
        )
        self.assertFalse(decision["decision_eligible"])
        self.assertIsNone(decision["selected_configuration"])

    def test_candidate_must_not_reduce_either_operational_recall(self) -> None:
        results = [
            self.result(REFERENCE_CONFIGURATION, 0.040, 0.050, 0.060, 2.0),
            self.result("higher_ap_lower_value", 0.060, 0.060, 0.050, 1.0),
            self.result("safe_improvement", 0.050, 0.051, 0.061, 3.0),
        ]
        decision = select_lightgbm_configuration(results, evidence_scope="full_data")
        self.assertEqual(decision["selected_configuration"], "safe_improvement")

    def test_reference_is_retained_when_no_candidate_passes(self) -> None:
        results = [
            self.result(REFERENCE_CONFIGURATION, 0.040, 0.050, 0.060, 2.0),
            self.result("lower_ap", 0.039, 0.060, 0.070, 1.0),
        ]
        decision = select_lightgbm_configuration(results, evidence_scope="full_data")
        self.assertTrue(decision["reference_retained"])

    def test_selection_rejects_test_metrics(self) -> None:
        result = self.result(REFERENCE_CONFIGURATION, 0.04, 0.05, 0.06, 2.0)
        result["test_average_precision"] = 1.0
        with self.assertRaisesRegex(ValueError, "test evidence"):
            select_lightgbm_configuration([result], evidence_scope="full_data")


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
