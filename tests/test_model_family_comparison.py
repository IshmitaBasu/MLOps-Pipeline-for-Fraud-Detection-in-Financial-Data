"""Tests for the frozen Stage 11 comparison rules."""

# %% Imports
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from model_family_comparison_utils import (
    FROZEN_FULL_SPLIT_HASH,
    REFERENCE_MODEL,
    build_model_pipeline,
    frozen_model_family_configs,
    select_model_family,
    stratified_sample_indices,
    validate_stage_10_handoff,
)


# %% Configuration tests
class ConfigurationTests(unittest.TestCase):
    def test_frozen_candidates_include_requested_families(self) -> None:
        configs = frozen_model_family_configs()
        self.assertEqual(
            [config.name for config in configs],
            [
                REFERENCE_MODEL,
                "hist_gradient_boosting",
                "lightgbm",
                "xgboost",
                "catboost",
                "linear_svm",
                "knn_sampled_feasibility",
            ],
        )
        self.assertTrue(all(config.full_data_candidate for config in configs[:-1]))
        self.assertFalse(configs[-1].full_data_candidate)

    def test_all_pinned_model_pipelines_build_with_class_weight_handoff(self) -> None:
        pipelines = {
            config.name: build_model_pipeline(config, positive_class_weight=25.0)
            for config in frozen_model_family_configs()
        }
        self.assertEqual(pipelines[REFERENCE_MODEL].named_steps["model"].class_weight, "balanced_subsample")
        self.assertEqual(pipelines["hist_gradient_boosting"].named_steps["model"].class_weight, "balanced")
        self.assertEqual(pipelines["xgboost"].named_steps["model"].scale_pos_weight, 25.0)
        self.assertEqual(pipelines["lightgbm"].named_steps["model"].scale_pos_weight, 25.0)
        self.assertEqual(pipelines["linear_svm"].named_steps["model"].class_weight, "balanced")

    def test_runner_has_cells_and_no_test_frame_parameter(self) -> None:
        source = Path("11_model_family_comparison_with_mlflow.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("# %%"), 3)
        self.assertNotIn("test_df:", source)
        self.assertIn("The test split will not be materialised or evaluated.", source)

    def test_stage_10_handoff_requires_full_validated_decision(self) -> None:
        valid = {
            "evidence_scope": "full_data",
            "decision": {"decision_eligible": True, "selected_strategy": "random_oversample_5_to_1"},
            "frozen_full_split_hash": FROZEN_FULL_SPLIT_HASH,
            "summary_run_id": "stage10-run",
            "test_split_evaluated": False,
        }
        handoff = validate_stage_10_handoff(valid)
        self.assertEqual(handoff["selected_strategy"], "random_oversample_5_to_1")

        for key, value in {
            "evidence_scope": "smoke_test",
            "frozen_full_split_hash": "changed",
            "summary_run_id": "",
        }.items():
            invalid = dict(valid)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_stage_10_handoff(invalid)
        for key, value in {"decision_eligible": False, "selected_strategy": "invented_strategy"}.items():
            invalid = {**valid, "decision": {**valid["decision"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_stage_10_handoff(invalid)

    def test_stage_10_handoff_rejects_test_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "test evidence"):
            validate_stage_10_handoff(
                {
                    "evidence_scope": "full_data",
                    "decision": {"decision_eligible": True, "selected_strategy": "class_weight_reference"},
                    "frozen_full_split_hash": FROZEN_FULL_SPLIT_HASH,
                    "summary_run_id": "stage10-run",
                    "test_split_evaluated": False,
                    "test_average_precision": 1.0,
                }
            )

    def test_stage_10_handoff_rejects_test_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "not evaluated"):
            validate_stage_10_handoff(
                {
                    "evidence_scope": "full_data",
                    "decision": {"decision_eligible": True, "selected_strategy": "class_weight_reference"},
                    "frozen_full_split_hash": FROZEN_FULL_SPLIT_HASH,
                    "summary_run_id": "stage10-run",
                    "test_split_evaluated": True,
                }
            )


# %% Sampling tests
class SamplingTests(unittest.TestCase):
    def test_knn_sample_is_deterministic_and_stratified(self) -> None:
        labels = np.array([0] * 900 + [1] * 100, dtype=np.int8)
        first = stratified_sample_indices(labels, 200)
        second = stratified_sample_indices(labels, 200)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 200)
        self.assertEqual(int(labels[first].sum()), 20)

    def test_knn_sampling_rejects_invalid_requests(self) -> None:
        with self.assertRaises(ValueError):
            stratified_sample_indices([0, 0, 0], 2)
        with self.assertRaises(ValueError):
            stratified_sample_indices([0, 1], 3)


# %% Selection tests
class SelectionTests(unittest.TestCase):
    @staticmethod
    def result(
        model: str,
        ap: float,
        count_recall: float,
        value_recall: float,
        inference_seconds: float,
        *,
        full_data_candidate: bool = True,
        feasible: bool = True,
    ) -> dict[str, object]:
        return {
            "model": model,
            "full_data_candidate": full_data_candidate,
            "feasible": feasible,
            "validation_average_precision": ap,
            "workload_5pct_fraud_count_recall": count_recall,
            "workload_5pct_fraud_value_recall": value_recall,
            "validation_inference_seconds": inference_seconds,
        }

    def test_smoke_results_cannot_select_model(self) -> None:
        decision = select_model_family(
            [self.result(REFERENCE_MODEL, 0.04, 0.05, 0.04, 2.0)], evidence_scope="smoke_test"
        )
        self.assertFalse(decision["decision_eligible"])
        self.assertIsNone(decision["selected_model"])

    def test_selection_requires_operational_improvement(self) -> None:
        results = [
            self.result(REFERENCE_MODEL, 0.0400, 0.050, 0.040, 2.0),
            self.result("higher_ap_but_lower_value", 0.0600, 0.060, 0.030, 1.0),
            self.result("valid_candidate", 0.0500, 0.051, 0.045, 3.0),
            self.result("knn_sampled_feasibility", 0.9000, 0.900, 0.900, 100.0, full_data_candidate=False),
        ]
        decision = select_model_family(results, evidence_scope="full_data")
        self.assertEqual(decision["selected_model"], "valid_candidate")

    def test_tie_breaks_on_value_then_inference_time(self) -> None:
        results = [
            self.result(REFERENCE_MODEL, 0.0400, 0.050, 0.040, 5.0),
            self.result("candidate_a", 0.05000, 0.051, 0.050, 4.0),
            self.result("candidate_b", 0.05005, 0.051, 0.060, 6.0),
            self.result("candidate_c", 0.05004, 0.051, 0.060, 2.0),
        ]
        decision = select_model_family(results, evidence_scope="full_data")
        self.assertEqual(decision["selected_model"], "candidate_c")

    def test_selection_rejects_test_metrics(self) -> None:
        result = self.result(REFERENCE_MODEL, 0.04, 0.05, 0.04, 2.0)
        result["test_average_precision"] = 1.0
        with self.assertRaisesRegex(ValueError, "test evidence"):
            select_model_family([result], evidence_scope="full_data")


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
