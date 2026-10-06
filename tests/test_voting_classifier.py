"""Voting arithmetic, train-only preprocessing, threshold reports, and handoff tests."""

# %% Imports
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier

from fraud_modeling_utils import CATEGORICAL_FEATURES
from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH
from oversampling_experiment_utils import FEATURE_SET_NAME, MODEL_NUMERIC_FEATURES
from voting_classifier_utils import (
    CANDIDATES,
    REFERENCE_MODEL,
    build_voting_candidate,
    select_voting_candidate,
    validate_stage_12_handoff,
    validation_threshold_table,
)


# %% Real estimator compatibility and averaging
class VotingTests(unittest.TestCase):
    def test_real_voting_pipelines_fit_and_average_member_probabilities(self) -> None:
        rows = 40
        frame = pd.DataFrame({column: np.arange(rows, dtype=float) for column in MODEL_NUMERIC_FEATURES})
        for column in CATEGORICAL_FEATURES:
            frame[column] = ["a", "b"] * (rows // 2)
        labels = np.array([0, 0, 0, 1] * (rows // 4))
        for name in tuple(CANDIDATES)[1:]:
            with self.subTest(candidate=name):
                model = build_voting_candidate(name, positive_class_weight=3.0)
                self.assertIsInstance(model, VotingClassifier)
                self.assertEqual(model.n_jobs, 1)
                updates = {"lightgbm__model__n_estimators": 3, "catboost__model__iterations": 3}
                if len(model.estimators) == 3:
                    updates["random_forest_reference__model__n_estimators"] = 3
                model.set_params(**updates)
                model.fit(frame, labels)
                expected = np.mean([member.predict_proba(frame) for member in model.estimators_], axis=0)
                np.testing.assert_allclose(model.predict_proba(frame), expected)


# %% Threshold evidence and validation boundaries
class ThresholdTests(unittest.TestCase):
    def test_thresholds_report_actual_confusion_counts_and_missed_fraud_value(self) -> None:
        table = validation_threshold_table([0, 0, 1, 1], np.array([0.1, 0.8, 0.7, 0.6]), [10, 20, 30, 40])
        row = table.loc[(table.rule == "fixed_score") & np.isclose(table.threshold, 0.7)].iloc[0]
        self.assertEqual(
            (row.true_negatives, row.false_positives, row.false_negatives, row.true_positives), (1, 1, 1, 1)
        )
        self.assertEqual(row.missed_fraud_value, 40)
        self.assertAlmostEqual(row.fraud_value_recall, 30 / 70)
        best = table.loc[table.rule == "validation_max_f1"].iloc[0]
        self.assertAlmostEqual(best.threshold, 0.6)
        self.assertAlmostEqual(best.f1, 0.8)

    def test_thresholds_reject_malformed_scores_and_amounts(self) -> None:
        with self.assertRaises(ValueError):
            validation_threshold_table([0, 1], np.array([0.2, np.nan]), [10, 20])
        with self.assertRaises(ValueError):
            validation_threshold_table([0, 1], np.array([0.2, 0.5]), [10, -20])


# %% Frozen handoff and selection
class DecisionTests(unittest.TestCase):
    def test_handoff_accepts_full_reference_and_rejects_changed_protocol(self) -> None:
        valid = {
            "test_split_evaluated": False,
            "evidence_scope": "full_data",
            "feature_set": FEATURE_SET_NAME,
            "imbalance_strategy": "class_weight_reference",
            "summary_run_id": "stage12-run",
            "decision": {"decision_eligible": True, "selected_configuration": REFERENCE_MODEL},
            "split_manifest": {"split_assignment_sha256": FROZEN_FULL_SPLIT_HASH},
        }
        self.assertEqual(validate_stage_12_handoff(valid)["selected_configuration"], REFERENCE_MODEL)
        for field, value in (
            ("evidence_scope", "smoke_test"),
            ("test_split_evaluated", True),
            ("feature_set", "changed"),
            ("test_average_precision", 1.0),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_stage_12_handoff({**valid, field: value})

    @staticmethod
    def results() -> list[dict[str, object]]:
        return [
            {
                "configuration": name,
                "validation_average_precision": 0.04 + index * 0.01,
                "workload_5pct_fraud_count_recall": 0.06 + index * 0.01,
                "workload_5pct_fraud_value_recall": 0.07 + index * 0.01,
                "validation_inference_seconds": 1 + index,
            }
            for index, name in enumerate(CANDIDATES)
        ]

    def test_smoke_is_ineligible_and_full_can_select_improved_voting(self) -> None:
        smoke = select_voting_candidate(self.results(), evidence_scope="smoke_test")
        self.assertIsNone(smoke["selected_configuration"])
        full = select_voting_candidate(self.results(), evidence_scope="full_data")
        self.assertEqual(full["selected_configuration"], "soft_vote_lgbm_catboost_rf")
        self.assertFalse(full["operating_rule_frozen"])

    def test_higher_ap_with_lower_value_recall_does_not_replace_reference(self) -> None:
        results = self.results()
        for result in results[1:]:
            result["workload_5pct_fraud_value_recall"] = 0.01
        decision = select_voting_candidate(results, evidence_scope="full_data")
        self.assertEqual(decision["selected_configuration"], REFERENCE_MODEL)

    def test_selection_rejects_test_evidence_and_missing_candidates(self) -> None:
        results = self.results()
        results[1]["test_average_precision"] = 0.99
        with self.assertRaises(ValueError):
            select_voting_candidate(results, evidence_scope="full_data")
        with self.assertRaises(ValueError):
            select_voting_candidate(self.results()[:1], evidence_scope="full_data")


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
