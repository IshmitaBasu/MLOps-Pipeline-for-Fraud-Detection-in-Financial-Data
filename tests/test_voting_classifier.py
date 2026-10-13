"""Voting arithmetic, train-only preprocessing, threshold reports, and handoff tests."""

# %% Imports
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier

from fraud_detection.common.modeling import CATEGORICAL_FEATURES
from fraud_detection.modeling.model_families import FROZEN_FULL_SPLIT_HASH
from fraud_detection.modeling.oversampling import FEATURE_SET_NAME, MODEL_NUMERIC_FEATURES
from fraud_detection.modeling.voting import (
    CANDIDATES,
    HARD_CANDIDATES,
    REFERENCE_MODEL,
    build_voting_candidate,
    hard_label_metrics,
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
        for name in (*tuple(CANDIDATES)[1:], *tuple(HARD_CANDIDATES)[1:]):
            with self.subTest(candidate=name):
                model = build_voting_candidate(name, positive_class_weight=3.0)
                self.assertIsInstance(model, VotingClassifier)
                self.assertEqual(model.n_jobs, 1)
                updates = {"lightgbm__model__n_estimators": 3, "catboost__model__iterations": 3}
                if len(model.estimators) == 3:
                    updates["random_forest_reference__model__n_estimators"] = 3
                model.set_params(**updates)
                model.fit(frame, labels)
                if model.voting == "soft":
                    expected = np.mean([member.predict_proba(frame) for member in model.estimators_], axis=0)
                    np.testing.assert_allclose(model.predict_proba(frame), expected)
                else:
                    votes = model.transform(frame)
                    expected = (votes.sum(axis=1) > votes.shape[1] / 2).astype(int)
                    np.testing.assert_array_equal(model.predict(frame), expected)
                    self.assertFalse(hasattr(model, "predict_proba"))


# %% Hard voting, decision metrics and isolated runner evidence
class HardVotingTests(unittest.TestCase):
    def test_two_member_tie_is_legitimate_and_three_member_majority_is_fraud(self) -> None:
        features, labels = np.zeros((4, 1)), [0, 1, 0, 1]
        members = [
            ("a", DummyClassifier(strategy="constant", constant=0)),
            ("b", DummyClassifier(strategy="constant", constant=1)),
        ]
        model = VotingClassifier(members, voting="hard").fit(features, labels)
        np.testing.assert_array_equal(model.predict(features), [0, 0, 0, 0])
        model = VotingClassifier(
            members + [("c", DummyClassifier(strategy="constant", constant=1))], voting="hard"
        ).fit(features, labels)
        np.testing.assert_array_equal(model.predict(features), [1, 1, 1, 1])

    def test_binary_metrics_report_confusion_and_value_without_ranking(self) -> None:
        metrics = hard_label_metrics([0, 0, 1, 1], [0, 1, 0, 1], [10, 20, 30, 40])
        for key in ("true_negatives", "false_positives", "false_negatives", "true_positives"):
            self.assertEqual(metrics[key], 1)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["fraud_value_recall"], 40 / 70)
        self.assertEqual(metrics["missed_fraud_value"], 30)
        self.assertNotIn("pr_auc", metrics)
        self.assertNotIn("roc_auc", metrics)
        zero = hard_label_metrics([0, 1], [0, 0], [0, 0])
        self.assertEqual(zero["precision"], 0)
        self.assertEqual(zero["fraud_value_recall"], 0)

    def test_binary_metrics_reject_invalid_inputs(self) -> None:
        cases = [
            ([0, 1], [0.2, 0.8], [10, 20]),
            ([0, 1], [0, 1], [10, -20]),
            ([0, 1], [0, 1], [10, np.nan]),
            ([0, 0], [0, 1], [10, 20]),
            ([0, 1], [0], [10, 20]),
            ([], [], []),
            ([[0, 1]], [0, 1], [10, 20]),
        ]
        for labels, predictions, amounts in cases:
            with self.subTest(labels=labels, predictions=predictions), self.assertRaises(ValueError):
                hard_label_metrics(labels, predictions, amounts)

    def test_hard_runner_pairs_same_members_and_cannot_promote_model(self) -> None:
        path = Path(__file__).resolve().parents[1] / "stages/13_voting_classifier_comparison_with_mlflow.py"
        spec = importlib.util.spec_from_file_location("voting_runner_test", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        frame = pd.DataFrame({column: [0.0] * 4 for column in runner.MODEL_FEATURES})
        frame[runner.TARGET_COLUMN] = [0, 0, 1, 1]
        frame["amount"] = [10, 20, 30, 40]
        model = VotingClassifier(
            [
                ("a", DummyClassifier(strategy="constant", constant=0)),
                ("b", DummyClassifier(strategy="constant", constant=1)),
            ],
            voting="hard",
        )
        fake_tracking = MagicMock()
        with (
            patch.object(runner, "mlflow", fake_tracking),
            patch.object(runner, "tracked_source_files", return_value=[]),
            patch.object(runner, "build_voting_candidate", return_value=model),
        ):
            result = runner.run_hard_candidate(
                "hard_vote_lgbm_catboost",
                train_df=frame,
                validation_df=frame,
                split_manifest={"split_assignment_sha256": "smoke-hash"},
                handoff={"summary_run_id": "stage12"},
                evidence_scope="smoke_test",
                class_weight_ratio=1,
                dataset_info={},
                feature_lineage={},
            )
        self.assertEqual(result["tie_count"], 4)
        self.assertEqual(result["classification"]["alert_count"], 0)
        self.assertEqual(result["policy_results"][1]["alert_count"], 0)
        self.assertEqual(result["policy_results"][2]["alert_count"], 4)
        self.assertFalse(result["test_split_evaluated"])
        self.assertFalse(result["frozen_model_changed"])
        self.assertEqual(fake_tracking.set_tags.call_args.args[0]["eligible_for_model_selection"], "false")
        self.assertNotIn("validation_average_precision", fake_tracking.log_metrics.call_args_list[0].args[0])
        self.assertIn("hard_voting_result.json", [call.args[1] for call in fake_tracking.log_dict.call_args_list])


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
