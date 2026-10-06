"""Final protocol, history isolation, and frozen-threshold checks."""

# %% Imports and test fixtures
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from final_test_evaluation_utils import (
    FROZEN_THRESHOLD,
    STAGE_13_SUMMARY_RUN_ID,
    add_final_history_features,
    evaluate_frozen_test,
    freeze_final_protocol,
    reserve_final_evaluation,
)
from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH
from oversampling_experiment_utils import FEATURE_SET_NAME
from predictive_quality_utils import TRAIN_CODE, VALIDATION_CODE
from sender_location_history_utils import HISTORY_FEATURES, add_sender_location_history_features


# %% Frozen protocol and evaluation
class FinalProtocolTests(unittest.TestCase):
    def test_receipt_prevents_a_second_scoring_reservation(self) -> None:
        with TemporaryDirectory() as directory:
            receipt = Path(directory) / "evaluation.json"
            reserve_final_evaluation(receipt, {"run_id": "first"})
            before = receipt.read_text(encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reserve_final_evaluation(receipt, {"run_id": "second"})
            self.assertEqual(receipt.read_text(encoding="utf-8"), before)

    @staticmethod
    def summary() -> dict:
        return {
            "evidence_scope": "full_data",
            "test_split_evaluated": False,
            "feature_set": FEATURE_SET_NAME,
            "imbalance_strategy": "class_weight_reference",
            "decision": {"decision_eligible": True, "selected_configuration": "lgbm_reference"},
            "split_manifest": {"split_assignment_sha256": FROZEN_FULL_SPLIT_HASH},
            "candidate_results": [
                {
                    "configuration": "lgbm_reference",
                    "run_id": "reference",
                    "validation_threshold_at_max_f1": FROZEN_THRESHOLD,
                }
            ],
        }

    def test_protocol_rejects_changed_threshold_test_evidence_and_wrong_split(self) -> None:
        valid = self.summary()
        protocol = freeze_final_protocol(valid, STAGE_13_SUMMARY_RUN_ID)
        self.assertEqual(protocol["threshold"], FROZEN_THRESHOLD)
        with self.assertRaises(ValueError):
            freeze_final_protocol({**valid, "test_accuracy": 0.99}, STAGE_13_SUMMARY_RUN_ID)
        with self.assertRaises(ValueError):
            freeze_final_protocol(
                {**valid, "split_manifest": {"split_assignment_sha256": "changed"}}, STAGE_13_SUMMARY_RUN_ID
            )
        changed = self.summary()
        changed["candidate_results"][0]["validation_threshold_at_max_f1"] = 0.5
        with self.assertRaises(ValueError):
            freeze_final_protocol(changed, STAGE_13_SUMMARY_RUN_ID)

    def test_final_metrics_use_fixed_threshold_without_optimisation(self) -> None:
        protocol = freeze_final_protocol(self.summary(), STAGE_13_SUMMARY_RUN_ID)
        with patch("fraud_modeling_utils.best_f1_threshold", side_effect=AssertionError("Test tuning prohibited")):
            metrics, workloads = evaluate_frozen_test(
                [0, 0, 1, 1], np.array([0.1, 0.8, 0.7, 0.4]), [10, 20, 30, 40], protocol
            )
        self.assertEqual(
            (
                metrics["true_negatives"],
                metrics["false_positives"],
                metrics["false_negatives"],
                metrics["true_positives"],
            ),
            (1, 1, 1, 1),
        )
        self.assertEqual(metrics["missed_fraud_value"], 40)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual(len(workloads), 3)
        with self.assertRaises(ValueError):
            evaluate_frozen_test([0, 1], np.array([0.1, 0.8]), [10, 20], {**protocol, "threshold": 0.5})


# %% Training history is independent of query labels and query events
class FinalHistoryTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
                "event_timestamp": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-04"], utc=True
                ),
                "sender_account": ["s"] * 5,
                "location": ["a", "a", "b", "b", "a"],
                "is_fraud": [0, 1, 0, 1, 0],
            }
        )

    def test_final_history_matches_development_definition_and_excludes_equal_times(self) -> None:
        data = self.frame()
        mask = np.array([True, True, False, False, False])
        final = add_final_history_features(data, mask)
        development = add_sender_location_history_features(data, np.where(mask, TRAIN_CODE, VALIDATION_CODE))
        pd.testing.assert_frame_equal(final[HISTORY_FEATURES], development[HISTORY_FEATURES])
        self.assertEqual(final.loc[2, HISTORY_FEATURES[0]], 1)
        self.assertEqual(final.loc[3, HISTORY_FEATURES[1]], 0)
        self.assertEqual(final.loc[4, HISTORY_FEATURES[0]], 2)

    def test_query_labels_and_added_query_events_do_not_change_existing_history(self) -> None:
        data = self.frame()
        mask = np.array([True, True, False, False, False])
        first = add_final_history_features(data, mask)
        changed = data.copy()
        changed["is_fraud"] = 1 - changed["is_fraud"]
        second = add_final_history_features(changed, mask)
        pd.testing.assert_frame_equal(first[HISTORY_FEATURES], second[HISTORY_FEATURES])
        extra = data.iloc[[3]].copy()
        extra["transaction_id"] = "T6"
        extended = add_final_history_features(pd.concat([data, extra], ignore_index=True), np.append(mask, False))
        pd.testing.assert_frame_equal(first[HISTORY_FEATURES], extended.iloc[:5][HISTORY_FEATURES])


# %% Direct entry point
if __name__ == "__main__":
    unittest.main()
