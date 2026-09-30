"""Safeguards for the controlled sender-location history experiment."""

# %% Imports and numbered-script loading
from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predictive_quality_utils import TEST_CODE, TRAIN_CODE, VALIDATION_CODE  # noqa: E402
from sender_location_history_utils import (  # noqa: E402
    HISTORY_FEATURES,
    add_sender_location_history_features,
    candidate_progression_decision,
    fixed_workload_table,
)


def load_stage_08_module():
    script_path = PROJECT_DIR / "08_sender_location_history_feature_experiment_with_mlflow.py"
    spec = importlib.util.spec_from_file_location("stage_08_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# %% Point-in-time feature safeguards
class SenderLocationHistoryTests(unittest.TestCase):
    def test_validation_rows_read_but_do_not_update_training_history(self) -> None:
        data = pd.DataFrame(
            {
                "transaction_id": ["t1", "v1", "v2", "t2", "v3", "t3"],
                "event_timestamp": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-02"]
                ),
                "sender_account": ["A", "A", "A", "A", "A", "B"],
                "location": ["X", "X", "Y", "Y", "Y", "Z"],
            }
        )
        partition = np.array(
            [TRAIN_CODE, VALIDATION_CODE, VALIDATION_CODE, TRAIN_CODE, VALIDATION_CODE, TRAIN_CODE], dtype=np.uint8
        )

        enriched = add_sender_location_history_features(data, partition)

        self.assertEqual(enriched[HISTORY_FEATURES[0]].tolist(), [0, 1, 1, 1, 2, 0])
        self.assertEqual(enriched[HISTORY_FEATURES[1]].tolist(), [0, 1, 0, 0, 1, 0])
        self.assertEqual(enriched[HISTORY_FEATURES[2]].tolist(), [0, 0, 1, 1, 0, 0])

    def test_equal_timestamp_training_row_is_not_counted_as_past(self) -> None:
        data = pd.DataFrame(
            {
                "transaction_id": ["first", "same-train", "same-validation"],
                "event_timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"]),
                "sender_account": ["A", "A", "A"],
                "location": ["X", "X", "X"],
            }
        )
        partition = np.array([TRAIN_CODE, TRAIN_CODE, VALIDATION_CODE], dtype=np.uint8)
        enriched = add_sender_location_history_features(data, partition)
        self.assertEqual(enriched[HISTORY_FEATURES[0]].tolist(), [0, 1, 1])
        self.assertEqual(enriched[HISTORY_FEATURES[1]].tolist(), [0, 1, 1])

    def test_feature_builder_rejects_test_rows(self) -> None:
        data = pd.DataFrame(
            {
                "transaction_id": ["train", "validation", "test"],
                "event_timestamp": pd.date_range("2025-01-01", periods=3),
                "sender_account": ["A", "A", "A"],
                "location": ["X", "X", "X"],
            }
        )
        partition = np.array([TRAIN_CODE, VALIDATION_CODE, TEST_CODE], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "training and validation"):
            add_sender_location_history_features(data, partition)


# %% Equal-workload and decision safeguards
class WorkloadEvaluationTests(unittest.TestCase):
    def test_fixed_workload_uses_exact_alert_count_and_fraud_value(self) -> None:
        labels = pd.Series([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.5, 0.4, 0.3, 0.2, 0.0])
        amounts = pd.Series([100.0, 50.0, 300.0, 20.0, 600.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        table = fixed_workload_table(labels, scores, amounts, alert_rates=(0.5,))
        row = table.iloc[0]

        self.assertEqual(row["alert_count"], 5)
        self.assertEqual(row["fraud_cases_captured"], 2)
        self.assertAlmostEqual(row["fraud_count_recall"], 2 / 3)
        self.assertAlmostEqual(row["fraud_value_recall"], 0.4)

    def test_candidate_progresses_only_when_ranking_and_operations_improve(self) -> None:
        control = pd.DataFrame(
            {"requested_alert_rate": [0.05], "fraud_count_recall": [0.10], "fraud_value_recall": [0.12]}
        )
        candidate = pd.DataFrame(
            {"requested_alert_rate": [0.05], "fraud_count_recall": [0.11], "fraud_value_recall": [0.13]}
        )
        decision = candidate_progression_decision(0.04, 0.041, control, candidate)
        self.assertTrue(decision["candidate_progresses"])
        self.assertFalse(decision["test_split_evaluated"])


# %% API-level test isolation
class Stage08ApiTests(unittest.TestCase):
    def test_development_apis_do_not_accept_test_frames(self) -> None:
        stage_08 = load_stage_08_module()
        for callable_object in (stage_08.build_development_frames, stage_08.run_feature_comparison):
            parameter_names = set(inspect.signature(callable_object).parameters)
            self.assertNotIn("test_data", parameter_names)
            self.assertNotIn("test_df", parameter_names)

    def test_model_configuration_is_identical_between_feature_groups(self) -> None:
        stage_08 = load_stage_08_module()
        control = stage_08.build_fixed_random_forest("original_v1", 42)
        candidate = stage_08.build_fixed_random_forest("sender_location_history_v1", 42)
        control_params = control.named_steps["model"].get_params()
        candidate_params = candidate.named_steps["model"].get_params()
        self.assertEqual(control_params, candidate_params)

    def test_raw_source_order_is_restored_before_split_creation(self) -> None:
        stage_08 = load_stage_08_module()
        modeling_table = pd.DataFrame({"transaction_id": ["T100002", "T100000", "T100001"], "is_fraud": [0, 0, 1]})
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.csv"
            pd.DataFrame({"transaction_id": ["T100000", "T100001", "T100002"], "unused": [1, 2, 3]}).to_csv(
                raw_path, index=False
            )
            ordered = stage_08.restore_raw_source_order(modeling_table, raw_path)

        self.assertEqual(ordered["transaction_id"].tolist(), ["T100000", "T100001", "T100002"])
        self.assertEqual(ordered["is_fraud"].tolist(), [0, 1, 0])


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
