"""Tests for the Stage 07 predictive-quality diagnostic safeguards."""

# %% Imports and numbered-script loading
from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predictive_quality_utils import (  # noqa: E402
    TARGET_COLUMN,
    build_split_manifest,
    entity_history_diagnostic,
    make_stratified_random_partition,
    past_transaction_count,
    run_training_diagnostics,
    training_partition,
)


def load_stage_07_module():
    script_path = PROJECT_DIR / "07_predictive_quality_improvement_with_mlflow.py"
    spec = importlib.util.spec_from_file_location("stage_07_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# %% Reusable test data
def split_fixture(rows: int = 1_000) -> pd.DataFrame:
    fraud = np.tile([0] * 24 + [1], rows // 25)
    return pd.DataFrame(
        {
            "transaction_id": [f"tx-{index:04d}" for index in range(rows)],
            "event_timestamp": pd.date_range("2025-01-01", periods=rows, freq="min"),
            TARGET_COLUMN: fraud,
        }
    )


# %% Split and test-isolation tests
class SplitSafeguardTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_exact(self) -> None:
        data = split_fixture()
        first = make_stratified_random_partition(data[TARGET_COLUMN], seed=42)
        second = make_stratified_random_partition(data[TARGET_COLUMN], seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(int((first == 0).sum()), 700)
        self.assertEqual(int((first == 1).sum()), 150)
        self.assertEqual(int((first == 2).sum()), 150)
        overall_rate = data[TARGET_COLUMN].mean()
        for split_code in (0, 1, 2):
            split_rate = data.loc[first == split_code, TARGET_COLUMN].mean()
            self.assertAlmostEqual(split_rate, overall_rate, places=2)

    def test_manifest_and_training_partition_keep_test_rows_separate(self) -> None:
        data = split_fixture()
        partition = make_stratified_random_partition(data[TARGET_COLUMN], seed=42)
        manifest = build_split_manifest(data, partition, 42, 0.70, 0.15)
        training_data = training_partition(data, partition)
        test_ids = set(data.loc[partition == 2, "transaction_id"])

        self.assertFalse(manifest["test_split_evaluated"])
        self.assertEqual(manifest["splits"]["test"]["rows"], 150)
        self.assertTrue(set(training_data["transaction_id"]).isdisjoint(test_ids))


# %% Past-only feature tests
class PastOnlyHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "transaction_id": ["later", "first", "other", "middle"],
                "event_timestamp": pd.to_datetime(
                    [
                        "2025-01-03",
                        "2025-01-01",
                        "2025-01-02",
                        "2025-01-02",
                    ]
                ),
                "sender_account": ["A", "A", "B", "A"],
                TARGET_COLUMN: [1, 0, 1, 0],
            }
        )

    def test_current_and_future_rows_are_excluded(self) -> None:
        counts = past_transaction_count(self.data, ("sender_account",))
        self.assertEqual(counts.tolist(), [2, 0, 0, 1])

    def test_target_changes_do_not_change_history_feature(self) -> None:
        original = past_transaction_count(self.data, ("sender_account",))
        relabelled = self.data.copy()
        relabelled[TARGET_COLUMN] = 1 - relabelled[TARGET_COLUMN]
        changed = past_transaction_count(relabelled, ("sender_account",))
        pd.testing.assert_series_equal(original, changed)

    def test_transactions_at_the_same_timestamp_are_not_past(self) -> None:
        simultaneous = pd.DataFrame(
            {
                "transaction_id": ["first", "same-a", "same-b"],
                "event_timestamp": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-02"]
                ),
                "sender_account": ["A", "A", "A"],
            }
        )
        counts = past_transaction_count(simultaneous, ("sender_account",))
        self.assertEqual(counts.tolist(), [0, 1, 1])


# %% API-level leakage test
class DiagnosticApiTests(unittest.TestCase):
    def test_diagnostic_apis_do_not_accept_validation_or_test_frames(self) -> None:
        stage_07 = load_stage_07_module()
        for callable_object in (run_training_diagnostics, stage_07.run_diagnostics):
            parameter_names = set(inspect.signature(callable_object).parameters)
            self.assertNotIn("validation_data", parameter_names)
            self.assertNotIn("validation_df", parameter_names)
            self.assertNotIn("test_data", parameter_names)
            self.assertNotIn("test_df", parameter_names)

    def test_complete_training_diagnostic_returns_expected_artifacts(self) -> None:
        rows = 300
        training_data = pd.DataFrame(
            {
                "transaction_id": [f"tx-{index}" for index in range(rows)],
                "event_timestamp": pd.date_range(
                    "2025-01-01", periods=rows, freq="min"
                ),
                "sender_account": [f"s-{index % 10}" for index in range(rows)],
                "receiver_account": [f"r-{index % 13}" for index in range(rows)],
                "device_hash": [f"d-{index % 7}" for index in range(rows)],
                "ip_address": [f"ip-{index % 11}" for index in range(rows)],
                "location": [f"loc-{index % 5}" for index in range(rows)],
                "amount": np.arange(1, rows + 1, dtype=float),
                TARGET_COLUMN: np.tile([0] * 24 + [1], rows // 25),
            }
        )

        result = run_training_diagnostics(training_data)

        self.assertEqual(len(result.entity_repetition_summary), 8)
        self.assertEqual(len(result.new_vs_returning_fraud_rates), 16)
        self.assertIn("recommendation", result.feature_feasibility_decision)
        self.assertFalse(
            result.feature_feasibility_decision[
                "target_used_to_construct_history_features"
            ]
        )

    def test_missing_returning_group_is_serialised_as_null(self) -> None:
        unique_entities = pd.DataFrame(
            {
                "transaction_id": ["a", "b", "c"],
                "event_timestamp": pd.date_range("2025-01-01", periods=3),
                "sender_account": ["s1", "s2", "s3"],
                TARGET_COLUMN: [0, 1, 0],
            }
        )
        _, _, decision = entity_history_diagnostic(
            unique_entities, "sender_account", ("sender_account",)
        )
        self.assertIsNone(decision["relative_fraud_rate_difference"])


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
