"""Tests for the controlled, row-level feature calculations."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def load_feature_engineering_module():
    module_path = PROJECT_DIR / "05_feature_engineering_with_mlflow.py"
    specification = importlib.util.spec_from_file_location(
        "feature_engineering_with_mlflow",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class FeatureEngineeringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_feature_engineering_module()

    def test_candidate_features_use_known_row_values_without_changing_rows(self) -> None:
        table = pd.DataFrame(
            {
                "transaction_id": ["tx-1", "tx-2", "tx-3"],
                "event_timestamp": pd.to_datetime(
                    [
                        "2024-01-01T00:00:00",  # Monday
                        "2024-06-08T13:30:00",  # Saturday
                        "2024-12-29T23:59:00",  # Sunday
                    ]
                ),
                "amount": [0.0, np.e - 1.0, 99.0],
                "time_since_last_transaction": [1.0, np.nan, 3.0],
            }
        )
        original_ids = table["transaction_id"].copy()

        result = self.module.add_candidate_features(table.copy())
        self.module.validate_candidate_features(result, expected_rows=3)

        pd.testing.assert_series_equal(result["transaction_id"], original_ids)
        np.testing.assert_allclose(result["amount_log1p"], [0.0, 1.0, np.log(100.0)])
        self.assertEqual(result["transaction_hour"].tolist(), [0, 13, 23])
        self.assertEqual(result["transaction_day_of_week"].tolist(), [0, 5, 6])
        self.assertEqual(result["transaction_month"].tolist(), [1, 6, 12])
        self.assertEqual(result["transaction_is_weekend"].tolist(), [0, 1, 1])
        self.assertEqual(result["time_since_last_transaction_missing"].tolist(), [0, 1, 0])

    def test_negative_amount_is_rejected(self) -> None:
        table = pd.DataFrame(
            {
                "event_timestamp": pd.to_datetime(["2024-01-01T00:00:00"]),
                "amount": [-0.01],
                "time_since_last_transaction": [1.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "negative amount"):
            self.module.add_candidate_features(table)

    def test_validation_rejects_a_changed_row_count(self) -> None:
        table = pd.DataFrame(
            {
                "event_timestamp": pd.to_datetime(["2024-01-01T00:00:00"]),
                "amount": [10.0],
                "time_since_last_transaction": [1.0],
            }
        )
        result = self.module.add_candidate_features(table)

        with self.assertRaisesRegex(ValueError, "changed the row count"):
            self.module.validate_candidate_features(result, expected_rows=2)


if __name__ == "__main__":
    unittest.main()
