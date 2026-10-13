"""Synthetic diagnostics, history timing and strict training extraction tests."""

# %% Imports and representative training fixtures
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from fraud_detection.common.modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from fraud_detection.common.training_data import load_training_data
from fraud_detection.diagnostics.predictive_quality import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    make_stratified_random_partition,
    stable_split_hash,
)
from fraud_detection.diagnostics.predictive_signal import (
    MISSING,
    add_training_history,
    analyse_training,
    cramers_v,
    extract_training_rows,
    group_rates,
    quantile_groups,
    save_signal_plots,
    wilson_interval,
)


def fixture(rows=40):
    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"T{i}" for i in range(rows)],
            TIME_COLUMN: pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC"),
            "sender_account": ["sender"] * rows,
            TARGET_COLUMN: [int(i % 5 == 0) for i in range(rows)],
        }
    )
    for column in CATEGORICAL_FEATURES:
        frame[column] = ["a", "b"] * (rows // 2)
    for column in NUMERIC_FEATURES:
        frame[column] = np.arange(rows, dtype=float)
    frame.loc[frame[TARGET_COLUMN] == 0, "time_since_last_transaction"] = np.nan
    frame["fraud_type"] = frame[TARGET_COLUMN].map({0: None, 1: "known_after_outcome"})
    return frame


# %% Known rate, support and effect-size arithmetic
class AssociationTests(unittest.TestCase):
    def test_wilson_endpoints_and_zero_fraud_support(self):
        low, high = wilson_interval([0, 50, 100], [100, 100, 100])
        self.assertEqual(low[0], 0)
        self.assertEqual(high[2], 1)
        self.assertAlmostEqual(low[1], 0.40383153, places=7)
        table = group_rates(pd.Series([0, 0, 1, 1]), pd.Series(["a", "a", "b", "b"]), "category")
        self.assertAlmostEqual(cramers_v(table), 1)
        self.assertEqual(table.loc[table.group == "a", "fraud_rows"].iloc[0], 0)
        self.assertTrue(table.small_group.all())
        with self.assertRaises(ValueError):
            wilson_interval([2], [1])

    def test_equal_rates_have_zero_association_and_alignment_is_required(self):
        table = group_rates(pd.Series([0, 1, 0, 1]), pd.Series(["a", "a", "b", "b"]), "category")
        self.assertAlmostEqual(cramers_v(table), 0)
        self.assertTrue(np.allclose(table.fraud_lift, 1))
        with self.assertRaises(ValueError):
            group_rates(pd.Series([0, 1]), pd.Series(["a", "b"], index=[1, 0]), "category")

    def test_quantile_bins_handle_constant_all_missing_and_infinity(self):
        groups, edges = quantile_groups(pd.Series([3.0, 3.0, np.nan, np.inf]))
        self.assertEqual(edges, [3])
        self.assertEqual(groups.tolist(), ["constant", "constant", MISSING, MISSING])
        groups, edges = quantile_groups(pd.Series([np.nan, np.nan]))
        self.assertEqual(edges, [])
        self.assertTrue((groups == MISSING).all())
        groups, _ = quantile_groups(pd.Series([0, 0, 1, 1]))
        self.assertEqual(groups.nunique(), 2)


# %% Labels cannot influence history; held-out rows cannot enter extraction
class IsolationTests(unittest.TestCase):
    def test_history_excludes_equal_time_and_is_label_independent(self):
        frame = fixture(10)
        frame.loc[1, TIME_COLUMN] = frame.loc[0, TIME_COLUMN]
        frame["location"] = "same"
        result = add_training_history(frame)
        self.assertEqual(result.sender_prior_transaction_count.iloc[:3].tolist(), [0, 0, 2])
        changed = add_training_history(frame.assign(is_fraud=1 - frame.is_fraud))
        np.testing.assert_array_equal(
            result.sender_location_prior_transaction_count, changed.sender_location_prior_transaction_count
        )
        mask = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype="uint8")
        retained = extract_training_rows(frame, mask)
        self.assertEqual(retained[ID_COLUMN].tolist(), ["T0", "T3", "T6", "T9"])

    def test_chunk_loader_retains_only_frozen_training_features_and_rejects_hash_change(self):
        frame = fixture(40)
        partition = make_stratified_random_partition(frame[TARGET_COLUMN])
        expected = stable_split_hash(frame[ID_COLUMN], partition)
        frame.loc[partition != 0, "amount"] = 999_999
        with TemporaryDirectory() as temporary:
            csv = Path(temporary) / "fixture.csv"
            frame.rename(columns={TIME_COLUMN: "timestamp"}).to_csv(csv, index=False)
            training, info = load_training_data(csv, "full", expected_hash=expected, expected_rows=40)
            self.assertEqual(len(training), 28)
            self.assertFalse((training.amount == 999_999).any())
            self.assertEqual(set(training[ID_COLUMN]), set(frame.loc[partition == 0, ID_COLUMN]))
            self.assertFalse(info["nontraining_feature_analysis"])
            with self.assertRaises(ValueError):
                load_training_data(csv, "full", expected_hash="wrong", expected_rows=40)


# %% Complete aggregate output and no inferred generation rule
class WorkflowTests(unittest.TestCase):
    def test_diagnostics_and_all_figures_work_with_missingness_and_constant_features(self):
        frame = fixture(40)
        frame["velocity_score"] = 1.0
        training = add_training_history(frame)
        summary, tables, definitions = analyse_training(training)
        self.assertFalse(summary["validation_analysed"])
        self.assertFalse(summary["test_analysed"])
        self.assertFalse(summary["model_trained"])
        self.assertNotIn("fraud_type", definitions)
        self.assertEqual(sum("_x_" in name for name in definitions), 2)
        row = (
            tables["group_rates"]
            .loc[
                (tables["group_rates"].feature == "time_since_last_transaction_missing")
                & (tables["group_rates"].group == "missing")
            ]
            .iloc[0]
        )
        self.assertEqual(row.fraud_rows, 0)
        self.assertEqual(row.rows, 32)
        self.assertTrue(row.ci_high > 0)
        self.assertEqual(len(tables["stability_group_rates"].slice.unique()), 6)
        with TemporaryDirectory() as temporary:
            figures = save_signal_plots(training, tables, Path(temporary))
            self.assertEqual(len(figures), 4)
            self.assertTrue(all(path.stat().st_size > 1_000 for path in figures))

    def test_invalid_target_rejected(self):
        training = add_training_history(fixture(10))
        training[TARGET_COLUMN] = 0
        with self.assertRaises(ValueError):
            analyse_training(training)


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
