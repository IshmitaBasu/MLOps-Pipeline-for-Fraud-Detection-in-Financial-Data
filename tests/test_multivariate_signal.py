"""Tests for the Stage 19 training-only multivariate diagnostic."""

# %% Imports and fixture
import unittest

import numpy as np
import pandas as pd

from fraud_detection.common.modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from fraud_detection.diagnostics.multivariate_signal import (
    INTERACTION_FEATURES,
    add_multivariate_features,
    attach_development_only_history,
    audit_slice_metrics,
    baseline_results,
    frozen_feature_sets,
    internal_development_audit_split,
    interpretation_summary,
    validate_feature_sets,
)
from fraud_detection.diagnostics.predictive_quality import ID_COLUMN, TARGET_COLUMN, TIME_COLUMN
from fraud_detection.features.sender_location_history import HISTORY_FEATURES


def fixture(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    labels = np.tile([0] * 19 + [1], rows // 20)
    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"T{i}" for i in range(rows)],
            TIME_COLUMN: pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC"),
            TARGET_COLUMN: labels,
            "amount": rng.uniform(1, 1000, rows),
            "time_since_last_transaction": rng.normal(10, 2, rows),
            "spending_deviation_score": rng.normal(size=rows),
            "velocity_score": rng.normal(size=rows),
            "geo_anomaly_score": rng.normal(size=rows),
            "sender_prior_transaction_count": np.arange(rows) % 8,
            "sender_location_prior_transaction_count": np.arange(rows) % 4,
            "is_new_location_for_returning_sender": (np.arange(rows) % 3 == 0).astype("int8"),
        }
    )
    frame.loc[frame.index % 7 == 0, "time_since_last_transaction"] = np.nan
    for index, column in enumerate(CATEGORICAL_FEATURES):
        frame[column] = np.where(np.arange(rows) % (index + 2) == 0, "a", "b")
    return frame


# %% Isolation and deterministic construction
class FeatureSafetyTests(unittest.TestCase):
    def test_interactions_are_label_independent_and_feature_sets_are_safe(self):
        frame = fixture()
        first = add_multivariate_features(frame)
        second = add_multivariate_features(frame.assign(is_fraud=1 - frame.is_fraud))
        pd.testing.assert_frame_equal(first[INTERACTION_FEATURES], second[INTERACTION_FEATURES])
        validate_feature_sets(frozen_feature_sets())
        for feature_set in frozen_feature_sets():
            self.assertNotIn(TARGET_COLUMN, feature_set.columns)
            self.assertNotIn("fraud_type", feature_set.columns)

    def test_internal_split_is_reproducible_disjoint_and_training_only(self):
        frame = fixture()
        development, audit, receipt = internal_development_audit_split(frame)
        development_again, audit_again, _ = internal_development_audit_split(frame)
        self.assertEqual(development[ID_COLUMN].tolist(), development_again[ID_COLUMN].tolist())
        self.assertEqual(audit[ID_COLUMN].tolist(), audit_again[ID_COLUMN].tolist())
        self.assertFalse(set(development[ID_COLUMN]).intersection(audit[ID_COLUMN]))
        self.assertEqual(len(development) + len(audit), len(frame))
        self.assertEqual(receipt["validation_rows_used"], 0)
        self.assertEqual(receipt["test_rows_used"], 0)

    def test_history_uses_development_events_but_not_audit_events(self):
        frame = fixture(40)
        frame["sender_account"] = "same_sender"
        frame["location"] = "same_location"
        development = frame.iloc[[0, 2]].reset_index(drop=True)
        audit = frame.iloc[[1, 3]].reset_index(drop=True)
        development_history, audit_history = attach_development_only_history(development, audit)
        self.assertEqual(development_history["sender_prior_transaction_count"].tolist(), [0, 1])
        self.assertEqual(audit_history["sender_prior_transaction_count"].tolist(), [1, 2])
        changed_development, changed_audit = attach_development_only_history(
            development, audit.assign(is_fraud=1 - audit.is_fraud)
        )
        pd.testing.assert_frame_equal(development_history[HISTORY_FEATURES], changed_development[HISTORY_FEATURES])
        pd.testing.assert_frame_equal(audit_history[HISTORY_FEATURES], changed_audit[HISTORY_FEATURES])


# %% Aggregate result and interpretation checks
class AggregateDiagnosticTests(unittest.TestCase):
    def test_baselines_slices_and_interpretation_are_complete(self):
        frame = add_multivariate_features(fixture())
        _, audit, _ = internal_development_audit_split(frame)
        baseline = pd.DataFrame(baseline_results(audit))
        self.assertEqual(set(baseline.candidate), {"prevalence_baseline", "time_observed_marginal_pattern"})
        self.assertAlmostEqual(
            baseline.loc[baseline.candidate == "prevalence_baseline", "average_precision"].iloc[0],
            audit[TARGET_COLUMN].mean(),
        )
        scores = np.linspace(0, 1, len(audit))
        slices = audit_slice_metrics(audit, scores, "all_supplied_main_fields")
        self.assertEqual(set(slices["slice"]), {"time_q1", "time_q2", "time_q3", "time_q4", "hash_a", "hash_b"})

        candidates = {
            "prevalence_baseline": 0.05,
            "time_observed_marginal_pattern": 0.055,
            "core_fields": 0.060,
            "all_supplied_main_fields": 0.065,
            "all_supplied_plus_interactions": 0.067,
            "all_supplied_plus_interactions_shuffled_label_control": 0.051,
        }
        results = pd.DataFrame([{"candidate": name, "average_precision": value} for name, value in candidates.items()])
        main_slices = slices.assign(candidate="all_supplied_main_fields", average_precision=0.060)
        interaction_slices = slices.assign(candidate="all_supplied_plus_interactions", average_precision=0.061)
        summary = interpretation_summary(results, pd.concat([main_slices, interaction_slices], ignore_index=True))
        self.assertAlmostEqual(summary["interaction_ap_lift_over_all_main_fields"], 0.002)
        self.assertEqual(summary["interaction_positive_stability_slices"], 6)
        self.assertFalse(summary["test_analysed"])


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
