"""Synthetic checks for monitoring arithmetic, evidence boundaries and decisions."""

# %% Imports and fixtures
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.monitoring.batch import (
    MonitoringThresholds,
    build_reference_profile,
    compare_with_reference,
    labelled_performance,
    load_prediction_ledger,
    monitoring_recommendation,
    population_stability_index,
    validate_scored_batch,
)
from fraud_detection.monitoring.reference import history_preparation_receipt, prepare_reference_history
from fraud_detection.modeling.oversampling import MODEL_FEATURES
from fraud_detection.diagnostics.predictive_quality import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    make_stratified_random_partition,
    stable_split_hash,
)


def frame(rows: int = 500, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = pd.DataFrame({name: rng.normal(shift, 1, rows) for name in MODEL_FEATURES})
    for name in ("transaction_type", "merchant_category", "location", "device_used", "payment_channel"):
        data[name] = rng.choice(["a", "b"], rows, p=[0.8 - 0.3 * shift, 0.2 + 0.3 * shift])
    data["transaction_id"] = [f"T{i}" for i in range(rows)]
    data["score"] = np.clip(rng.beta(2 + shift, 3, rows), 0, 1)
    data["is_fraud"] = (np.arange(rows) % 10 == 0).astype(int)
    data["amount"] = np.arange(rows, dtype=float) + 1
    return data


def profile(reference: pd.DataFrame) -> dict:
    return build_reference_profile(
        reference, threshold=0.5, model_run_id="model", state_version="state", thresholds=MonitoringThresholds()
    )


# %% Profile and drift tests
class MonitoringProfileTests(unittest.TestCase):
    def test_identical_population_has_zero_psi(self):
        reference = frame()
        drift = compare_with_reference(reference, profile(reference))
        self.assertLess(float(drift.psi.max()), 1e-12)

    def test_shifted_population_is_detected(self):
        reference, current = frame(), frame(500, 1.0)
        drift = compare_with_reference(current, profile(reference))
        self.assertGreater(float(drift.psi.max()), 0.25)

    def test_population_stability_requires_matching_bins(self):
        with self.assertRaises(ValueError):
            population_stability_index([0.5, 0.5], [1.0])

    def test_missing_required_feature_fails(self):
        broken = frame().drop(columns=[MODEL_FEATURES[-1]])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_scored_batch(broken)

    def test_invalid_reference_is_rejected(self):
        reference = frame()
        reference.loc[0, "score"] = 2.0
        with self.assertRaisesRegex(ValueError, "monitoring reference"):
            profile(reference)


# %% Label, decision and ledger tests
class MonitoringDecisionTests(unittest.TestCase):
    def test_delayed_label_metrics_include_missed_value(self):
        current = frame(500)
        result = labelled_performance(current, 0.5, MonitoringThresholds())
        self.assertEqual(result["status"], "available")
        self.assertIn("missed_fraud_amount", result)
        self.assertIn("fraud_value_recall", result)

    def test_no_labels_are_reported_as_unavailable(self):
        result = labelled_performance(frame().drop(columns="is_fraud"), 0.5, MonitoringThresholds())
        self.assertEqual(result["status"], "unavailable_no_labels")

    def test_critical_drift_never_retrains_automatically(self):
        reference, current = frame(), frame(500, 1.0)
        reference_profile = profile(reference)
        drift = compare_with_reference(current, reference_profile)
        quality = validate_scored_batch(current)
        performance = labelled_performance(current, 0.5, MonitoringThresholds())
        decision = monitoring_recommendation(reference_profile, current, drift, quality, performance)
        self.assertEqual(decision["severity"], "critical")
        self.assertFalse(decision["automatic_retraining_triggered"])
        self.assertTrue(decision["human_approval_required"])

    def test_private_ledger_loader_reads_success_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.sqlite"
            features = {
                name: (
                    "a"
                    if name in {"transaction_type", "merchant_category", "location", "device_used", "payment_channel"}
                    else 1
                )
                for name in MODEL_FEATURES
            }
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE TABLE predictions (transaction_id TEXT, request_json TEXT, response_json TEXT, features_json TEXT)"
                )
                connection.execute(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?)",
                    ("T1", "{}", json.dumps({"score": 0.4}), json.dumps(features)),
                )
                connection.commit()
            finally:
                connection.close()
            loaded = load_prediction_ledger(path)
            self.assertEqual(loaded.loc[0, "transaction_id"], "T1")
            self.assertEqual(loaded.loc[0, "score"], 0.4)


# %% Training-only reference-history integration
class MonitoringReferencePreparationTests(unittest.TestCase):
    def test_reference_history_uses_only_frozen_training_rows(self):
        rows = 40
        raw = frame(rows).rename(columns={TIME_COLUMN: "timestamp"})
        raw[ID_COLUMN] = [f"T{i}" for i in range(rows)]
        raw[TARGET_COLUMN] = [int(i % 5 == 0) for i in range(rows)]
        raw["timestamp"] = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")
        raw["sender_account"] = [f"S{i % 3}" for i in range(rows)]
        partition = make_stratified_random_partition(raw[TARGET_COLUMN])
        expected_hash = stable_split_hash(raw[ID_COLUMN], partition)
        raw.loc[partition != 0, "amount"] = 999_999
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            raw.to_csv(path, index=False)
            prepared, lineage = prepare_reference_history(path, "full", expected_hash=expected_hash, expected_rows=rows)
        self.assertEqual(len(prepared), 28)
        self.assertFalse((prepared.amount == 999_999).any())
        self.assertFalse(lineage["target_used_to_construct_history"])
        receipt = history_preparation_receipt(prepared, lineage)
        self.assertFalse(receipt["model_scored"])
        self.assertFalse(receipt["reference_profile_created"])


if __name__ == "__main__":
    unittest.main()
