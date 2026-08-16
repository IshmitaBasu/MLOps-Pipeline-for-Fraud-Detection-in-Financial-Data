"""Integration tests for incremental SQLite-backed data ingestion."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from automation.run_data_pipeline import (  # noqa: E402
    DEFAULT_DATABASE_NAME,
    database_status,
    run_once,
)


def valid_batch(transaction_prefix: str = "tx") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"{transaction_prefix}-1", f"{transaction_prefix}-2", f"{transaction_prefix}-3"],
            "timestamp": ["2024-01-01T10:00:00", "2024-01-01T11:00:00", "2024-01-01T12:00:00"],
            "sender_account": ["s1", "s2", "s3"],
            "receiver_account": ["r1", "r2", "r3"],
            "amount": [10.0, 20.0, 30.0],
            "transaction_type": ["payment", "transfer", "payment"],
            "merchant_category": ["retail", "travel", "retail"],
            "location": ["DE", "FR", "DE"],
            "device_used": ["mobile", "web", "mobile"],
            "is_fraud": [False, True, False],
            "fraud_type": [None, "card", None],
            "time_since_last_transaction": [1.0, None, 3.0],
            "spending_deviation_score": [0.1, 0.2, 0.3],
            "velocity_score": [1, 2, 3],
            "geo_anomaly_score": [0.0, 0.5, 0.1],
            "payment_channel": ["card", "bank", "card"],
            "ip_address": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            "device_hash": ["d1", "d2", "d3"],
        }
    )


class AutomatedDataPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="fraud_data_automation_test_")
        self.root = Path(self.temporary_directory.name)
        self.inbox = self.root / "inbox"
        self.inbox.mkdir(parents=True)
        self.database = self.root / DEFAULT_DATABASE_NAME

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_batch_appends_raw_and_clean_rows_then_removes_source(self) -> None:
        input_path = self.inbox / "august_batch.csv"
        valid_batch().to_csv(input_path, index=False)

        result = run_once(self.root, stability_seconds=0)[0]

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["ml_pipeline_triggered"])
        self.assertEqual(result["raw_row_count"], 3)
        self.assertEqual(result["clean_row_count"], 3)
        self.assertFalse(input_path.exists())

        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM raw_transactions").fetchone()[0], 3)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM clean_transactions").fetchone()[0], 3)
            stored_raw = connection.execute(
                "SELECT timestamp, fraud_type FROM raw_transactions WHERE transaction_id = 'tx-2'"
            ).fetchone()
            self.assertEqual(stored_raw, ("2024-01-01T11:00:00", "card"))
            ml_triggered = connection.execute(
                "SELECT ml_pipeline_triggered FROM ingestion_batches WHERE status = 'success'"
            ).fetchone()[0]
            self.assertEqual(ml_triggered, 0)

    def test_second_unique_batch_appends_only_its_rows(self) -> None:
        first_path = self.inbox / "first.csv"
        valid_batch("first").to_csv(first_path, index=False)
        run_once(self.root, stability_seconds=0)

        second_path = self.inbox / "second.csv"
        valid_batch("second").to_csv(second_path, index=False)
        second_result = run_once(self.root, stability_seconds=0)[0]

        self.assertEqual(second_result["status"], "success")
        status = database_status(self.root)
        self.assertEqual(status["batch_count"], 2)
        self.assertEqual(status["raw_row_count"], 6)
        self.assertEqual(status["clean_row_count"], 6)

    def test_duplicate_file_is_logged_but_not_appended_twice(self) -> None:
        first_path = self.inbox / "first.csv"
        valid_batch().to_csv(first_path, index=False)
        first_result = run_once(self.root, stability_seconds=0)[0]

        duplicate_path = self.inbox / "same_content.csv"
        valid_batch().to_csv(duplicate_path, index=False)
        duplicate_result = run_once(self.root, stability_seconds=0)[0]

        self.assertEqual(first_result["status"], "success")
        self.assertEqual(duplicate_result["status"], "skipped_duplicate")
        self.assertEqual(duplicate_result["duplicate_of_batch_id"], first_result["batch_id"])
        self.assertFalse(duplicate_path.exists())
        status = database_status(self.root)
        self.assertEqual(status["batch_count"], 2)
        self.assertEqual(status["raw_row_count"], 3)
        self.assertEqual(status["clean_row_count"], 3)

    def test_invalid_batch_is_logged_and_retained_with_failed_suffix(self) -> None:
        invalid_path = self.inbox / "invalid.csv"
        pd.DataFrame({"transaction_id": ["tx-1"], "amount": [10.0]}).to_csv(invalid_path, index=False)

        result = run_once(self.root, stability_seconds=0)[0]

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["ml_pipeline_triggered"])
        self.assertFalse(invalid_path.exists())
        self.assertTrue(Path(result["failed_file"]).is_file())
        self.assertTrue(Path(result["failed_file"]).name.endswith(".csv.failed"))
        status = database_status(self.root)
        self.assertEqual(status["batch_count"], 1)
        self.assertEqual(status["raw_row_count"], 0)
        self.assertEqual(status["clean_row_count"], 0)

    def test_existing_transaction_id_rejects_entire_new_batch(self) -> None:
        first_path = self.inbox / "first.csv"
        valid_batch().to_csv(first_path, index=False)
        run_once(self.root, stability_seconds=0)

        overlap_path = self.inbox / "overlap.csv"
        overlap = valid_batch("new")
        overlap.loc[0, "transaction_id"] = "tx-1"
        overlap.to_csv(overlap_path, index=False)
        result = run_once(self.root, stability_seconds=0)[0]

        self.assertEqual(result["status"], "failed")
        status = database_status(self.root)
        self.assertEqual(status["raw_row_count"], 3)
        self.assertEqual(status["clean_row_count"], 3)

    def test_recent_file_is_left_in_inbox(self) -> None:
        input_path = self.inbox / "still_copying.csv"
        valid_batch().to_csv(input_path, index=False)
        os.utime(input_path, None)

        results = run_once(self.root, stability_seconds=60)

        self.assertEqual(results, [])
        self.assertTrue(input_path.is_file())

    def test_automation_source_has_no_ml_pipeline_dependency(self) -> None:
        source = (PROJECT_DIR / "automation" / "run_data_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("04_baseline_model_comparison_with_mlflow", source)
        self.assertNotIn("fraud_modeling_utils", source)
        self.assertNotIn("mlflow", source.lower())


if __name__ == "__main__":
    unittest.main()
