"""Focused integration test for the local Feast handoff."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource
from feast import ValueType
from feast.types import Float64, Int64, String

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from fraud_feature_store import load_feast_training_table  # noqa: E402


def load_data_pipeline_module():
    module_path = PROJECT_DIR / "02_data_pipeline_preprocessing.py"
    specification = importlib.util.spec_from_file_location(
        "data_pipeline_preprocessing",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FeatureStoreHandoffTest(unittest.TestCase):
    def test_round_trip_preserves_rows_labels_and_features(self) -> None:
        pipeline_module = load_data_pipeline_module()
        timestamps = pd.date_range("2024-01-01", periods=6, freq="h")
        cleaned_table = pd.DataFrame(
            {
                "transaction_id": [f"tx-{index}" for index in range(6)],
                "event_timestamp": timestamps,
                "transaction_type": ["payment", "transfer"] * 3,
                "merchant_category": ["retail", "travel", "retail"] * 2,
                "location": ["DE", "FR", "DE"] * 2,
                "device_used": ["mobile", "web"] * 3,
                "payment_channel": ["card", "bank"] * 3,
                "amount": [10.0, 25.5, 40.0, 75.0, 120.0, 250.0],
                "time_since_last_transaction": [1.0, None, 3.0, 4.0, 5.0, 6.0],
                "spending_deviation_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "velocity_score": [1, 2, 3, 4, 5, 6],
                "geo_anomaly_score": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "is_fraud": [0, 1, 0, 0, 1, 0],
            }
        )

        with tempfile.TemporaryDirectory(prefix="fraud_feast_test_") as temp_dir:
            repo_path = Path(temp_dir)
            data_dir = repo_path / "feature_repo" / "data"
            metadata_dir = repo_path / "feature_repo" / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            (repo_path / "feature_store.yaml").write_text(
                "\n".join(
                    [
                        "project: financial_fraud_detection",
                        "registry: feature_repo/data/registry.db",
                        "provider: local",
                        "online_store: null",
                    ]
                ),
                encoding="utf-8",
            )
            (repo_path / "fraud_feature_definitions.py").write_text(
                "# Test definition snapshot.\n",
                encoding="utf-8",
            )
            raw_file = repo_path / "raw.csv"
            raw_file.write_text("test-source\n", encoding="utf-8")

            feature_path = data_dir / "fraud_features_v1.parquet"
            label_path = data_dir / "fraud_labels_v1.parquet"
            schema_path = metadata_dir / "feature_schema_v1.json"
            metadata_path = metadata_dir / "feature_metadata_v1.json"

            pipeline_module.save_feature_store_handoff(
                cleaned_table,
                raw_file,
                feature_path,
                label_path,
                schema_path,
                metadata_path,
                {
                    "initial_rows": len(cleaned_table),
                    "duplicate_rows_removed": 0,
                    "invalid_amount_rows": 0,
                    "invalid_timestamp_rows": 0,
                },
            )

            transaction = Entity(
                name="transaction",
                join_keys=["transaction_id"],
                value_type=ValueType.STRING,
            )
            source = FileSource(
                name="fraud_transaction_source_v1",
                path=str(feature_path),
                timestamp_field="event_timestamp",
            )
            feature_view = FeatureView(
                name="fraud_transaction_features",
                entities=[transaction],
                source=source,
                ttl=timedelta(days=3650),
                online=False,
                offline=True,
                schema=[
                    Field(name="transaction_type", dtype=String),
                    Field(name="merchant_category", dtype=String),
                    Field(name="location", dtype=String),
                    Field(name="device_used", dtype=String),
                    Field(name="payment_channel", dtype=String),
                    Field(name="amount", dtype=Float64),
                    Field(name="time_since_last_transaction", dtype=Float64),
                    Field(name="spending_deviation_score", dtype=Float64),
                    Field(name="velocity_score", dtype=Int64),
                    Field(name="geo_anomaly_score", dtype=Float64),
                ],
            )
            feature_service = FeatureService(
                name="fraud_model_features_v1",
                features=[feature_view],
            )

            FeatureStore(repo_path=str(repo_path)).apply([transaction, feature_view, feature_service])
            retrieved, dataset_info, lineage = load_feast_training_table(repo_path)

            self.assertEqual(len(retrieved), len(cleaned_table))
            self.assertEqual(int(retrieved["is_fraud"].sum()), 2)
            self.assertEqual(set(retrieved["transaction_id"]), set(cleaned_table["transaction_id"]))
            self.assertEqual(lineage["data_interface"], "feast_historical_retrieval")
            self.assertEqual(lineage["feature_service"], "fraud_model_features_v1")
            self.assertEqual(dataset_info["dataset_name"], "fraud_features_v1.parquet")


if __name__ == "__main__":
    unittest.main()
