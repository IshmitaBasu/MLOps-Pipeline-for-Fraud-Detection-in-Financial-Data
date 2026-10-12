"""Synthetic serving checks; no canonical transaction data or test evaluation."""

# %% Fixtures
from __future__ import annotations

import json
from contextlib import closing
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from feast import FeatureStore
from feast.repo_config import RepoConfig

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from research_prediction_api import TransactionRequest, create_app
from research_serving_utils import (
    FINAL_RUN_ID,
    PROJECT,
    FeastFrozenHistory,
    ResearchPredictor,
    ServingError,
    build_state_index,
)
from oversampling_experiment_utils import MODEL_FEATURES
from sender_location_feature_store import (
    build_behaviour_feature_artifacts,
    build_behaviour_feast_objects,
    close_local_online_store,
)
from predictive_quality_utils import TRAIN_CODE, VALIDATION_CODE, TEST_CODE
from final_test_evaluation_utils import FROZEN_THRESHOLD, add_final_history_features


def payload() -> dict:
    return {
        "transaction_id": "demo-1",
        "event_timestamp": "2026-01-01T00:00:00Z",
        "sender_account": "A",
        "transaction_type": "transfer",
        "merchant_category": "other",
        "location": "X",
        "device_used": "mobile",
        "payment_channel": "online",
        "amount": 20.0,
        "time_since_last_transaction": None,
        "spending_deviation_score": 0.1,
        "velocity_score": 2,
        "geo_anomaly_score": 0.1,
    }


class Model:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.last_input = None
        self.calls = 0

    def predict_proba(self, frame):
        self.last_input = frame.copy()
        self.calls += 1
        return np.array([[1 - FROZEN_THRESHOLD, FROZEN_THRESHOLD]])


class History:
    def __init__(self):
        self.values = (5, 2)
        self.healthy = True

    def health(self):
        if not self.healthy:
            raise ServingError("Store unavailable.")

    def counts(self, sender, location):
        return self.values

    def close(self):
        pass


# %% Request, feature, response and ledger safeguards
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.model, self.history = Model(), History()
        manifest = {
            "model_run_id": FINAL_RUN_ID,
            "state_cutoff": "2025-01-01T00:00:00.000001Z",
            "state_version": "fixture",
        }
        self.service = ResearchPredictor(self.model, self.history, manifest, Path(self.temporary.name) / "log.sqlite")
        self.client = TestClient(create_app(predictor=self.service))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_success_preserves_order_null_and_frozen_threshold(self):
        result = self.client.post("/predict", json=payload())
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["decision"], "review")
        self.assertEqual(result.json()["threshold"], FROZEN_THRESHOLD)
        self.assertEqual(list(self.model.last_input.columns), MODEL_FEATURES)
        self.assertTrue(pd.isna(self.model.last_input.iloc[0]["time_since_last_transaction"]))
        self.assertEqual(self.model.last_input.iloc[0]["sender_prior_transaction_count"], 5)
        self.assertNotIn("sender_account", self.model.last_input.columns)
        self.assertFalse(result.json()["production_approved"])
        with closing(sqlite3.connect(self.service.log_path)) as connection:
            row = connection.execute("SELECT features_json FROM predictions").fetchone()
        self.assertEqual(json.loads(row[0])["sender_location_prior_transaction_count"], 2)

    def test_direct_and_api_scores_match(self):
        result = self.client.post("/predict", json=payload()).json()
        direct = self.model.predict_proba(self.model.last_input)[0, 1]
        self.assertEqual(result["score"], direct)

    def test_cold_start_and_new_location(self):
        for counts, expected in [((0, 0), 0), ((5, 0), 1)]:
            self.history.values = counts
            request = payload()
            request["transaction_id"] = str(counts)
            response = self.client.post("/predict", json=request)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(int(self.model.last_input.iloc[0]["is_new_location_for_returning_sender"]), expected)

    def test_duplicate_does_not_rescore_and_conflict_is_rejected(self):
        first = self.client.post("/predict", json=payload())
        second = self.client.post("/predict", json=payload())
        self.assertEqual(first.json(), second.json())
        self.assertEqual(self.model.calls, 1)
        changed = payload()
        changed["amount"] = 99
        self.assertEqual(self.client.post("/predict", json=changed).status_code, 409)

    def test_historical_request_rejected_without_model_call(self):
        request = payload()
        request["event_timestamp"] = "2025-01-01T00:00:00Z"
        self.assertEqual(self.client.post("/predict", json=request).status_code, 422)
        self.assertEqual(self.model.calls, 0)

    def test_outage_blocks_ready_predictions_and_cached_response(self):
        self.client.post("/predict", json=payload())
        self.history.healthy = False
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        self.assertEqual(self.client.get("/health/ready").status_code, 503)
        self.assertEqual(self.client.post("/predict", json=payload()).status_code, 503)

    def test_invalid_counts_fail_closed(self):
        for counts in [(-1, 0), (2, 3), (True, 0), (2.5, 0)]:
            self.history.values = counts
            self.assertEqual(self.client.post("/predict", json=payload()).status_code, 503)
        self.assertEqual(self.model.calls, 0)

    def test_labels_and_history_injection_rejected(self):
        for field in ["is_fraud", "fraud_type", "sender_prior_transaction_count"]:
            request = payload()
            request[field] = 1
            self.assertEqual(self.client.post("/predict", json=request).status_code, 422)

    def test_bad_types_missing_fields_and_timestamp_rejected(self):
        invalid = [
            ("amount", True),
            ("amount", -1),
            ("velocity_score", 1.5),
            ("location", None),
            ("sender_account", " A"),
            ("event_timestamp", "2026-01-01T00:00:00"),
            ("event_timestamp", "2026-01-01T00:00:00.000001Z"),
        ]
        for key, value in invalid:
            request = payload()
            request[key] = value
            self.assertEqual(self.client.post("/predict", json=request).status_code, 422)
        request = payload()
        del request["velocity_score"]
        self.assertEqual(self.client.post("/predict", json=request).status_code, 422)

    def test_timezone_normalised(self):
        request = payload()
        request["event_timestamp"] = "2026-01-01T01:00:00+01:00"
        self.assertEqual(TransactionRequest(**request).event_timestamp, "2026-01-01T00:00:00+00:00")


# %% Snapshot population, equal-time and future-history parity
class StateIndexTests(unittest.TestCase):
    def test_future_counts_match_training_helper_and_exclude_query_events(self):
        events = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
                "event_timestamp": pd.to_datetime(
                    ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"], utc=True
                ),
                "sender_account": ["A"] * 5,
                "location": ["X", "X", "Y", "X", "X"],
                "is_fraud": [0, 1, 0, 1, 0],
            }
        )
        partition = np.array([TRAIN_CODE, TRAIN_CODE, TRAIN_CODE, VALIDATION_CODE, TEST_CODE])
        artifacts = build_behaviour_feature_artifacts(events, partition)
        # At the equal-time events neither training row may see its sibling.
        direct = add_final_history_features(events, partition == TRAIN_CODE)
        self.assertEqual(direct.iloc[0]["sender_prior_transaction_count"], 0)
        self.assertEqual(direct.iloc[1]["sender_prior_transaction_count"], 0)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = {"sender": directory / "sender.parquet", "pair": directory / "pair.parquet"}
            artifacts.sender_table.to_parquet(paths["sender"], index=False)
            artifacts.sender_location_table.to_parquet(paths["pair"], index=False)
            index = directory / "index.sqlite"
            info = build_state_index(paths, index)
            with closing(sqlite3.connect(index)) as connection:
                sender = connection.execute("SELECT value FROM sender WHERE entity='A'").fetchone()[0]
                pair = connection.execute("SELECT value FROM pair WHERE entity='1:A|X'").fetchone()[0]
            self.assertEqual((sender, pair), (3, 2))
            self.assertEqual(sender, direct.iloc[4]["sender_prior_transaction_count"])
            self.assertEqual(pair, direct.iloc[4]["sender_location_prior_transaction_count"])
            self.assertLess(pd.Timestamp(info["state_cutoff"]), events.iloc[3]["event_timestamp"])
            with self.assertRaises(ValueError):
                build_state_index(paths, index)

    def test_real_feast_online_parity_cold_start_and_missing_known_entity(self):
        events = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3", "T4"],
                "event_timestamp": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"], utc=True),
                "sender_account": ["A", "A", "A", "B"],
                "location": ["X", "X", "Y", "X"],
                "is_fraud": [0, 1, 0, 1],
            }
        )
        partition = np.array([TRAIN_CODE, TRAIN_CODE, VALIDATION_CODE, TEST_CODE])
        artifacts = build_behaviour_feature_artifacts(events, partition)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            paths = {"sender": bundle / "sender.parquet", "pair": bundle / "pair.parquet"}
            artifacts.sender_table.to_parquet(paths["sender"], index=False)
            artifacts.sender_location_table.to_parquet(paths["pair"], index=False)
            manifest = build_state_index(paths, bundle / "state_index.sqlite")
            objects = build_behaviour_feast_objects(paths["sender"], paths["pair"])
            config = RepoConfig(
                project=PROJECT,
                provider="local",
                registry=str(bundle / "registry.db"),
                online_store={"type": "sqlite", "path": str(bundle / "online_store.sqlite")},
            )
            store = FeatureStore(config=config)
            history = None
            try:
                store.apply(list(objects))
                start = artifacts.sender_table["event_timestamp"].min() - pd.Timedelta(microseconds=1)
                end = pd.Timestamp(manifest["state_cutoff"]) + pd.Timedelta(microseconds=1)
                store.materialize(start.to_pydatetime(), end.to_pydatetime())
                # Historical counts at a tied first event must both be zero.
                historical = store.get_historical_features(
                    entity_df=pd.DataFrame(
                        {"sender_account": ["A"], "event_timestamp": [events.iloc[0]["event_timestamp"]]}
                    ),
                    features=["sender_history_features_v2:sender_prior_transaction_count"],
                ).to_df()
                self.assertEqual(int(historical.iloc[0]["sender_prior_transaction_count"]), 0)
                history = FeastFrozenHistory(bundle, manifest)
                history.health()
                self.assertEqual(history.counts("A", "X"), (2, 2))
                self.assertEqual(history.counts("A", "NEW"), (2, 0))
                self.assertEqual(history.counts("NEVER-SEEN", "X"), (0, 0))
                # Remove only a synthetic test-store row, not canonical Feast data.
                with closing(sqlite3.connect(bundle / "online_store.sqlite")) as connection:
                    connection.execute(f'DELETE FROM "{PROJECT}_sender_history_features_v2"')
                    connection.commit()
                with self.assertRaises(ServingError):
                    history.health()
                with self.assertRaises(ServingError):
                    history.counts("A", "X")
            finally:
                if history is not None:
                    history.close()
                close_local_online_store(store)


if __name__ == "__main__":
    unittest.main()
