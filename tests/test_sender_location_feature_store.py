"""Tests for the versioned Stage 09 sender-location Feast contract."""

# %% Imports
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predictive_quality_utils import TEST_CODE, TRAIN_CODE, VALIDATION_CODE  # noqa: E402
from sender_location_feature_store import (  # noqa: E402
    NEW_LOCATION_FEATURE,
    SENDER_COLUMN,
    SENDER_COUNT_FEATURE,
    SENDER_LOCATION_COUNT_FEATURE,
    SENDER_LOCATION_KEY,
    SENDER_LOCATION_VIEW_NAME,
    SENDER_VIEW_NAME,
    build_behaviour_feast_objects,
    build_behaviour_feature_artifacts,
    close_local_online_store,
    compose_sender_location_key,
    save_parquet_atomically,
    validate_offline_online_consistency,
)


# %% Fixtures
def source_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "event_timestamp": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-01", "2025-01-01"], utc=True
            ),
            "sender_account": ["A", "A", "A", "A", "B", "C"],
            "location": ["X", "X", "Y", "NEW", "Z", "Z"],
            "is_fraud": [0, 1, 0, 1, 0, 1],
        }
    )


def source_partition() -> np.ndarray:
    return np.array([TRAIN_CODE, VALIDATION_CODE, TRAIN_CODE, VALIDATION_CODE, TRAIN_CODE, TEST_CODE], dtype=np.uint8)


def write_config(repo_path: Path) -> None:
    data_dir = repo_path / "feature_repo" / "data"
    data_dir.mkdir(parents=True)
    (repo_path / "feature_store.yaml").write_text(
        "\n".join(
            [
                "project: stage_09_test",
                "registry: feature_repo/data/registry.db",
                "provider: local",
                "online_store:",
                "  type: sqlite",
                "  path: feature_repo/data/online_store.db",
            ]
        ),
        encoding="utf-8",
    )


# %% Feature calculation tests
class BehaviourArtifactTests(unittest.TestCase):
    def test_snapshots_use_training_events_and_exclude_test_queries(self) -> None:
        artifacts = build_behaviour_feature_artifacts(source_events(), source_partition())

        sender_a = artifacts.sender_table.loc[artifacts.sender_table[SENDER_COLUMN] == "A"]
        self.assertEqual(sender_a[SENDER_COUNT_FEATURE].tolist(), [0, 1, 2])
        self.assertNotIn("C", set(artifacts.sender_table[SENDER_COLUMN]))
        self.assertEqual(len(artifacts.development_queries), 5)
        self.assertNotIn("T6", set(artifacts.development_queries["transaction_id"]))
        self.assertNotIn("is_fraud", artifacts.sender_table.columns)
        self.assertNotIn("is_fraud", artifacts.sender_location_table.columns)

    def test_target_values_do_not_change_feature_snapshots(self) -> None:
        events = source_events()
        original = build_behaviour_feature_artifacts(events, source_partition())
        changed_events = events.copy()
        changed_events["is_fraud"] = 1 - changed_events["is_fraud"]
        changed = build_behaviour_feature_artifacts(changed_events, source_partition())
        pd.testing.assert_frame_equal(original.sender_table, changed.sender_table)
        pd.testing.assert_frame_equal(original.sender_location_table, changed.sender_location_table)

    def test_composite_key_is_unambiguous(self) -> None:
        keys = compose_sender_location_key(pd.Series(["AB", "A"]), pd.Series(["C", "BC"]))
        self.assertEqual(keys.tolist(), ["2:AB|C", "1:A|BC"])
        self.assertEqual(len(set(keys)), 2)


# %% Local Feast integration test
class BehaviourFeastRoundTripTests(unittest.TestCase):
    def test_offline_and_online_values_match(self) -> None:
        artifacts = build_behaviour_feature_artifacts(source_events(), source_partition())
        with tempfile.TemporaryDirectory(prefix="stage_09_feast_test_") as directory:
            repo_path = Path(directory)
            write_config(repo_path)
            sender_path = repo_path / "feature_repo" / "data" / "sender_history_v2.parquet"
            pair_path = repo_path / "feature_repo" / "data" / "sender_location_history_v2.parquet"
            save_parquet_atomically(artifacts.sender_table, sender_path)
            save_parquet_atomically(artifacts.sender_location_table, pair_path)

            objects = build_behaviour_feast_objects(sender_path, pair_path)
            store = FeatureStore(repo_path=str(repo_path))
            try:
                store.apply(list(objects))
                start = min(
                    artifacts.sender_table["event_timestamp"].min(),
                    artifacts.sender_location_table["event_timestamp"].min(),
                ) - pd.Timedelta(microseconds=1)
                end = max(
                    artifacts.sender_table["event_timestamp"].max(),
                    artifacts.sender_location_table["event_timestamp"].max(),
                ) + pd.Timedelta(microseconds=1)
                store.materialize(
                    start.to_pydatetime(),
                    end.to_pydatetime(),
                    feature_views=[SENDER_VIEW_NAME, SENDER_LOCATION_VIEW_NAME],
                )

                pair_keys = compose_sender_location_key(pd.Series(["A", "A"]), pd.Series(["Y", "NEW"]))
                queries = pd.DataFrame(
                    {SENDER_COLUMN: ["A", "A"], SENDER_LOCATION_KEY: pair_keys, "event_timestamp": [end, end]}
                )
                result = validate_offline_online_consistency(store, objects[-1], queries)
            finally:
                close_local_online_store(store)

        self.assertEqual(int(result.iloc[0][SENDER_COUNT_FEATURE]), 2)
        new_location_row = result.loc[result[SENDER_LOCATION_KEY].str.endswith("|NEW")].iloc[0]
        known_location_row = result.loc[result[SENDER_LOCATION_KEY].str.endswith("|Y")].iloc[0]
        self.assertEqual(int(known_location_row[SENDER_LOCATION_COUNT_FEATURE]), 1)
        self.assertFalse(bool(known_location_row[NEW_LOCATION_FEATURE]))
        self.assertEqual(int(new_location_row[SENDER_LOCATION_COUNT_FEATURE]), 0)
        self.assertTrue(bool(new_location_row[NEW_LOCATION_FEATURE]))


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
