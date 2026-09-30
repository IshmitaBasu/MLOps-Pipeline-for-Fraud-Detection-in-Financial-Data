"""Versioned Feast contract helpers for sender-location behavioural features."""

# %% Imports and contract constants
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Int64

from project_io_utils import save_json_atomically, save_parquet_atomically, sha256_file

from predictive_quality_utils import (
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_CODE,
    TIME_COLUMN,
    TRAIN_CODE,
    VALIDATION_CODE,
    stable_split_hash,
)

BEHAVIOUR_FEATURE_VERSION = "v2"
SENDER_COLUMN = "sender_account"
LOCATION_COLUMN = "location"
SENDER_LOCATION_KEY = "sender_location_key"
SENDER_COUNT_FEATURE = "sender_prior_transaction_count"
SENDER_LOCATION_COUNT_FEATURE = "sender_location_prior_transaction_count"
NEW_LOCATION_FEATURE = "is_new_location_for_returning_sender"
STATE_EFFECTIVE_DELAY = pd.Timedelta(microseconds=1)

SENDER_VIEW_NAME = f"sender_history_features_{BEHAVIOUR_FEATURE_VERSION}"
SENDER_LOCATION_VIEW_NAME = f"sender_location_history_features_{BEHAVIOUR_FEATURE_VERSION}"
DERIVED_ADAPTER_NAME = f"derive_new_location_flag_{BEHAVIOUR_FEATURE_VERSION}"
FEATURE_SERVICE_NAME = f"fraud_behaviour_features_{BEHAVIOUR_FEATURE_VERSION}"


@dataclass(frozen=True)
class BehaviourFeatureArtifacts:
    sender_table: pd.DataFrame
    sender_location_table: pd.DataFrame
    development_queries: pd.DataFrame
    split_hash: str
    split_counts: dict[str, int]


# %% Deterministic entity keys and point-in-time snapshots
def compose_sender_location_key(sender: pd.Series, location: pd.Series) -> pd.Series:
    """Create an unambiguous composite entity key shared by training and serving."""
    sender_text = sender.astype("string")
    location_text = location.astype("string")
    if sender_text.isna().any() or location_text.isna().any():
        raise ValueError("Sender and location must be present to create the entity key.")
    return sender_text.str.len().astype("string") + ":" + sender_text + "|" + location_text


def _post_event_snapshots(
    training_events: pd.DataFrame,
    entity_universe: pd.DataFrame,
    entity_columns: list[str],
    output_entity_columns: list[str],
    count_column: str,
) -> pd.DataFrame:
    grouped = (
        training_events.groupby([*entity_columns, TIME_COLUMN], dropna=False, observed=True, sort=False)
        .size()
        .rename("events_at_timestamp")
        .reset_index()
        .sort_values([*entity_columns, TIME_COLUMN], kind="mergesort")
    )
    grouped[count_column] = (
        grouped.groupby(entity_columns, dropna=False, observed=True, sort=False)["events_at_timestamp"]
        .cumsum()
        .astype("int64")
    )
    grouped[TIME_COLUMN] = grouped[TIME_COLUMN] + STATE_EFFECTIVE_DELAY
    baseline = entity_universe[entity_columns].drop_duplicates().copy()
    baseline[TIME_COLUMN] = entity_universe[TIME_COLUMN].min() - STATE_EFFECTIVE_DELAY
    baseline[count_column] = np.int64(0)
    snapshots = pd.concat(
        [
            baseline[[*output_entity_columns, TIME_COLUMN, count_column]],
            grouped[[*output_entity_columns, TIME_COLUMN, count_column]],
        ],
        ignore_index=True,
    ).sort_values([*output_entity_columns, TIME_COLUMN], kind="mergesort")
    for column in output_entity_columns:
        snapshots[column] = snapshots[column].astype(str).astype(object)
    return snapshots.reset_index(drop=True)


def build_behaviour_feature_artifacts(events: pd.DataFrame, partition: np.ndarray) -> BehaviourFeatureArtifacts:
    """Build training-state snapshots and train/validation query entities."""
    required = {ID_COLUMN, TIME_COLUMN, SENDER_COLUMN, LOCATION_COLUMN, TARGET_COLUMN}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing behaviour-source columns: {sorted(missing)}")
    if len(events) != len(partition):
        raise ValueError("Events and partition lengths do not match.")
    if events[ID_COLUMN].duplicated().any():
        raise ValueError("Transaction IDs must be unique.")
    if events[[ID_COLUMN, TIME_COLUMN, SENDER_COLUMN, LOCATION_COLUMN]].isna().any().any():
        raise ValueError("Behaviour entity and timestamp columns cannot be missing.")

    observed_codes = set(np.unique(partition).tolist())
    if observed_codes != {int(TRAIN_CODE), int(VALIDATION_CODE), int(TEST_CODE)}:
        raise ValueError("The frozen train, validation, and test codes are required.")

    working = events[[ID_COLUMN, TIME_COLUMN, SENDER_COLUMN, LOCATION_COLUMN, TARGET_COLUMN]].copy()
    working[TIME_COLUMN] = pd.to_datetime(working[TIME_COLUMN], errors="raise", utc=True)
    working[SENDER_LOCATION_KEY] = compose_sender_location_key(working[SENDER_COLUMN], working[LOCATION_COLUMN])

    training_events = working.loc[partition == TRAIN_CODE].copy()
    development_mask = partition != TEST_CODE
    development_events = working.loc[development_mask].copy()
    development_queries = development_events[
        [ID_COLUMN, TIME_COLUMN, SENDER_COLUMN, SENDER_LOCATION_KEY, TARGET_COLUMN]
    ].copy()
    development_queries["split"] = np.where(partition[development_mask] == TRAIN_CODE, "train", "validation")
    development_queries[SENDER_COLUMN] = development_queries[SENDER_COLUMN].astype(str).astype(object)
    development_queries[SENDER_LOCATION_KEY] = development_queries[SENDER_LOCATION_KEY].astype(str).astype(object)

    sender_table = _post_event_snapshots(
        training_events, development_events, [SENDER_COLUMN], [SENDER_COLUMN], SENDER_COUNT_FEATURE
    )
    sender_location_table = _post_event_snapshots(
        training_events, development_events, [SENDER_LOCATION_KEY], [SENDER_LOCATION_KEY], SENDER_LOCATION_COUNT_FEATURE
    )

    return BehaviourFeatureArtifacts(
        sender_table=sender_table,
        sender_location_table=sender_location_table,
        development_queries=development_queries.reset_index(drop=True),
        split_hash=stable_split_hash(events[ID_COLUMN], partition),
        split_counts={
            "train": int((partition == TRAIN_CODE).sum()),
            "validation": int((partition == VALIDATION_CODE).sum()),
            "test": int((partition == TEST_CODE).sum()),
        },
    )


# %% Feast objects and shared derived-feature adapter
def derive_new_location_flag(inputs: pd.DataFrame) -> pd.DataFrame:
    """Apply identical missing-value defaults and derivation offline and online."""
    sender_count = pd.to_numeric(inputs[SENDER_COUNT_FEATURE], errors="coerce").fillna(0)
    pair_count = pd.to_numeric(inputs[SENDER_LOCATION_COUNT_FEATURE], errors="coerce").fillna(0)
    return pd.DataFrame(
        {NEW_LOCATION_FEATURE: ((sender_count > 0) & (pair_count == 0)).astype(bool)}, index=inputs.index
    )


def build_behaviour_feast_objects(
    sender_table_path: Path, sender_location_table_path: Path
) -> tuple[Entity, Entity, FeatureView, FeatureView, FeatureService]:
    """Create the versioned entities, views, and feature service."""
    sender_entity = Entity(
        name="sender",
        join_keys=[SENDER_COLUMN],
        value_type=ValueType.STRING,
        description="Sender account used only as a behavioural-state lookup key.",
        tags={"feature_version": BEHAVIOUR_FEATURE_VERSION},
    )
    sender_location_entity = Entity(
        name="sender_location",
        join_keys=[SENDER_LOCATION_KEY],
        value_type=ValueType.STRING,
        description="Length-prefixed sender and location composite lookup key.",
        tags={"feature_version": BEHAVIOUR_FEATURE_VERSION},
    )

    sender_source = FileSource(
        name=f"sender_history_source_{BEHAVIOUR_FEATURE_VERSION}",
        path=str(sender_table_path),
        timestamp_field=TIME_COLUMN,
        description="Post-event sender count snapshots from frozen training events.",
    )
    sender_location_source = FileSource(
        name=f"sender_location_history_source_{BEHAVIOUR_FEATURE_VERSION}",
        path=str(sender_location_table_path),
        timestamp_field=TIME_COLUMN,
        description=("Post-event sender-location count snapshots from frozen training events."),
    )

    sender_view = FeatureView(
        name=SENDER_VIEW_NAME,
        entities=[sender_entity],
        source=sender_source,
        ttl=timedelta(days=3650),
        online=True,
        offline=False,
        schema=[Field(name=SENDER_COUNT_FEATURE, dtype=Int64)],
        description="Prior sender activity available at a transaction timestamp.",
        tags={"feature_version": BEHAVIOUR_FEATURE_VERSION, "target_excluded": "true", "point_in_time": "true"},
    )
    sender_location_view = FeatureView(
        name=SENDER_LOCATION_VIEW_NAME,
        entities=[sender_location_entity],
        source=sender_location_source,
        ttl=timedelta(days=3650),
        online=True,
        offline=False,
        schema=[Field(name=SENDER_LOCATION_COUNT_FEATURE, dtype=Int64)],
        description=("Prior activity for the exact sender-location pair at transaction time."),
        tags={"feature_version": BEHAVIOUR_FEATURE_VERSION, "target_excluded": "true", "point_in_time": "true"},
    )
    service = FeatureService(
        name=FEATURE_SERVICE_NAME,
        features=[sender_view, sender_location_view],
        description=(
            "Versioned sender-location behavioural feature contract retained by " "the Stage 08 validation experiment."
        ),
        tags={"feature_version": BEHAVIOUR_FEATURE_VERSION},
    )
    return (sender_entity, sender_location_entity, sender_view, sender_location_view, service)


# %% Artifact persistence and metadata
def feature_contract_metadata(
    artifacts: BehaviourFeatureArtifacts,
    raw_path: Path,
    sender_path: Path,
    sender_location_path: Path,
    expected_split_hash: str,
) -> dict[str, Any]:
    if artifacts.split_hash != expected_split_hash:
        raise ValueError(
            "Behaviour artifacts do not use the frozen Stage 07 split: "
            f"expected {expected_split_hash}, observed {artifacts.split_hash}."
        )
    return {
        "feature_version": BEHAVIOUR_FEATURE_VERSION,
        "feature_service": FEATURE_SERVICE_NAME,
        "feature_views": [SENDER_VIEW_NAME, SENDER_LOCATION_VIEW_NAME],
        "derived_feature_adapter": DERIVED_ADAPTER_NAME,
        "features": {
            SENDER_COUNT_FEATURE: ("strictly earlier frozen-training transactions for the sender"),
            SENDER_LOCATION_COUNT_FEATURE: (
                "strictly earlier frozen-training transactions for the " "sender-location pair"
            ),
            NEW_LOCATION_FEATURE: ("sender count > 0 and sender-location count == 0"),
        },
        "state_snapshot_semantics": (
            "Source snapshots store post-event counts one microsecond after each "
            "source timestamp, so a query at the transaction timestamp receives "
            "strictly earlier state."
        ),
        "missing_count_default": 0,
        "target_registered_as_feature": False,
        "target_used_in_feature_calculation": False,
        "validation_events_update_history": False,
        "test_events_used": False,
        "frozen_split_sha256": artifacts.split_hash,
        "split_counts": artifacts.split_counts,
        "artifacts": {
            "raw_source": {
                "name": raw_path.name,
                "size_bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            },
            "sender_table": {
                "name": sender_path.name,
                "rows": len(artifacts.sender_table),
                "size_bytes": sender_path.stat().st_size,
                "sha256": sha256_file(sender_path),
            },
            "sender_location_table": {
                "name": sender_location_path.name,
                "rows": len(artifacts.sender_location_table),
                "size_bytes": sender_location_path.stat().st_size,
                "sha256": sha256_file(sender_location_path),
            },
        },
    }


# %% Offline/online consistency validation
def normalise_served_features(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented zero defaults to Feast count responses."""
    normalised = table.copy()
    for column in (SENDER_COUNT_FEATURE, SENDER_LOCATION_COUNT_FEATURE):
        normalised[column] = pd.to_numeric(normalised[column], errors="coerce").fillna(0).astype("int64")
    normalised[NEW_LOCATION_FEATURE] = derive_new_location_flag(normalised)[NEW_LOCATION_FEATURE]
    return normalised


def validate_offline_online_consistency(
    store: FeatureStore, service: FeatureService, query_rows: pd.DataFrame
) -> pd.DataFrame:
    """Compare Feast historical and online results at the same source state."""
    if service.name != FEATURE_SERVICE_NAME:
        raise ValueError(f"Expected feature service {FEATURE_SERVICE_NAME}, got {service.name}.")
    required = {SENDER_COLUMN, SENDER_LOCATION_KEY, TIME_COLUMN}
    missing = required.difference(query_rows.columns)
    if missing:
        raise ValueError(f"Missing consistency-query columns: {sorted(missing)}")

    # Feast joins each feature view by its own entity key. Fetching both entity
    # granularities in one historical call can collapse repeated sender/time
    # requests. Perform each distinct lookup once, then expand it back to the
    # original request rows.
    entity_df = query_rows[[SENDER_COLUMN, SENDER_LOCATION_KEY, TIME_COLUMN]].copy()
    entity_df[SENDER_COLUMN] = entity_df[SENDER_COLUMN].astype(str).astype(object)
    entity_df[SENDER_LOCATION_KEY] = entity_df[SENDER_LOCATION_KEY].astype(str).astype(object)
    entity_df[TIME_COLUMN] = pd.to_datetime(entity_df[TIME_COLUMN], errors="raise", utc=True)
    sender_ref = f"{SENDER_VIEW_NAME}:{SENDER_COUNT_FEATURE}"
    pair_ref = f"{SENDER_LOCATION_VIEW_NAME}:{SENDER_LOCATION_COUNT_FEATURE}"
    sender_queries = entity_df[[SENDER_COLUMN, TIME_COLUMN]].drop_duplicates()
    pair_queries = entity_df[[SENDER_LOCATION_KEY, TIME_COLUMN]].drop_duplicates()
    offline_sender = store.get_historical_features(
        entity_df=sender_queries, features=[sender_ref], full_feature_names=False
    ).to_df()
    offline_pair = store.get_historical_features(
        entity_df=pair_queries, features=[pair_ref], full_feature_names=False
    ).to_df()
    offline = entity_df.merge(
        offline_sender[[SENDER_COLUMN, TIME_COLUMN, SENDER_COUNT_FEATURE]],
        on=[SENDER_COLUMN, TIME_COLUMN],
        how="left",
        validate="many_to_one",
    ).merge(
        offline_pair[[SENDER_LOCATION_KEY, TIME_COLUMN, SENDER_LOCATION_COUNT_FEATURE]],
        on=[SENDER_LOCATION_KEY, TIME_COLUMN],
        how="left",
        validate="many_to_one",
    )
    offline = normalise_served_features(offline)

    online_entities = query_rows[[SENDER_COLUMN, SENDER_LOCATION_KEY]].copy()
    online_entities[SENDER_COLUMN] = online_entities[SENDER_COLUMN].astype(str)
    online_entities[SENDER_LOCATION_KEY] = online_entities[SENDER_LOCATION_KEY].astype(str)
    online_sender = store.get_online_features(
        features=[sender_ref],
        entity_rows=online_entities[[SENDER_COLUMN]].to_dict(orient="records"),
        full_feature_names=False,
    ).to_df()
    online_pair = store.get_online_features(
        features=[pair_ref],
        entity_rows=online_entities[[SENDER_LOCATION_KEY]].to_dict(orient="records"),
        full_feature_names=False,
    ).to_df()
    online = online_entities.reset_index(drop=True)
    online[SENDER_COUNT_FEATURE] = online_sender[SENDER_COUNT_FEATURE].to_numpy()
    online[SENDER_LOCATION_COUNT_FEATURE] = online_pair[SENDER_LOCATION_COUNT_FEATURE].to_numpy()
    online = normalise_served_features(online)

    key_columns = [SENDER_COLUMN, SENDER_LOCATION_KEY]
    feature_columns = [SENDER_COUNT_FEATURE, SENDER_LOCATION_COUNT_FEATURE, NEW_LOCATION_FEATURE]
    offline_comparison = (
        offline[key_columns + feature_columns].sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    )
    online_comparison = (
        online[key_columns + feature_columns].sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(offline_comparison, online_comparison, check_dtype=False)
    return offline_comparison


def close_local_online_store(store: FeatureStore) -> None:
    """Release the local SQLite connection so temporary repos cleanly close."""
    provider = store._get_provider()
    online_store = provider.online_store
    connection = getattr(online_store, "_conn", None)
    if connection is not None:
        connection.close()
        online_store._conn = None
