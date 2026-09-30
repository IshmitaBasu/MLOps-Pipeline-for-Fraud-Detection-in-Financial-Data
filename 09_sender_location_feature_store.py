"""Stage 09: build and verify the versioned sender-location Feast contract."""

# %% Imports and paths
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from feast import FeatureStore

from fraud_modeling_utils import PROJECT_DIR, configure_mlflow
from predictive_quality_utils import TARGET_COLUMN, TIME_COLUMN, make_stratified_random_partition
from sender_location_feature_store import (
    BEHAVIOUR_FEATURE_VERSION,
    FEATURE_SERVICE_NAME,
    ID_COLUMN,
    LOCATION_COLUMN,
    SENDER_COLUMN,
    SENDER_LOCATION_KEY,
    SENDER_LOCATION_VIEW_NAME,
    SENDER_VIEW_NAME,
    BehaviourFeatureArtifacts,
    build_behaviour_feast_objects,
    build_behaviour_feature_artifacts,
    close_local_online_store,
    feature_contract_metadata,
    save_json_atomically,
    save_parquet_atomically,
    validate_offline_online_consistency,
)

RAW_DATA_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
DATA_DIR = PROJECT_DIR / "feature_repo" / "data"
METADATA_DIR = PROJECT_DIR / "feature_repo" / "metadata"
SENDER_TABLE_PATH = DATA_DIR / "sender_history_features_v2.parquet"
SENDER_LOCATION_TABLE_PATH = DATA_DIR / "sender_location_history_features_v2.parquet"
METADATA_PATH = METADATA_DIR / "sender_location_feature_metadata_v2.json"
SCHEMA_PATH = METADATA_DIR / "sender_location_feature_schema_v2.json"
FROZEN_SPLIT_HASH = "e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242"
DEFAULT_EXPERIMENT = "financial-fraud-stage-09-sender-location-feature-store"


# %% Command-line interface
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, register, materialize, and validate the retained versioned " "sender-location Feast features."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "build", "apply", "materialize", "validate", "all"), default="all")
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--consistency-rows", type=int, default=50)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    return parser.parse_args()


# %% Source loading and snapshot persistence
def load_source_events(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    if sample_rows is not None and sample_rows < 1_000:
        raise ValueError("A smoke sample must contain at least 1,000 rows.")
    events = pd.read_csv(
        path,
        usecols=[ID_COLUMN, "timestamp", SENDER_COLUMN, LOCATION_COLUMN, TARGET_COLUMN],
        nrows=sample_rows,
        dtype={ID_COLUMN: "string", SENDER_COLUMN: "string", LOCATION_COLUMN: "string", TARGET_COLUMN: "boolean"},
    ).rename(columns={"timestamp": TIME_COLUMN})
    events[TIME_COLUMN] = pd.to_datetime(events[TIME_COLUMN], errors="raise", format="ISO8601", utc=True)
    events[TARGET_COLUMN] = events[TARGET_COLUMN].astype("int8")
    return events


def create_artifacts(events: pd.DataFrame, random_state: int) -> BehaviourFeatureArtifacts:
    partition = make_stratified_random_partition(
        events[TARGET_COLUMN], train_fraction=0.70, validation_fraction=0.15, seed=random_state
    )
    return build_behaviour_feature_artifacts(events, partition)


def schema_payload(artifacts: BehaviourFeatureArtifacts) -> dict[str, Any]:
    return {
        "feature_version": BEHAVIOUR_FEATURE_VERSION,
        "feature_service": FEATURE_SERVICE_NAME,
        "tables": {
            "sender": [
                {"name": column, "pandas_dtype": str(dtype)} for column, dtype in artifacts.sender_table.dtypes.items()
            ],
            "sender_location": [
                {"name": column, "pandas_dtype": str(dtype)}
                for column, dtype in artifacts.sender_location_table.dtypes.items()
            ],
        },
        "target_registered_as_feature": False,
    }


def persist_canonical_artifacts(artifacts: BehaviourFeatureArtifacts, raw_path: Path) -> dict[str, Any]:
    save_parquet_atomically(artifacts.sender_table, SENDER_TABLE_PATH)
    save_parquet_atomically(artifacts.sender_location_table, SENDER_LOCATION_TABLE_PATH)
    metadata = feature_contract_metadata(
        artifacts, raw_path, SENDER_TABLE_PATH, SENDER_LOCATION_TABLE_PATH, FROZEN_SPLIT_HASH
    )
    save_json_atomically(metadata, METADATA_PATH)
    save_json_atomically(schema_payload(artifacts), SCHEMA_PATH)
    return metadata


# %% Feast registration, materialization, and validation
def apply_contract(repo_path: Path, sender_path: Path, sender_location_path: Path) -> tuple[FeatureStore, Any]:
    objects = build_behaviour_feast_objects(sender_path, sender_location_path)
    store = FeatureStore(repo_path=str(repo_path))
    store.apply(list(objects))
    return store, objects[-1]


def materialize_contract(
    store: FeatureStore, sender_path: Path, sender_location_path: Path
) -> tuple[pd.Timestamp, pd.Timestamp]:
    sender_times = pd.read_parquet(sender_path, columns=[TIME_COLUMN])[TIME_COLUMN]
    pair_times = pd.read_parquet(sender_location_path, columns=[TIME_COLUMN])[TIME_COLUMN]
    start = min(sender_times.min(), pair_times.min()) - pd.Timedelta(microseconds=1)
    end = max(sender_times.max(), pair_times.max()) + pd.Timedelta(microseconds=1)
    store.materialize(
        start_date=start.to_pydatetime(),
        end_date=end.to_pydatetime(),
        feature_views=[SENDER_VIEW_NAME, SENDER_LOCATION_VIEW_NAME],
    )
    return start, end


def sender_from_composite_key(key: str) -> str:
    length_text, remainder = key.split(":", maxsplit=1)
    sender_length = int(length_text)
    sender = remainder[:sender_length]
    separator_position = sender_length
    if remainder[separator_position : separator_position + 1] != "|":
        raise ValueError("Invalid sender-location composite key.")
    return sender


def latest_state_queries(sender_location_path: Path, query_timestamp: pd.Timestamp, row_count: int) -> pd.DataFrame:
    if row_count < 1:
        raise ValueError("At least one consistency row is required.")
    pair_states = pd.read_parquet(sender_location_path, columns=[SENDER_LOCATION_KEY, TIME_COLUMN])
    selected = (
        pair_states.sort_values(TIME_COLUMN, kind="mergesort")
        .drop_duplicates(SENDER_LOCATION_KEY, keep="last")
        .tail(row_count)
        .copy()
    )
    selected[SENDER_COLUMN] = selected[SENDER_LOCATION_KEY].map(sender_from_composite_key)
    selected[TIME_COLUMN] = query_timestamp
    return selected[[SENDER_COLUMN, SENDER_LOCATION_KEY, TIME_COLUMN]].reset_index(drop=True)


def validate_contract(
    store: FeatureStore, service: Any, sender_location_path: Path, materialization_end: pd.Timestamp, row_count: int
) -> pd.DataFrame:
    query_rows = latest_state_queries(sender_location_path, materialization_end, row_count)
    return validate_offline_online_consistency(store, service, query_rows)


# %% MLflow evidence
def log_stage_09_run(
    *, evidence_scope: str, metadata: dict[str, Any], consistency: pd.DataFrame, experiment_name: str
) -> str:
    configure_mlflow(experiment_name)
    split_hash = metadata.get("frozen_split_sha256", metadata.get("split_assignment_sha256"))
    if split_hash is None:
        raise ValueError("Feature-store metadata is missing the split hash.")
    source_files = [
        Path(__file__).resolve(),
        PROJECT_DIR / "sender_location_feature_store.py",
        PROJECT_DIR / "project_io_utils.py",
        PROJECT_DIR / "fraud_feature_definitions.py",
        PROJECT_DIR / "feature_store.yaml",
        PROJECT_DIR / "Markdown files" / "09_sender_location_feature_store_implementation_guide.md",
        PROJECT_DIR / "Markdown files" / "09_sender_location_feature_store_results.md",
    ]
    with mlflow.start_run(run_name=f"sender_location_feature_store_{evidence_scope}") as run:
        mlflow.set_tags(
            {
                "stage": "09_sender_location_feature_store",
                "evidence_scope": evidence_scope,
                "feature_version": BEHAVIOUR_FEATURE_VERSION,
                "offline_online_consistency": "passed",
                "model_trained": "false",
                "test_split_evaluated": "false",
            }
        )
        mlflow.log_params(
            {
                "feature_service": FEATURE_SERVICE_NAME,
                "sender_feature_view": SENDER_VIEW_NAME,
                "sender_location_feature_view": SENDER_LOCATION_VIEW_NAME,
                "consistency_rows": len(consistency),
                "split_assignment_sha256": split_hash,
            }
        )
        mlflow.log_dict(metadata, "feature_contract_metadata.json")
        mlflow.log_text(consistency.to_csv(index=False), "offline_online_consistency.csv")
        for source_file in source_files:
            if source_file.exists():
                mlflow.log_artifact(str(source_file), artifact_path="source_snapshot")
        return run.info.run_id


# %% Isolated smoke workflow
def write_temporary_config(repo_path: Path) -> None:
    data_dir = repo_path / "feature_repo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (repo_path / "feature_store.yaml").write_text(
        "\n".join(
            [
                "project: financial_fraud_stage_09_smoke",
                "registry: feature_repo/data/registry.db",
                "provider: local",
                "online_store:",
                "  type: sqlite",
                "  path: feature_repo/data/online_store.db",
            ]
        ),
        encoding="utf-8",
    )


def run_smoke(args: argparse.Namespace) -> None:
    events = load_source_events(args.raw_data_path, args.sample_rows)
    artifacts = create_artifacts(events, args.random_state)
    with TemporaryDirectory(prefix="fraud_stage_09_smoke_") as directory:
        repo_path = Path(directory)
        write_temporary_config(repo_path)
        sender_path = repo_path / "feature_repo" / "data" / SENDER_TABLE_PATH.name
        pair_path = repo_path / "feature_repo" / "data" / SENDER_LOCATION_TABLE_PATH.name
        save_parquet_atomically(artifacts.sender_table, sender_path)
        save_parquet_atomically(artifacts.sender_location_table, pair_path)
        store, service = apply_contract(repo_path, sender_path, pair_path)
        try:
            _, end = materialize_contract(store, sender_path, pair_path)
            consistency = validate_contract(
                store, service, pair_path, end, min(args.consistency_rows, len(artifacts.sender_location_table))
            )
        finally:
            close_local_online_store(store)
        metadata = {
            "feature_version": BEHAVIOUR_FEATURE_VERSION,
            "feature_service": FEATURE_SERVICE_NAME,
            "split_assignment_sha256": artifacts.split_hash,
            "split_counts": artifacts.split_counts,
            "sender_snapshot_rows": len(artifacts.sender_table),
            "sender_location_snapshot_rows": len(artifacts.sender_location_table),
            "target_registered_as_feature": False,
            "target_used_in_feature_calculation": False,
            "test_events_used": False,
            "smoke_result_is_thesis_evidence": False,
        }
        run_id = log_stage_09_run(
            evidence_scope="smoke_test",
            metadata=metadata,
            consistency=consistency,
            experiment_name=args.experiment_name,
        )
    print(f"Stage 09 smoke test passed for {len(consistency):,} entity rows.")
    print(f"MLflow run: {run_id}")
    print("Temporary Feast artifacts were removed; canonical v2 was not changed.")


# %% Canonical workflow
def run_canonical(args: argparse.Namespace) -> None:
    metadata: dict[str, Any] | None = None
    artifacts: BehaviourFeatureArtifacts | None = None
    if args.mode in ("build", "all"):
        events = load_source_events(args.raw_data_path)
        artifacts = create_artifacts(events, args.random_state)
        metadata = persist_canonical_artifacts(artifacts, args.raw_data_path)
        print(
            f"Saved {len(artifacts.sender_table):,} sender snapshots and "
            f"{len(artifacts.sender_location_table):,} sender-location snapshots."
        )
        if args.mode == "build":
            return

    required_paths = [SENDER_TABLE_PATH, SENDER_LOCATION_TABLE_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Build the Stage 09 feature tables first. Missing: " + ", ".join(missing))

    store, service = apply_contract(PROJECT_DIR, SENDER_TABLE_PATH, SENDER_LOCATION_TABLE_PATH)
    print(f"Applied Feast feature service: {FEATURE_SERVICE_NAME}.")
    if args.mode == "apply":
        return

    _, end = materialize_contract(store, SENDER_TABLE_PATH, SENDER_LOCATION_TABLE_PATH)
    print("Materialized sender and sender-location state to the SQLite online store.")
    if args.mode == "materialize":
        return

    consistency = validate_contract(store, service, SENDER_LOCATION_TABLE_PATH, end, args.consistency_rows)
    print(f"Offline/online consistency passed for {len(consistency):,} entity rows.")
    if args.mode == "validate":
        return

    if metadata is None:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    run_id = log_stage_09_run(
        evidence_scope="full_data", metadata=metadata, consistency=consistency, experiment_name=args.experiment_name
    )
    print(f"Stage 09 complete. MLflow run: {run_id}")
    print("No model was trained and the test split was not evaluated.")


# %% Cell-aware executable entry point
def main() -> None:
    args = parse_args()
    if args.random_state != 42:
        raise ValueError("Stage 09 is frozen to random state 42.")
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_canonical(args)


if __name__ == "__main__":
    main()
