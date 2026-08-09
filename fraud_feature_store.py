"""Load and validate the canonical Feast feature handoff for model training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_VERSION = "v1"
FEATURE_VIEW_NAME = "fraud_transaction_features"
FEATURE_SERVICE_NAME = f"fraud_model_features_{FEATURE_VERSION}"
ENTITY_COLUMN = "transaction_id"
TIMESTAMP_COLUMN = "event_timestamp"
TARGET_COLUMN = "is_fraud"

MODEL_FEATURE_COLUMNS = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_used",
    "payment_channel",
    "amount",
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]

FEATURE_REFERENCES = [f"{FEATURE_VIEW_NAME}:{feature}" for feature in MODEL_FEATURE_COLUMNS]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def handoff_paths(repo_path: Path) -> dict[str, Path]:
    return {
        "feature_table": (repo_path / "feature_repo" / "data" / f"fraud_features_{FEATURE_VERSION}.parquet"),
        "label_table": (repo_path / "feature_repo" / "data" / f"fraud_labels_{FEATURE_VERSION}.parquet"),
        "registry": repo_path / "feature_repo" / "data" / "registry.db",
        "metadata": (repo_path / "feature_repo" / "metadata" / f"feature_metadata_{FEATURE_VERSION}.json"),
        "schema": (repo_path / "feature_repo" / "metadata" / f"feature_schema_{FEATURE_VERSION}.json"),
        "definitions": repo_path / "fraud_feature_definitions.py",
        "config": repo_path / "feature_store.yaml",
    }


def require_handoff(paths: dict[str, Path]) -> None:
    required = [
        "feature_table",
        "label_table",
        "registry",
        "metadata",
        "schema",
        "definitions",
        "config",
    ]
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        missing_text = "\n- ".join(missing)
        raise FileNotFoundError(
            "The canonical Feast handoff is incomplete. Missing:\n"
            f"- {missing_text}\n"
            "Run 02_data_pipeline_preprocessing.py, then run `feast apply` from "
            "the Code snippets directory."
        )


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "feature_version": FEATURE_VERSION,
        "feature_view": FEATURE_VIEW_NAME,
        "feature_service": FEATURE_SERVICE_NAME,
        "target_registered_as_feature": False,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Feature handoff metadata does not match the code: {mismatches}")


def validate_handoff_hashes(paths: dict[str, Path], metadata: dict[str, Any]) -> None:
    artifact_metadata = metadata["artifacts"]
    expected_hashes = {
        "feature_table": artifact_metadata["feature_table"]["sha256"],
        "label_table": artifact_metadata["label_table"]["sha256"],
        "schema": artifact_metadata["feature_schema"]["sha256"],
    }
    for name, expected_hash in expected_hashes.items():
        actual_hash = sha256_file(paths[name])
        if actual_hash != expected_hash:
            raise ValueError(
                f"{name} hash does not match feature_metadata_{FEATURE_VERSION}.json. "
                "Regenerate the handoff before training."
            )


def validate_retrieved_table(table: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    required_columns = {
        ENTITY_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        *MODEL_FEATURE_COLUMNS,
    }
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(f"Feast retrieval is missing required columns: {missing}")
    if len(table) != int(metadata["row_count"]):
        raise ValueError(f"Feast returned {len(table):,} rows; expected {metadata['row_count']:,}.")
    if table[ENTITY_COLUMN].duplicated().any():
        raise ValueError("Feast retrieval duplicated transaction_id values.")
    if table[TARGET_COLUMN].isna().any():
        raise ValueError("Feast retrieval contains missing labels.")
    if int(table[TARGET_COLUMN].sum()) != int(metadata["fraud_count"]):
        raise ValueError("Feast retrieval changed the recorded fraud count.")

    table = table[[ENTITY_COLUMN, TIMESTAMP_COLUMN, *MODEL_FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    table[TIMESTAMP_COLUMN] = pd.to_datetime(table[TIMESTAMP_COLUMN], errors="raise", format="mixed")
    table[TARGET_COLUMN] = table[TARGET_COLUMN].astype("int8")
    return table


def load_feast_training_table(
    repo_path: Path,
    *,
    validate_hashes: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str | int | float | bool]]:
    """Retrieve the canonical point-in-time training table through Feast."""
    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise RuntimeError(
            "Feast is required for the canonical modeling path. Install the "
            "project requirements or use --data-source csv for an explicit fallback."
        ) from exc

    repo_path = repo_path.resolve()
    paths = handoff_paths(repo_path)
    require_handoff(paths)
    metadata = load_metadata(paths["metadata"])
    validate_metadata(metadata)
    if validate_hashes:
        validate_handoff_hashes(paths, metadata)

    labels = pd.read_parquet(
        paths["label_table"],
        columns=[ENTITY_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN],
    )
    labels[TIMESTAMP_COLUMN] = pd.to_datetime(labels[TIMESTAMP_COLUMN], errors="raise", format="mixed")

    store = FeatureStore(repo_path=str(repo_path))
    retrieval_job = store.get_historical_features(
        entity_df=labels,
        features=FEATURE_REFERENCES,
        full_feature_names=False,
    )
    table = retrieval_job.to_df()
    table = validate_retrieved_table(table, metadata)

    feature_artifact = metadata["artifacts"]["feature_table"]
    label_artifact = metadata["artifacts"]["label_table"]
    dataset_info: dict[str, Any] = {
        "dataset_name": paths["feature_table"].name,
        "dataset_absolute_path": str(paths["feature_table"].resolve()),
        "dataset_size_bytes": int(feature_artifact["size_bytes"]) + int(label_artifact["size_bytes"]),
        "dataset_modified_time_ns": max(
            int(paths["feature_table"].stat().st_mtime_ns),
            int(paths["label_table"].stat().st_mtime_ns),
        ),
        "dataset_sha256": feature_artifact["sha256"],
    }
    lineage: dict[str, str | int | float | bool] = {
        "data_interface": "feast_historical_retrieval",
        "feature_store": "feast_local_file_offline_store",
        "feature_version": metadata["feature_version"],
        "feature_view": metadata["feature_view"],
        "feature_service": metadata["feature_service"],
        "feature_entity": metadata["entity"],
        "feature_entity_join_key": metadata["entity_join_key"],
        "feature_table_sha256": feature_artifact["sha256"],
        "label_table_sha256": label_artifact["sha256"],
        "feature_schema_sha256": metadata["artifacts"]["feature_schema"]["sha256"],
        "feature_registry_sha256": sha256_file(paths["registry"]),
        "feature_definitions_sha256": sha256_file(paths["definitions"]),
        "feature_handoff_row_count": int(metadata["row_count"]),
        "feature_handoff_fraud_rate": float(metadata["fraud_rate"]),
        "feature_hash_validation_enabled": validate_hashes,
    }
    return table, dataset_info, lineage
