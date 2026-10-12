"""Verified frozen-history serving; no training, test scoring or registry writes."""

# %% Imports and fixed references
from __future__ import annotations

import json
import math
import shutil
import sqlite3
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from feast import FeatureStore
from feast.protos.feast.serving.ServingService_pb2 import PRESENT, NOT_FOUND
from feast.repo_config import RepoConfig

from final_test_evaluation_utils import FROZEN_THRESHOLD
from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH
from oversampling_experiment_utils import FEATURE_SET_NAME, MODEL_FEATURES
from project_io_utils import save_json_atomically, sha256_file
from sender_location_feature_store import (
    FEATURE_SERVICE_NAME,
    NEW_LOCATION_FEATURE,
    SENDER_COLUMN,
    SENDER_COUNT_FEATURE,
    SENDER_LOCATION_COUNT_FEATURE,
    SENDER_LOCATION_KEY,
    SENDER_LOCATION_VIEW_NAME,
    SENDER_VIEW_NAME,
    close_local_online_store,
    compose_sender_location_key,
)

FINAL_RUN_ID = "37b41018f28646598344fbcfe4ea4db6"
CONTRACT_VERSION = "research_serving_v1"
PROJECT = "financial_fraud_detection"
STATE_SPECS = (
    ("sender", SENDER_COLUMN, SENDER_COUNT_FEATURE, SENDER_VIEW_NAME),
    ("pair", SENDER_LOCATION_KEY, SENDER_LOCATION_COUNT_FEATURE, SENDER_LOCATION_VIEW_NAME),
)


class ServingError(RuntimeError):
    """Public, non-sensitive failure with an appropriate HTTP status."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


@contextmanager
def database(path: Path, read_only: bool = False):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) if read_only else sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def read_only_connection(path: Path):
    return database(path, read_only=True)


# %% Immutable latest-state membership and count index
def build_state_index(paths: dict[str, Path], destination: Path) -> dict[str, Any]:
    """Stream snapshots into a disk index; never load millions of keys into RAM."""
    if destination.exists():
        raise ValueError("Refusing to replace an existing state index.")
    boundary = None
    counts = {}
    with database(destination) as connection:
        for table, key, count, _ in STATE_SPECS:
            connection.execute(f"CREATE TABLE {table} (entity TEXT PRIMARY KEY, value INTEGER, effective_ns INTEGER)")
            parquet = pq.ParquetFile(paths[table])
            for batch in parquet.iter_batches(batch_size=100_000, columns=[key, count, "event_timestamp"]):
                frame = batch.to_pandas()
                times = pd.to_datetime(frame["event_timestamp"], utc=True).astype("int64")
                if frame[[key, count]].isna().any().any() or (frame[count] < 0).any():
                    raise ValueError("Invalid source state.")
                latest = int(times.max())
                boundary = latest if boundary is None else max(boundary, latest)
                records = ((str(k), int(v), int(t)) for k, v, t in zip(frame[key], frame[count], times))
                connection.executemany(
                    f"INSERT INTO {table} VALUES (?, ?, ?) ON CONFLICT(entity) DO UPDATE SET "
                    "value=excluded.value, effective_ns=excluded.effective_ns "
                    f"WHERE excluded.effective_ns > {table}.effective_ns",
                    records,
                )
            counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if boundary is None or not all(counts.values()):
        raise ValueError("Empty history state.")
    return {"state_cutoff": pd.Timestamp(boundary, tz="UTC").isoformat(), "entity_counts": counts}


def verify_online_rows(online_path: Path, expected: dict[str, int]) -> None:
    """Reject incomplete materialisation; per-request values are checked against the index."""
    with read_only_connection(online_path) as connection:
        for table, _, feature, view in STATE_SPECS:
            actual = connection.execute(
                f'SELECT COUNT(*) FROM "{PROJECT}_{view}" WHERE feature_name=?', (feature,)
            ).fetchone()[0]
            if actual != expected[table]:
                raise ServingError("Online state is incompletely materialised.")


def prepare_deployment(project_dir: Path, output_dir: Path) -> Path:
    """Create an isolated local serving bundle without touching canonical Feast state."""
    if output_dir.exists():
        raise ValueError("Serving bundle already exists; verify it instead of overwriting it.")
    artifacts = project_dir / "mlartifacts" / FINAL_RUN_ID / "artifacts"
    protocol = json.loads((artifacts / "frozen_evaluation_protocol.json").read_text(encoding="utf-8"))
    metadata_path = project_dir / "feature_repo/metadata/sender_location_feature_metadata_v2.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        protocol["threshold"] != FROZEN_THRESHOLD
        or protocol["feature_set"] != FEATURE_SET_NAME
        or protocol["split_assignment_sha256"] != FROZEN_FULL_SPLIT_HASH
        or metadata["frozen_split_sha256"] != FROZEN_FULL_SPLIT_HASH
        or metadata["feature_service"] != FEATURE_SERVICE_NAME
        or metadata["validation_events_update_history"] is not False
        or metadata["test_events_used"] is not False
    ):
        raise ValueError("Model and feature-state references do not match the frozen protocol.")
    versions = json.loads((artifacts / "library_versions.json").read_text(encoding="utf-8"))
    for package in ("scikit-learn", "lightgbm", "pandas", "numpy", "joblib"):
        if version(package) != versions[package]:
            raise ValueError(f"Evaluated model library version mismatch: {package}.")
    paths = {}
    for table, _, _, _ in STATE_SPECS:
        entry = metadata["artifacts"]["sender_table" if table == "sender" else "sender_location_table"]
        paths[table] = project_dir / "feature_repo/data" / entry["name"]
        if sha256_file(paths[table]) != entry["sha256"]:
            raise ValueError("Canonical snapshot checksum mismatch.")
    output_dir.mkdir(parents=True)
    # An interrupted preparation is deliberately not made ready; no manifest is written until checks pass.
    state = build_state_index(paths, output_dir / "state_index.sqlite")
    with read_only_connection(project_dir / "feature_repo/data/online_store.db") as source:
        with database(output_dir / "online_store.sqlite") as destination:
            source.backup(destination)
    verify_online_rows(output_dir / "online_store.sqlite", state["entity_counts"])
    shutil.copyfile(project_dir / "feature_repo/data/registry.db", output_dir / "registry.db")
    shutil.copyfile(artifacts / "evaluated_model/evaluated_lightgbm_pipeline.joblib", output_dir / "model.joblib")
    shutil.copyfile(metadata_path, output_dir / "feature_metadata.json")
    files = {
        name: sha256_file(output_dir / name)
        for name in (
            "model.joblib",
            "registry.db",
            "online_store.sqlite",
            "state_index.sqlite",
            "feature_metadata.json",
        )
    }
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "model_run_id": FINAL_RUN_ID,
        "model_registered": False,
        "production_approved": False,
        "threshold": FROZEN_THRESHOLD,
        "feature_set": FEATURE_SET_NAME,
        "feature_service": FEATURE_SERVICE_NAME,
        "split_sha256": FROZEN_FULL_SPLIT_HASH,
        "history_population": "frozen_training_only",
        "model_features": MODEL_FEATURES,
        "library_versions": versions,
        "files": files,
        **state,
    }
    manifest["state_version"] = files["state_index.sqlite"]
    manifest_path = save_json_atomically(manifest, output_dir / "manifest.json")
    service = None
    try:
        service = load_deployment(manifest_path, output_dir / "predictions.sqlite")
        service.check_ready()
    except Exception:
        # A failing bundle remains available for diagnosis but cannot be served.
        manifest_path.unlink()
        raise
    finally:
        if service is not None:
            service.close()
    return manifest_path


# %% Feast reads with explicit status and expected-value checks
class FeastFrozenHistory:
    def __init__(self, bundle: Path, manifest: dict[str, Any]):
        self.bundle, self.manifest = bundle, manifest
        config = RepoConfig(
            project=PROJECT,
            provider="local",
            registry=str(bundle / "registry.db"),
            online_store={"type": "sqlite", "path": str(bundle / "online_store.sqlite")},
        )
        self.store = FeatureStore(config=config)
        for _, key, feature, view_name in STATE_SPECS:
            view = self.store.get_feature_view(view_name)
            if feature not in {field.name for field in view.features} or key not in {
                self.store.get_entity(entity).join_key for entity in view.entities
            }:
                raise ServingError("Feast registry schema mismatch.")
        verify_online_rows(bundle / "online_store.sqlite", manifest["entity_counts"])
        with read_only_connection(bundle / "state_index.sqlite") as connection:
            self.health_keys = {
                table: connection.execute(f"SELECT entity FROM {table} ORDER BY effective_ns DESC LIMIT 1").fetchone()[
                    0
                ]
                for table, _, _, _ in STATE_SPECS
            }
        self.file_fingerprints = {
            name: ((bundle / name).stat().st_size, (bundle / name).stat().st_mtime_ns)
            for name in ("online_store.sqlite", "state_index.sqlite", "registry.db")
        }

    def _expected(self, table: str, key: str) -> int | None:
        with read_only_connection(self.bundle / "state_index.sqlite") as connection:
            row = connection.execute(f"SELECT value FROM {table} WHERE entity=?", (key,)).fetchone()
        return None if row is None else int(row[0])

    def _read(self, table: str, key_name: str, key: str, feature: str, view: str) -> int:
        expected = self._expected(table, key)
        response = self.store.get_online_features(features=[f"{view}:{feature}"], entity_rows=[{key_name: key}])
        name = next((i for i, n in enumerate(response.proto.metadata.feature_names.val) if n == feature), None)
        if name is None:
            raise ServingError("Feature response is missing a required count.")
        statuses = response.proto.results[name].statuses
        if len(statuses) != 1:
            raise ServingError("Invalid feature response status.")
        if expected is None and statuses[0] == NOT_FOUND:
            return 0
        value = response.to_dict().get(feature, [None])[0]
        if statuses[0] != PRESENT or expected is None or isinstance(value, bool) or not isinstance(value, int):
            raise ServingError("Required history state is missing or invalid.")
        if value != expected:
            raise ServingError("Online history does not match the frozen state.")
        return value

    def counts(self, sender: str, location: str) -> tuple[int, int]:
        pair_key = compose_sender_location_key(pd.Series([sender]), pd.Series([location])).iloc[0]
        sender_count = self._read("sender", SENDER_COLUMN, sender, SENDER_COUNT_FEATURE, SENDER_VIEW_NAME)
        pair_count = self._read(
            "pair", SENDER_LOCATION_KEY, pair_key, SENDER_LOCATION_COUNT_FEATURE, SENDER_LOCATION_VIEW_NAME
        )
        return sender_count, pair_count

    def health(self) -> None:
        for name, expected in self.file_fingerprints.items():
            stat = (self.bundle / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ServingError("Frozen serving state changed; prepare a separately versioned bundle.")
        self._read("sender", SENDER_COLUMN, self.health_keys["sender"], SENDER_COUNT_FEATURE, SENDER_VIEW_NAME)
        self._read(
            "pair",
            SENDER_LOCATION_KEY,
            self.health_keys["pair"],
            SENDER_LOCATION_COUNT_FEATURE,
            SENDER_LOCATION_VIEW_NAME,
        )

    def close(self) -> None:
        close_local_online_store(self.store)


# %% Prediction ledger, feature assembly and unchanged research rule
class ResearchPredictor:
    def __init__(self, model: Any, history: Any, manifest: dict[str, Any], log_path: Path):
        self.model, self.history, self.manifest, self.log_path = model, history, manifest, log_path
        self.lock = RLock()
        self.cutoff = pd.Timestamp(manifest["state_cutoff"])
        with database(log_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions "
                "(transaction_id TEXT PRIMARY KEY, request_json TEXT, response_json TEXT, features_json TEXT)"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS events " "(recorded_at TEXT, kind TEXT, detail TEXT)")

    def event(self, kind: str, detail: str) -> None:
        with database(self.log_path) as connection:
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?)", (pd.Timestamp.now(tz="UTC").isoformat(), kind, detail)
            )

    def check_ready(self) -> None:
        with self.lock:
            try:
                self.history.health()
            except Exception as exc:
                raise ServingError("Verified history state is unavailable.") from exc

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            start = perf_counter()
            timestamp = pd.Timestamp(request["event_timestamp"])
            if timestamp <= self.cutoff:
                raise ServingError(
                    "Online requests must be after the frozen state cutoff; historical replay is separate.", 422
                )
            canonical = json.dumps(request, sort_keys=True, allow_nan=False)
            with database(self.log_path) as connection:
                previous = connection.execute(
                    "SELECT request_json, response_json FROM predictions WHERE transaction_id=?",
                    (request["transaction_id"],),
                ).fetchone()
            self.check_ready()
            if previous:
                if canonical != previous[0]:
                    raise ServingError("Transaction ID already used for a different request.", 409)
                self.event("duplicate", request["transaction_id"])
                return json.loads(previous[1])
            try:
                sender, pair = self.history.counts(request[SENDER_COLUMN], request["location"])
                if (
                    isinstance(sender, bool)
                    or isinstance(pair, bool)
                    or not isinstance(sender, int)
                    or not isinstance(pair, int)
                ):
                    raise ServingError("History counts must be integers.")
                if not 0 <= pair <= sender:
                    raise ServingError("History counts are inconsistent.")
                features = {name: request[name] for name in MODEL_FEATURES if name in request}
                features.update(
                    {
                        SENDER_COUNT_FEATURE: sender,
                        SENDER_LOCATION_COUNT_FEATURE: pair,
                        NEW_LOCATION_FEATURE: int(sender > 0 and pair == 0),
                    }
                )
                frame = pd.DataFrame(
                    [{key: np.nan if value is None else value for key, value in features.items()}],
                    columns=MODEL_FEATURES,
                )
                classes = list(self.model.classes_)
                score = float(self.model.predict_proba(frame)[0, classes.index(1)])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ServingError("Model returned an invalid score.")
                result = {
                    "transaction_id": request["transaction_id"],
                    "score": score,
                    "decision": "review" if score >= FROZEN_THRESHOLD else "not_flagged",
                    "threshold": FROZEN_THRESHOLD,
                    "model_run_id": self.manifest["model_run_id"],
                    "model_registered": False,
                    "production_approved": False,
                    "contract_version": CONTRACT_VERSION,
                    "feature_state_version": self.manifest["state_version"],
                    "state_cutoff": self.cutoff.isoformat(),
                    "event_timestamp": request["event_timestamp"],
                    "cold_start": sender == 0,
                    "processing_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                    "latency_ms": (perf_counter() - start) * 1000,
                }
                with database(self.log_path) as connection:
                    connection.execute(
                        "INSERT INTO predictions VALUES (?, ?, ?, ?)",
                        (request["transaction_id"], canonical, json.dumps(result), json.dumps(features)),
                    )
                return result
            except ServingError:
                raise
            except Exception as exc:
                raise ServingError("Prediction failed; no decision was recorded.") from exc

    def close(self) -> None:
        self.history.close()


def load_deployment(manifest_path: Path, log_path: Path) -> ResearchPredictor:
    """Verify trusted local artifacts before unpickling; never accept uploaded joblib files."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["model_run_id"] != FINAL_RUN_ID
        or manifest["contract_version"] != CONTRACT_VERSION
        or manifest["threshold"] != FROZEN_THRESHOLD
        or manifest["split_sha256"] != FROZEN_FULL_SPLIT_HASH
        or manifest["model_features"] != MODEL_FEATURES
        or manifest["feature_set"] != FEATURE_SET_NAME
        or manifest["feature_service"] != FEATURE_SERVICE_NAME
        or manifest["history_population"] != "frozen_training_only"
        or manifest["production_approved"] is not False
        or manifest["model_registered"] is not False
    ):
        raise ServingError("Serving manifest is inconsistent with the research protocol.")
    bundle = manifest_path.parent.resolve()
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256_file(bundle / name) != expected:
            raise ServingError("Serving artifact checksum mismatch.")
    if set(manifest["files"]) != {
        "model.joblib",
        "registry.db",
        "online_store.sqlite",
        "state_index.sqlite",
        "feature_metadata.json",
    }:
        raise ServingError("Required serving artifacts are missing.")
    if manifest["state_version"] != manifest["files"]["state_index.sqlite"]:
        raise ServingError("Feature-state version does not match its artifact.")
    with read_only_connection(bundle / "state_index.sqlite") as connection:
        latest = max(
            connection.execute(f"SELECT MAX(effective_ns) FROM {table}").fetchone()[0] for table, _, _, _ in STATE_SPECS
        )
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table, _, _, _ in STATE_SPECS
        }
    if pd.Timestamp(manifest["state_cutoff"]) != pd.Timestamp(latest, tz="UTC") or counts != manifest["entity_counts"]:
        raise ServingError("History boundary or membership counts mismatch.")
    for package in ("scikit-learn", "lightgbm", "pandas", "numpy", "joblib"):
        if version(package) != manifest["library_versions"][package]:
            raise ServingError("Model runtime version mismatch.")
    model = joblib.load(bundle / "model.joblib")
    if list(model.feature_names_in_) != MODEL_FEATURES or set(model.classes_) != {0, 1}:
        raise ServingError("Model input schema or classes mismatch.")
    predictor = ResearchPredictor(model, FeastFrozenHistory(bundle, manifest), manifest, log_path)
    try:
        predictor.check_ready()
    except Exception:
        predictor.close()
        raise
    return predictor
