"""Strict reconstruction of frozen training rows without retaining held-out features."""

# %% Imports and fixed split contract
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.common.modeling import FEATURE_COLUMNS, make_optional_sample
from fraud_detection.modeling.model_families import FROZEN_FULL_SPLIT_HASH
from fraud_detection.diagnostics.predictive_quality import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    make_stratified_random_partition,
    stable_split_hash,
)
from fraud_detection.diagnostics.predictive_signal import extract_training_rows

SMOKE_ROWS = 50_000
FULL_ROWS = 5_000_000


# %% Membership reconstruction from identifiers and labels only
def frozen_training_membership(
    path: Path,
    mode: str,
    *,
    expected_hash: str = FROZEN_FULL_SPLIT_HASH,
    expected_rows: int = FULL_ROWS,
    smoke_rows: int = SMOKE_ROWS,
) -> tuple[pd.DataFrame, dict, tuple[int, int]]:
    if mode not in {"smoke", "full"}:
        raise ValueError("Mode must be smoke or full.")
    before = path.stat()
    metadata = pd.read_csv(path, usecols=[ID_COLUMN, TARGET_COLUMN], dtype={ID_COLUMN: "string"})
    if (
        metadata[ID_COLUMN].isna().any()
        or metadata[ID_COLUMN].duplicated().any()
        or not metadata[ID_COLUMN].str.fullmatch(r"T\d+").all()
        or metadata[TARGET_COLUMN].isna().any()
        or not metadata[TARGET_COLUMN].isin([0, 1]).all()
    ):
        raise ValueError("Missing/invalid IDs or labels prevent safe frozen partition reconstruction.")
    if mode == "full" and len(metadata) != expected_rows:
        raise ValueError("Full reconstruction requires the canonical source row count.")
    metadata[TARGET_COLUMN] = metadata[TARGET_COLUMN].astype("int8")
    metadata = metadata.assign(_order=metadata[ID_COLUMN].str.slice(1).astype("int64"))
    if metadata._order.duplicated().any():
        raise ValueError("Numeric transaction-ID suffixes must be unique.")
    metadata = metadata.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)
    if mode == "smoke":
        metadata = make_optional_sample(metadata, smoke_rows, 42)
        metadata = metadata.assign(_order=metadata[ID_COLUMN].str.slice(1).astype("int64"))
        metadata = metadata.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)
    partition = make_stratified_random_partition(metadata[TARGET_COLUMN], seed=42)
    split_hash = stable_split_hash(metadata[ID_COLUMN], partition)
    if mode == "full" and split_hash != expected_hash:
        raise ValueError("Source data does not reproduce the frozen training split.")
    membership = extract_training_rows(metadata, partition).reset_index(drop=True)
    info = {
        "scope": "training_partition_only",
        "evidence_scope": "full_data" if mode == "full" else "smoke_test",
        "seed": 42,
        "split_assignment_sha256": split_hash,
        "partition_rows": {
            name: int((partition == code).sum()) for code, name in enumerate(("train", "validation", "test"))
        },
        "diagnostic_training_membership_sha256": stable_split_hash(
            membership[ID_COLUMN], np.zeros(len(membership), dtype="uint8")
        ),
        "nontraining_feature_analysis": False,
        "source_filename": path.name,
        "source_size_bytes": before.st_size,
    }
    return membership, info, (before.st_size, before.st_mtime_ns)


# %% Chunked feature extraction for training identifiers only
def load_training_data(
    path: Path,
    mode: str,
    *,
    expected_hash: str = FROZEN_FULL_SPLIT_HASH,
    expected_rows: int = FULL_ROWS,
    smoke_rows: int = SMOKE_ROWS,
) -> tuple[pd.DataFrame, dict]:
    membership, info, source_stat = frozen_training_membership(
        path, mode, expected_hash=expected_hash, expected_rows=expected_rows, smoke_rows=smoke_rows
    )
    print(f"Selected training={len(membership):,}; validation and test features are excluded.", flush=True)
    gc.collect()
    header = pd.read_csv(path, nrows=0).columns
    columns = [ID_COLUMN, "timestamp", "sender_account", *FEATURE_COLUMNS, TARGET_COLUMN]
    if "fraud_type" in header:
        columns.append("fraud_type")
    missing = set(columns).difference(header)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    ids = pd.Index(membership[ID_COLUMN])
    parts = []
    for chunk in pd.read_csv(
        path, usecols=columns, dtype={ID_COLUMN: "string", "sender_account": "string"}, chunksize=250_000
    ):
        selected = chunk.loc[chunk[ID_COLUMN].isin(ids)].copy()
        if len(selected):
            parts.append(selected)
    training = pd.concat(parts, ignore_index=True).set_index(ID_COLUMN).reindex(ids).reset_index()
    if len(training) != len(membership) or not np.array_equal(
        training[TARGET_COLUMN].to_numpy(), membership[TARGET_COLUMN].to_numpy()
    ):
        raise ValueError("Retrieved training rows do not match frozen membership and labels.")
    training = training.rename(columns={"timestamp": TIME_COLUMN})
    training[TIME_COLUMN] = pd.to_datetime(training[TIME_COLUMN], format="ISO8601", utc=True, errors="coerce")
    after = path.stat()
    if source_stat != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while reconstructing the training frame.")
    return training, info
