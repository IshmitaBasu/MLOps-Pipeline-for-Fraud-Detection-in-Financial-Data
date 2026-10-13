"""Shared hashing and atomic persistence helpers for modeling artifacts."""

# %% Imports
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


# %% File integrity and persistence
def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 checksum."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_parquet_atomically(table: pd.DataFrame, path: Path) -> Path:
    """Write Parquet through a sibling temporary file before replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}_temporary{path.suffix}")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def save_json_atomically(payload: Any, path: Path) -> Path:
    """Write JSON through a sibling temporary file before replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}_temporary{path.suffix}")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
    return path
