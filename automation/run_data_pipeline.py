"""Incrementally ingest stable CSV batches into the local data-pipeline database."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import time
from types import ModuleType
from typing import Any, Iterable

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PIPELINE_SCRIPT = PROJECT_DIR / "02_data_pipeline_preprocessing.py"
DEFAULT_DATA_ROOT = PROJECT_DIR / "data"
DEFAULT_DATABASE_NAME = "fraud_pipeline.db"

RAW_COLUMN_TYPES = {
    "transaction_id": "TEXT NOT NULL",
    "timestamp": "TEXT NOT NULL",
    "sender_account": "TEXT",
    "receiver_account": "TEXT",
    "amount": "REAL",
    "transaction_type": "TEXT",
    "merchant_category": "TEXT",
    "location": "TEXT",
    "device_used": "TEXT",
    "is_fraud": "INTEGER NOT NULL",
    "fraud_type": "TEXT",
    "time_since_last_transaction": "REAL",
    "spending_deviation_score": "REAL",
    "velocity_score": "INTEGER",
    "geo_anomaly_score": "REAL",
    "payment_channel": "TEXT",
    "ip_address": "TEXT",
    "device_hash": "TEXT",
}

CLEAN_COLUMN_TYPES = {
    "transaction_id": "TEXT NOT NULL UNIQUE",
    "event_timestamp": "TEXT NOT NULL",
    "transaction_type": "TEXT",
    "merchant_category": "TEXT",
    "location": "TEXT",
    "device_used": "TEXT",
    "payment_channel": "TEXT",
    "amount": "REAL",
    "time_since_last_transaction": "REAL",
    "spending_deviation_score": "REAL",
    "velocity_score": "INTEGER",
    "geo_anomaly_score": "REAL",
    "is_fraud": "INTEGER NOT NULL",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(moment: datetime | None = None) -> str:
    return (moment or utc_now()).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_pipeline_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("data_pipeline_preprocessing", DATA_PIPELINE_SCRIPT)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {DATA_PIPELINE_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def ensure_data_store(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    inbox = root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    database = root / DEFAULT_DATABASE_NAME
    with open_database(database) as connection:
        create_tables(connection)
    return inbox, database


def connect_database(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=60)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def open_database(database: Path) -> Iterable[sqlite3.Connection]:
    connection = connect_database(database)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def create_tables(connection: sqlite3.Connection) -> None:
    raw_columns = ",\n                ".join(f'"{name}" {sql_type}' for name, sql_type in RAW_COLUMN_TYPES.items())
    clean_columns = ",\n                ".join(f'"{name}" {sql_type}' for name, sql_type in CLEAN_COLUMN_TYPES.items())
    connection.executescript(f"""
        CREATE TABLE IF NOT EXISTS ingestion_batches (
            batch_id TEXT PRIMARY KEY,
            source_file_name TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            input_size_bytes INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            raw_row_count INTEGER NOT NULL DEFAULT 0,
            clean_row_count INTEGER NOT NULL DEFAULT 0,
            fraud_count INTEGER NOT NULL DEFAULT 0,
            duplicate_of_batch_id TEXT,
            error_message TEXT,
            input_disposition TEXT,
            ml_pipeline_triggered INTEGER NOT NULL DEFAULT 0 CHECK (ml_pipeline_triggered = 0)
        );

        CREATE INDEX IF NOT EXISTS idx_ingestion_batches_checksum_status
            ON ingestion_batches (input_sha256, status);

        CREATE TABLE IF NOT EXISTS raw_transactions (
            batch_id TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            {raw_columns},
            PRIMARY KEY (batch_id, source_row_number),
            FOREIGN KEY (batch_id) REFERENCES ingestion_batches(batch_id)
        );

        CREATE TABLE IF NOT EXISTS clean_transactions (
            batch_id TEXT NOT NULL,
            {clean_columns},
            FOREIGN KEY (batch_id) REFERENCES ingestion_batches(batch_id)
        );
        """)


def is_stable(path: Path, stability_seconds: int, now_epoch: float | None = None) -> bool:
    if stability_seconds < 0:
        raise ValueError("stability_seconds cannot be negative.")
    observed_at = time() if now_epoch is None else now_epoch
    return observed_at - path.stat().st_mtime >= stability_seconds


def sqlite_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def insert_dataframe(
    connection: sqlite3.Connection,
    *,
    table: str,
    frame: pd.DataFrame,
    columns: list[str],
    batch_id: str,
    include_source_row: bool,
    chunk_size: int = 10_000,
) -> None:
    prefix_columns = ["batch_id", "source_row_number"] if include_source_row else ["batch_id"]
    insert_columns = prefix_columns + columns
    column_sql = ", ".join(f'"{column}"' for column in insert_columns)
    placeholders = ", ".join("?" for _ in insert_columns)
    statement = f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})'

    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start : start + chunk_size][columns]
        records = []
        for offset, row in enumerate(chunk.itertuples(index=False, name=None), start=start):
            values = tuple(sqlite_value(value) for value in row)
            prefix = (batch_id, offset) if include_source_row else (batch_id,)
            records.append(prefix + values)
        connection.executemany(statement, records)


def successful_batch_for_checksum(connection: sqlite3.Connection, checksum: str) -> str | None:
    row = connection.execute(
        """
        SELECT batch_id
        FROM ingestion_batches
        WHERE input_sha256 = ? AND status = 'success'
        ORDER BY completed_at_utc
        LIMIT 1
        """,
        (checksum,),
    ).fetchone()
    return None if row is None else str(row[0])


def new_batch_id(checksum: str) -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"batch_{timestamp}_{checksum[:12]}"


def insert_batch_record(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    source_file: Path,
    checksum: str,
    status: str,
    started_at: datetime,
    completed_at: datetime | None = None,
    raw_rows: int = 0,
    clean_rows: int = 0,
    fraud_count: int = 0,
    duplicate_of: str | None = None,
    error_message: str | None = None,
    disposition: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO ingestion_batches (
            batch_id, source_file_name, input_sha256, input_size_bytes, status,
            started_at_utc, completed_at_utc, raw_row_count, clean_row_count,
            fraud_count, duplicate_of_batch_id, error_message, input_disposition,
            ml_pipeline_triggered
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            batch_id,
            source_file.name,
            checksum,
            source_file.stat().st_size,
            status,
            iso_utc(started_at),
            iso_utc(completed_at) if completed_at else None,
            raw_rows,
            clean_rows,
            fraud_count,
            duplicate_of,
            error_message,
            disposition,
        ),
    )


def failed_file_path(source_file: Path, batch_id: str) -> Path:
    candidate = source_file.with_suffix(f"{source_file.suffix}.failed")
    if candidate.exists():
        candidate = source_file.with_name(f"{source_file.name}.{batch_id}.failed")
    return candidate


def consume_ingested_file(source_file: Path, database: Path, batch_id: str, reason: str) -> str:
    try:
        source_file.unlink()
        disposition = f"deleted_{reason}"
    except OSError as exc:
        disposition = f"retained_after_ingestion: {type(exc).__name__}: {exc}"
    with open_database(database) as connection:
        connection.execute(
            "UPDATE ingestion_batches SET input_disposition = ? WHERE batch_id = ?",
            (disposition, batch_id),
        )
    return disposition


def process_file(
    source_file: Path,
    database: Path,
    pipeline_module: ModuleType,
) -> dict[str, Any]:
    started_at = utc_now()
    checksum = sha256_file(source_file)
    batch_id = new_batch_id(checksum)

    duplicate_of: str | None
    with open_database(database) as connection:
        duplicate_of = successful_batch_for_checksum(connection, checksum)
        if duplicate_of is not None:
            insert_batch_record(
                connection,
                batch_id=batch_id,
                source_file=source_file,
                checksum=checksum,
                status="skipped_duplicate",
                started_at=started_at,
                completed_at=utc_now(),
                duplicate_of=duplicate_of,
            )
    if duplicate_of is not None:
        disposition = consume_ingested_file(source_file, database, batch_id, "after_duplicate_check")
        return {
            "batch_id": batch_id,
            "status": "skipped_duplicate",
            "duplicate_of_batch_id": duplicate_of,
            "input_disposition": disposition,
            "ml_pipeline_triggered": False,
        }

    try:
        raw_df, clean_df, _, _ = pipeline_module.prepare_dataframes(source_file)
        with open_database(database) as connection:
            with connection:
                insert_batch_record(
                    connection,
                    batch_id=batch_id,
                    source_file=source_file,
                    checksum=checksum,
                    status="success",
                    started_at=started_at,
                    completed_at=utc_now(),
                    raw_rows=len(raw_df),
                    clean_rows=len(clean_df),
                    fraud_count=int(clean_df["is_fraud"].sum()),
                )
                insert_dataframe(
                    connection,
                    table="raw_transactions",
                    frame=raw_df,
                    columns=list(RAW_COLUMN_TYPES),
                    batch_id=batch_id,
                    include_source_row=True,
                )
                insert_dataframe(
                    connection,
                    table="clean_transactions",
                    frame=clean_df,
                    columns=list(CLEAN_COLUMN_TYPES),
                    batch_id=batch_id,
                    include_source_row=False,
                )
    except Exception as exc:
        failed_path = failed_file_path(source_file, batch_id)
        if source_file.exists():
            source_file.replace(failed_path)
        with open_database(database) as connection:
            insert_batch_record(
                connection,
                batch_id=batch_id,
                source_file=failed_path,
                checksum=checksum,
                status="failed",
                started_at=started_at,
                completed_at=utc_now(),
                error_message=f"{type(exc).__name__}: {exc}",
                disposition=f"retained_as_{failed_path.name}",
            )
        return {
            "batch_id": batch_id,
            "status": "failed",
            "failed_file": str(failed_path.resolve()),
            "error_message": f"{type(exc).__name__}: {exc}",
            "ml_pipeline_triggered": False,
        }

    disposition = consume_ingested_file(source_file, database, batch_id, "after_ingestion")
    return {
        "batch_id": batch_id,
        "status": "success",
        "raw_row_count": int(len(raw_df)),
        "clean_row_count": int(len(clean_df)),
        "fraud_count": int(clean_df["is_fraud"].sum()),
        "input_disposition": disposition,
        "ml_pipeline_triggered": False,
    }


def run_once(
    root: Path = DEFAULT_DATA_ROOT,
    *,
    stability_seconds: int = 60,
) -> list[dict[str, Any]]:
    inbox, database = ensure_data_store(root.resolve())
    pipeline_module = load_data_pipeline_module()
    ready_files = [
        path for path in sorted(inbox.glob("*.csv")) if path.is_file() and is_stable(path, stability_seconds)
    ]
    return [process_file(path, database, pipeline_module) for path in ready_files]


def table_count(connection: sqlite3.Connection, table: str) -> int:
    allowed_tables = {"ingestion_batches", "raw_transactions", "clean_transactions"}
    if table not in allowed_tables:
        raise ValueError(f"Unknown table: {table}")
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def database_status(root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    _, database = ensure_data_store(root.resolve())
    with open_database(database) as connection:
        recent_batches: Iterable[tuple[Any, ...]] = connection.execute("""
            SELECT batch_id, source_file_name, status, raw_row_count, clean_row_count,
                   completed_at_utc, ml_pipeline_triggered
            FROM ingestion_batches
            ORDER BY started_at_utc DESC
            LIMIT 10
            """).fetchall()
        return {
            "database": str(database.resolve()),
            "batch_count": table_count(connection, "ingestion_batches"),
            "raw_row_count": table_count(connection, "raw_transactions"),
            "clean_row_count": table_count(connection, "clean_transactions"),
            "recent_batches": [
                {
                    "batch_id": row[0],
                    "source_file_name": row[1],
                    "status": row[2],
                    "raw_row_count": row[3],
                    "clean_row_count": row[4],
                    "completed_at_utc": row[5],
                    "ml_pipeline_triggered": bool(row[6]),
                }
                for row in recent_batches
            ],
        }


def print_status(status: dict[str, Any]) -> None:
    print("Database:", status["database"])
    print("Recorded batches:", status["batch_count"])
    print("Raw rows:", status["raw_row_count"])
    print("Clean rows:", status["clean_row_count"])
    if status["recent_batches"]:
        print("Recent batches:")
        for batch in status["recent_batches"]:
            print(
                f"- {batch['status']}: {batch['source_file_name']} "
                f"(raw={batch['raw_row_count']}, clean={batch['clean_row_count']}, "
                f"ml_triggered={batch['ml_pipeline_triggered']})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append stable CSV batches to the raw and clean data tables without training a model."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--stability-seconds",
        type=int,
        default=60,
        help="Minimum age of a CSV before ingestion; protects against partially copied files.",
    )
    parser.add_argument("--status", action="store_true", help="Show database row counts and recent batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status:
        print_status(database_status(args.root))
        return

    results = run_once(args.root, stability_seconds=args.stability_seconds)
    if not results:
        print("No stable, unprocessed CSV batches were found.")
        return
    for result in results:
        print(f"{result['status']}: {result['batch_id']}")


if __name__ == "__main__":
    main()
