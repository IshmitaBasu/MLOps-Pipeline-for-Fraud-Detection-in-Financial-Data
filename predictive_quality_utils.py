"""Utilities for Stage 07 predictive-quality diagnostics.

The functions in this module deliberately operate on the training partition only.
They test whether past-only behavioural features are promising before those
features are added to model experiments.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# %% Constants and result containers
TARGET_COLUMN: Final = "is_fraud"
ID_COLUMN: Final = "transaction_id"
TIME_COLUMN: Final = "event_timestamp"

TRAIN_CODE: Final = np.uint8(0)
VALIDATION_CODE: Final = np.uint8(1)
TEST_CODE: Final = np.uint8(2)

SPLIT_NAMES: Final = {
    int(TRAIN_CODE): "train",
    int(VALIDATION_CODE): "validation",
    int(TEST_CODE): "test",
}

ENTITY_SPECS: Final = {
    "sender_account": ("sender_account",),
    "receiver_account": ("receiver_account",),
    "device_hash": ("device_hash",),
    "ip_address": ("ip_address",),
    "sender_receiver_pair": ("sender_account", "receiver_account"),
    "sender_device_pair": ("sender_account", "device_hash"),
    "sender_ip_pair": ("sender_account", "ip_address"),
    "sender_location_pair": ("sender_account", "location"),
}


@dataclass(frozen=True)
class DiagnosticResult:
    """Tables and recommendation produced by the training-only diagnostic."""

    entity_repetition_summary: pd.DataFrame
    new_vs_returning_fraud_rates: pd.DataFrame
    sender_amount_deviation_summary: pd.DataFrame
    feature_feasibility_decision: dict[str, object]


# %% Random split construction and audit information
def make_stratified_random_partition(
    target: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Create deterministic 70/15/15-style split codes with stratification."""
    if target.isna().any():
        raise ValueError("The target contains missing values.")
    if target.nunique() < 2:
        raise ValueError("A stratified split requires both target classes.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError(
            "validation_fraction must be positive and leave room for a test split."
        )

    total_rows = len(target)
    train_rows = int(round(total_rows * train_fraction))
    validation_rows = int(round(total_rows * validation_fraction))
    test_rows = total_rows - train_rows - validation_rows
    if min(train_rows, validation_rows, test_rows) <= 0:
        raise ValueError("The requested fractions create an empty split.")

    row_positions = np.arange(total_rows, dtype=np.int64)
    train_positions, remaining_positions = train_test_split(
        row_positions,
        train_size=train_rows,
        random_state=seed,
        stratify=target.to_numpy(),
    )
    validation_positions, test_positions = train_test_split(
        remaining_positions,
        train_size=validation_rows,
        random_state=seed,
        stratify=target.iloc[remaining_positions].to_numpy(),
    )

    partition = np.full(len(target), TEST_CODE, dtype=np.uint8)
    partition[train_positions] = TRAIN_CODE
    partition[validation_positions] = VALIDATION_CODE
    partition[test_positions] = TEST_CODE
    validate_partition(partition)
    return partition


def validate_partition(partition: np.ndarray) -> None:
    """Reject malformed split assignments before any diagnostic is run."""
    if partition.ndim != 1:
        raise ValueError("The partition must be a one-dimensional array.")
    observed = set(np.unique(partition).tolist())
    expected = set(SPLIT_NAMES)
    if observed != expected:
        raise ValueError(f"Expected split codes {expected}, but found {observed}.")


def stable_split_hash(transaction_ids: pd.Series, partition: np.ndarray) -> str:
    """Return an auditable hash of transaction IDs and their split assignments."""
    import hashlib

    if len(transaction_ids) != len(partition):
        raise ValueError("Transaction IDs and partition lengths do not match.")
    transaction_hashes = pd.util.hash_pandas_object(
        transaction_ids.astype("string"), index=False
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(transaction_hashes.tobytes())
    digest.update(np.asarray(partition, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def build_split_manifest(
    data: pd.DataFrame,
    partition: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, object]:
    """Describe the frozen split without evaluating validation or test features."""
    required = {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing split-manifest columns: {sorted(missing)}")
    if len(data) != len(partition):
        raise ValueError("Data and partition lengths do not match.")

    split_rows: dict[str, object] = {}
    for code, name in SPLIT_NAMES.items():
        mask = partition == code
        timestamps = data.loc[mask, TIME_COLUMN]
        split_rows[name] = {
            "rows": int(mask.sum()),
            "fraud_rows": int(data.loc[mask, TARGET_COLUMN].sum()),
            "fraud_rate": float(data.loc[mask, TARGET_COLUMN].mean()),
            "minimum_timestamp": timestamps.min().isoformat(),
            "maximum_timestamp": timestamps.max().isoformat(),
        }

    return {
        "split_method": "stratified_random",
        "seed": seed,
        "requested_train_fraction": train_fraction,
        "requested_validation_fraction": validation_fraction,
        "requested_test_fraction": round(
            1 - train_fraction - validation_fraction, 12
        ),
        "split_assignment_sha256": stable_split_hash(data[ID_COLUMN], partition),
        "splits": split_rows,
        "test_split_evaluated": False,
    }


def training_partition(data: pd.DataFrame, partition: np.ndarray) -> pd.DataFrame:
    """Materialise only the training rows for exploratory diagnostics."""
    if len(data) != len(partition):
        raise ValueError("Data and partition lengths do not match.")
    return data.loc[partition == TRAIN_CODE].copy()


# %% Past-only behavioural feature calculations
def past_transaction_count(
    data: pd.DataFrame,
    entity_columns: tuple[str, ...],
) -> pd.Series:
    """Count strictly earlier rows for an entity; the current row is excluded."""
    required = {ID_COLUMN, TIME_COLUMN, *entity_columns}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")

    ordered = data.sort_values(
        [TIME_COLUMN, ID_COLUMN], kind="mergesort", na_position="first"
    )
    entity_position = ordered.groupby(
        list(entity_columns), dropna=False, observed=True, sort=False
    ).cumcount()
    timestamp_position = ordered.groupby(
        [*entity_columns, TIME_COLUMN],
        dropna=False,
        observed=True,
        sort=False,
    ).cumcount()
    counts = entity_position - timestamp_position
    return counts.reindex(data.index).astype("int64")


def entity_history_diagnostic(
    training_data: pd.DataFrame,
    entity_name: str,
    entity_columns: tuple[str, ...],
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    """Measure entity repetition and fraud-rate separation on training rows."""
    prior_count = past_transaction_count(training_data, entity_columns)
    returning = prior_count > 0

    entity_counts = training_data.groupby(
        list(entity_columns), dropna=False, observed=True
    ).size()
    total_rows = len(training_data)
    repeated_rows = int((entity_counts[entity_counts > 1] - 1).sum())
    history_coverage = float(returning.mean())

    repetition_row = {
        "entity": entity_name,
        "entity_columns": "+".join(entity_columns),
        "unique_entities": int(len(entity_counts)),
        "maximum_transactions_per_entity": int(entity_counts.max()),
        "rows_with_prior_history": int(returning.sum()),
        "rows_without_prior_history": int((~returning).sum()),
        "prior_history_coverage": history_coverage,
        "repeated_rows_cross_check": repeated_rows,
    }

    status = np.where(returning, "returning", "new")
    association_rows = []
    status_results: dict[str, dict[str, object]] = {}
    for history_status in ("new", "returning"):
        mask = status == history_status
        fraud_rows = int(training_data.loc[mask, TARGET_COLUMN].sum())
        rows = int(mask.sum())
        fraud_rate = (
            float(training_data.loc[mask, TARGET_COLUMN].mean())
            if rows
            else None
        )
        status_results[history_status] = {
            "rows": rows,
            "fraud_rows": fraud_rows,
            "fraud_rate": fraud_rate,
        }
        association_rows.append(
            {
                "entity": entity_name,
                "history_status": history_status,
                "rows": rows,
                "fraud_rows": fraud_rows,
                "fraud_rate": fraud_rate,
            }
        )

    new_rate = status_results["new"]["fraud_rate"]
    returning_rate = status_results["returning"]["fraud_rate"]
    if new_rate is None or returning_rate is None:
        relative_rate_difference = None
    else:
        denominator = max(new_rate, returning_rate, np.finfo(float).eps)
        relative_rate_difference = abs(new_rate - returning_rate) / denominator
    sufficient_fraud_evidence = (
        int(status_results["new"]["fraud_rows"]) >= 100
        and int(status_results["returning"]["fraud_rows"]) >= 100
    )
    passes_initial_gate = (
        history_coverage >= 0.05
        and sufficient_fraud_evidence
        and relative_rate_difference is not None
        and relative_rate_difference >= 0.10
    )
    decision = {
        "entity": entity_name,
        "history_coverage": history_coverage,
        "relative_fraud_rate_difference": (
            float(relative_rate_difference)
            if relative_rate_difference is not None
            else None
        ),
        "sufficient_fraud_evidence": sufficient_fraud_evidence,
        "passes_initial_screening_gate": bool(passes_initial_gate),
    }
    return repetition_row, pd.DataFrame(association_rows), decision


def sender_amount_deviation_diagnostic(training_data: pd.DataFrame) -> pd.DataFrame:
    """Relate current amount to the sender's strictly earlier mean amount."""
    required = {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, "sender_account", "amount"}
    missing = required.difference(training_data.columns)
    if missing:
        raise ValueError(f"Missing amount-deviation columns: {sorted(missing)}")

    ordered = training_data.sort_values(
        [TIME_COLUMN, ID_COLUMN], kind="mergesort", na_position="first"
    ).copy()
    grouped = ordered.groupby(
        "sender_account", dropna=False, observed=True, sort=False
    )
    timestamp_grouped = ordered.groupby(
        ["sender_account", TIME_COLUMN],
        dropna=False,
        observed=True,
        sort=False,
    )
    prior_count = grouped.cumcount() - timestamp_grouped.cumcount()
    prior_amount = grouped["amount"].cumsum() - timestamp_grouped["amount"].cumsum()
    prior_mean = prior_amount / prior_count.replace(0, np.nan)
    relative_amount = ordered["amount"] / prior_mean.replace(0, np.nan)

    valid = relative_amount.replace([np.inf, -np.inf], np.nan).notna()
    analysis = ordered.loc[valid, [TARGET_COLUMN]].copy()
    analysis["relative_amount"] = relative_amount.loc[valid]
    if analysis.empty:
        return pd.DataFrame(
            columns=["relative_amount_band", "rows", "fraud_rows", "fraud_rate"]
        )

    quantile_count = min(10, int(analysis["relative_amount"].nunique()))
    analysis["relative_amount_band"] = pd.qcut(
        analysis["relative_amount"],
        q=quantile_count,
        duplicates="drop",
    ).astype("string")
    return (
        analysis.groupby("relative_amount_band", observed=True, sort=False)
        .agg(
            rows=(TARGET_COLUMN, "size"),
            fraud_rows=(TARGET_COLUMN, "sum"),
            fraud_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )


# %% Training-only diagnostic entry point
def run_training_diagnostics(training_data: pd.DataFrame) -> DiagnosticResult:
    """Run all pre-model feature checks using training rows only."""
    required = {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, "amount"}
    for entity_columns in ENTITY_SPECS.values():
        required.update(entity_columns)
    missing = required.difference(training_data.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")
    if training_data.empty:
        raise ValueError("The training partition is empty.")

    repetition_rows = []
    association_tables = []
    entity_decisions = []
    for entity_name, entity_columns in ENTITY_SPECS.items():
        repetition, associations, decision = entity_history_diagnostic(
            training_data, entity_name, entity_columns
        )
        repetition_rows.append(repetition)
        association_tables.append(associations)
        entity_decisions.append(decision)

    promising_entities = [
        item["entity"]
        for item in entity_decisions
        if item["passes_initial_screening_gate"]
    ]
    feasibility_decision = {
        "scope": "training_partition_only",
        "target_used_to_construct_history_features": False,
        "history_definition": "strictly earlier transaction count",
        "screening_thresholds": {
            "minimum_prior_history_coverage": 0.05,
            "minimum_fraud_rows_per_history_group": 100,
            "minimum_relative_fraud_rate_difference": 0.10,
        },
        "entity_decisions": entity_decisions,
        "promising_entities": promising_entities,
        "recommendation": (
            "engineer_and_validate_selected_history_features"
            if promising_entities
            else "do_not_prioritise_history_features_without_more_evidence"
        ),
    }

    return DiagnosticResult(
        entity_repetition_summary=pd.DataFrame(repetition_rows),
        new_vs_returning_fraud_rates=pd.concat(
            association_tables, ignore_index=True
        ),
        sender_amount_deviation_summary=sender_amount_deviation_diagnostic(
            training_data
        ),
        feature_feasibility_decision=feasibility_decision,
    )
