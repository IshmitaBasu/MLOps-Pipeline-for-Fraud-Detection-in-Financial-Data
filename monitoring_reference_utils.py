"""Training-only preparation and lineage for a versioned monitoring reference."""

# %% Imports and fixed feature contract
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frozen_training_data import FULL_ROWS, SMOKE_ROWS, load_training_data
from model_family_comparison_utils import FROZEN_FULL_SPLIT_HASH
from oversampling_experiment_utils import MODEL_FEATURES
from predictive_signal_utils import add_training_history
from sender_location_history_utils import HISTORY_FEATURES


# %% Frozen training reconstruction followed by label-independent history
def prepare_reference_history(
    raw_path: Path,
    mode: str,
    *,
    expected_hash: str = FROZEN_FULL_SPLIT_HASH,
    expected_rows: int = FULL_ROWS,
    smoke_rows: int = SMOKE_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    training, lineage = load_training_data(
        raw_path, mode, expected_hash=expected_hash, expected_rows=expected_rows, smoke_rows=smoke_rows
    )
    labels_before = training["is_fraud"].copy()
    prepared = add_training_history(training)
    if not prepared["is_fraud"].equals(labels_before):
        raise ValueError("History preparation changed target labels.")
    if not set(MODEL_FEATURES).issubset(prepared.columns):
        raise ValueError("Prepared training data does not satisfy the frozen model feature contract.")
    if (prepared[HISTORY_FEATURES[1]] > prepared[HISTORY_FEATURES[0]]).any():
        raise ValueError("Sender-location history cannot exceed sender history.")
    lineage.update(
        {
            "feature_set": "original_v1_plus_fraud_behaviour_features_v2",
            "model_features": MODEL_FEATURES,
            "history_features": HISTORY_FEATURES,
            "history_definition": "strictly_earlier_training_events_only",
            "target_used_to_construct_history": False,
            "validation_events_used_to_construct_history": False,
            "test_events_used_to_construct_history": False,
        }
    )
    return prepared, lineage


# %% Aggregate receipt; no transaction rows or identifiers are exported
def history_preparation_receipt(prepared: pd.DataFrame, lineage: dict[str, Any]) -> dict[str, Any]:
    history = prepared[HISTORY_FEATURES].astype("int64")
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(history, index=True).to_numpy(dtype="uint64").tobytes()
    ).hexdigest()
    return {
        "status": "history_prepared",
        "scope": lineage["scope"],
        "evidence_scope": lineage["evidence_scope"],
        "training_rows": int(len(prepared)),
        "training_fraud_rows": int(prepared["is_fraud"].sum()),
        "history_sha256": digest,
        "sender_prior_count_max": int(history[HISTORY_FEATURES[0]].max()),
        "sender_location_prior_count_max": int(history[HISTORY_FEATURES[1]].max()),
        "new_location_rows": int(history[HISTORY_FEATURES[2]].sum()),
        "model_scored": False,
        "model_fitted": False,
        "reference_profile_created": False,
        "validation_features_retained": False,
        "test_features_retained": False,
        "transaction_rows_exported": False,
        "automatic_retraining": False,
    }
