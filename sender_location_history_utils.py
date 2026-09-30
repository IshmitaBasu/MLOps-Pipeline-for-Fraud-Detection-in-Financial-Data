"""Past-only feature and evaluation helpers for the Stage 08 experiment."""

# %% Imports and constants
from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from predictive_quality_utils import ID_COLUMN, TEST_CODE, TIME_COLUMN, TRAIN_CODE, VALIDATION_CODE

SENDER_COLUMN: Final = "sender_account"
LOCATION_COLUMN: Final = "location"
HISTORY_FEATURES: Final = [
    "sender_prior_transaction_count",
    "sender_location_prior_transaction_count",
    "is_new_location_for_returning_sender",
]
DEFAULT_ALERT_RATES: Final = (0.01, 0.05, 0.10)


# %% Shared development-frame construction
def restore_raw_source_order(modeling_table: pd.DataFrame, raw_data_path: Path) -> pd.DataFrame:
    """Restore the verified numeric transaction-ID order used by the frozen split."""

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")
    if modeling_table[ID_COLUMN].duplicated().any():
        raise ValueError("The modeling table contains duplicate transaction IDs.")
    identifiers = modeling_table[ID_COLUMN].astype("string")
    if not identifiers.str.fullmatch(r"T\d+").all():
        raise ValueError("Canonical transaction IDs must use the T<number> format.")
    numeric_order = identifiers.str.slice(1).astype("int64")
    if numeric_order.duplicated().any():
        raise ValueError("Numeric transaction-ID suffixes must be unique.")
    return (
        modeling_table.assign(_raw_source_order=numeric_order.to_numpy())
        .sort_values("_raw_source_order", kind="mergesort")
        .drop(columns="_raw_source_order")
        .reset_index(drop=True)
    )


def attach_sender_account(
    development_data: pd.DataFrame, raw_data_path: Path, chunk_rows: int = 500_000
) -> pd.DataFrame:
    """Attach sender keys only to already-filtered train/validation rows."""

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")
    required_ids = pd.Index(development_data[ID_COLUMN].astype("string"))
    key_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_data_path,
        usecols=[ID_COLUMN, SENDER_COLUMN],
        dtype={ID_COLUMN: "string", SENDER_COLUMN: "string"},
        chunksize=chunk_rows,
    ):
        selected = chunk.loc[chunk[ID_COLUMN].isin(required_ids)]
        if not selected.empty:
            key_parts.append(selected)
    if not key_parts:
        raise ValueError("No sender keys matched the development transactions.")
    sender_keys = pd.concat(key_parts, ignore_index=True)
    if sender_keys[ID_COLUMN].duplicated().any():
        raise ValueError("The raw data contains duplicate development transaction IDs.")
    original_ids = development_data[ID_COLUMN].astype("string").reset_index(drop=True)
    enriched = development_data.merge(sender_keys, on=ID_COLUMN, how="left", sort=False, validate="one_to_one")
    if not enriched[ID_COLUMN].astype("string").reset_index(drop=True).equals(original_ids):
        raise ValueError("Joining sender keys changed the development row order.")
    if enriched[SENDER_COLUMN].isna().any():
        raise ValueError("One or more development rows have no sender key.")
    return enriched


def build_development_frames(
    modeling_table: pd.DataFrame, partition: np.ndarray, raw_data_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/validation features after removing test rows and keys."""

    if len(modeling_table) != len(partition):
        raise ValueError("Modeling table and partition lengths do not match.")
    observed = set(np.unique(partition).tolist())
    if observed != {int(TRAIN_CODE), int(VALIDATION_CODE), int(TEST_CODE)}:
        raise ValueError("The frozen train, validation, and test codes are required.")
    development_mask = partition != TEST_CODE
    development_partition = np.asarray(partition)[development_mask]
    development_data = modeling_table.loc[development_mask].reset_index(drop=True)
    development_data = attach_sender_account(development_data, raw_data_path)
    development_data = add_sender_location_history_features(development_data, development_partition)
    train_df = development_data.loc[development_partition == TRAIN_CODE].reset_index(drop=True)
    validation_df = development_data.loc[development_partition == VALIDATION_CODE].reset_index(drop=True)
    if train_df.empty or validation_df.empty:
        raise ValueError("Training and validation frames must both be non-empty.")
    return train_df, validation_df


# %% Point-in-time history construction
def _strictly_earlier_training_count(ordered: pd.DataFrame, entity_columns: Sequence[str]) -> pd.Series:
    """Count earlier training events without using validation or equal-time rows."""
    entity_columns = list(entity_columns)
    entity_cumulative = ordered.groupby(entity_columns, dropna=False, observed=True, sort=False)[
        "_is_training_event"
    ].cumsum()
    same_time_cumulative = ordered.groupby([*entity_columns, TIME_COLUMN], dropna=False, observed=True, sort=False)[
        "_is_training_event"
    ].cumsum()
    return entity_cumulative - same_time_cumulative


def add_sender_location_history_features(
    development_data: pd.DataFrame, development_partition: np.ndarray
) -> pd.DataFrame:
    """Add history features to train/validation rows using training history only.

    ``development_partition`` may contain train and validation codes only. This
    prevents the test partition from entering the feature calculation API.
    """
    required = {ID_COLUMN, TIME_COLUMN, SENDER_COLUMN, LOCATION_COLUMN}
    missing = required.difference(development_data.columns)
    if missing:
        raise ValueError(f"Missing history-feature columns: {sorted(missing)}")
    if len(development_data) != len(development_partition):
        raise ValueError("Development data and partition lengths do not match.")
    observed_codes = set(np.unique(development_partition).tolist())
    allowed_codes = {int(TRAIN_CODE), int(VALIDATION_CODE)}
    if not observed_codes.issubset(allowed_codes):
        raise ValueError("History feature construction accepts training and validation rows only.")
    if observed_codes != allowed_codes:
        raise ValueError("Both training and validation rows are required.")

    ordered = development_data[[ID_COLUMN, TIME_COLUMN, SENDER_COLUMN, LOCATION_COLUMN]].copy()
    ordered["_is_training_event"] = (np.asarray(development_partition) == TRAIN_CODE).astype("int8")
    ordered = ordered.sort_values([TIME_COLUMN, ID_COLUMN], kind="mergesort", na_position="first")

    sender_prior = _strictly_earlier_training_count(ordered, [SENDER_COLUMN])
    sender_location_prior = _strictly_earlier_training_count(ordered, [SENDER_COLUMN, LOCATION_COLUMN])

    features = pd.DataFrame(index=ordered.index)
    features[HISTORY_FEATURES[0]] = sender_prior.astype("int32")
    features[HISTORY_FEATURES[1]] = sender_location_prior.astype("int32")
    features[HISTORY_FEATURES[2]] = ((sender_prior > 0) & (sender_location_prior == 0)).astype("int8")

    enriched = development_data.copy()
    for feature in HISTORY_FEATURES:
        enriched[feature] = features[feature].reindex(enriched.index)
    validate_history_features(enriched)
    return enriched


def validate_history_features(data: pd.DataFrame) -> None:
    """Validate feature ranges and the definition of the new-location flag."""
    missing = set(HISTORY_FEATURES).difference(data.columns)
    if missing:
        raise ValueError(f"Missing calculated history features: {sorted(missing)}")
    if data[HISTORY_FEATURES].isna().any().any():
        raise ValueError("History features contain missing values.")
    if (data[HISTORY_FEATURES[:2]] < 0).any().any():
        raise ValueError("History counts cannot be negative.")
    if not data[HISTORY_FEATURES[2]].isin([0, 1]).all():
        raise ValueError("The new-location feature must be binary.")

    expected_flag = ((data[HISTORY_FEATURES[0]] > 0) & (data[HISTORY_FEATURES[1]] == 0)).astype("int8")
    if not expected_flag.equals(data[HISTORY_FEATURES[2]].astype("int8")):
        raise ValueError("The new-location feature does not match its definition.")


# %% Equal-workload validation evaluation
def fixed_workload_table(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    transaction_amount: pd.Series | np.ndarray,
    alert_rates: Sequence[float] = DEFAULT_ALERT_RATES,
) -> pd.DataFrame:
    """Evaluate count and fraudulent-value capture for identical alert volumes."""
    labels = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(y_score, dtype=float)
    amounts = np.asarray(transaction_amount, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or amounts.ndim != 1:
        raise ValueError("Labels, scores, and amounts must be one-dimensional.")
    if not (len(labels) == len(scores) == len(amounts)) or len(labels) == 0:
        raise ValueError("Labels, scores, and amounts must have equal non-zero lengths.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Transaction amounts must be finite and non-negative.")

    total_fraud_count = int(labels.sum())
    total_fraud_value = float(amounts[labels == 1].sum())
    if total_fraud_count == 0:
        raise ValueError("At least one fraud case is required for evaluation.")

    ranked_positions = np.argsort(-scores, kind="mergesort")
    rows = []
    for requested_rate in alert_rates:
        if not 0 < requested_rate <= 1:
            raise ValueError("Alert rates must be greater than zero and at most one.")
        alert_count = max(1, int(np.floor(len(labels) * requested_rate)))
        selected = ranked_positions[:alert_count]
        selected_labels = labels[selected]
        fraud_captured = int(selected_labels.sum())
        fraudulent_value_captured = float(amounts[selected][selected_labels == 1].sum())
        rows.append(
            {
                "requested_alert_rate": float(requested_rate),
                "alert_count": alert_count,
                "actual_alert_rate": alert_count / len(labels),
                "fraud_cases_captured": fraud_captured,
                "legitimate_alerts": alert_count - fraud_captured,
                "precision": fraud_captured / alert_count,
                "fraud_count_recall": fraud_captured / total_fraud_count,
                "fraudulent_value_captured": fraudulent_value_captured,
                "fraud_value_recall": (fraudulent_value_captured / total_fraud_value if total_fraud_value > 0 else 0.0),
            }
        )
    return pd.DataFrame(rows)


def candidate_progression_decision(
    control_pr_auc: float,
    candidate_pr_auc: float,
    control_workload: pd.DataFrame,
    candidate_workload: pd.DataFrame,
    comparison_alert_rate: float = 0.05,
) -> dict[str, object]:
    """Apply the predeclared validation rule without consulting test evidence."""
    control_row = control_workload.loc[np.isclose(control_workload["requested_alert_rate"], comparison_alert_rate)]
    candidate_row = candidate_workload.loc[
        np.isclose(candidate_workload["requested_alert_rate"], comparison_alert_rate)
    ]
    if len(control_row) != 1 or len(candidate_row) != 1:
        raise ValueError("Each workload table must contain the comparison alert rate once.")
    control = control_row.iloc[0]
    candidate = candidate_row.iloc[0]

    pr_auc_improved = candidate_pr_auc > control_pr_auc
    count_not_worse = candidate["fraud_count_recall"] >= control["fraud_count_recall"]
    value_not_worse = candidate["fraud_value_recall"] >= control["fraud_value_recall"]
    operational_improvement = (
        candidate["fraud_count_recall"] > control["fraud_count_recall"]
        or candidate["fraud_value_recall"] > control["fraud_value_recall"]
    )
    progresses = bool(pr_auc_improved and count_not_worse and value_not_worse and operational_improvement)
    return {
        "selection_data": "validation_only",
        "comparison_alert_rate": comparison_alert_rate,
        "control_validation_pr_auc": float(control_pr_auc),
        "candidate_validation_pr_auc": float(candidate_pr_auc),
        "absolute_pr_auc_change": float(candidate_pr_auc - control_pr_auc),
        "relative_pr_auc_change": float(
            (candidate_pr_auc - control_pr_auc) / control_pr_auc if control_pr_auc else 0.0
        ),
        "control_fraud_count_recall": float(control["fraud_count_recall"]),
        "candidate_fraud_count_recall": float(candidate["fraud_count_recall"]),
        "control_fraud_value_recall": float(control["fraud_value_recall"]),
        "candidate_fraud_value_recall": float(candidate["fraud_value_recall"]),
        "pr_auc_improved": bool(pr_auc_improved),
        "fraud_count_recall_not_worse": bool(count_not_worse),
        "fraud_value_recall_not_worse": bool(value_not_worse),
        "operational_measure_improved": bool(operational_improvement),
        "candidate_progresses": progresses,
        "recommendation": (
            "retain_sender_location_history_for_later_model_experiments"
            if progresses
            else "retain_original_v1_and_do_not_progress_this_feature_group"
        ),
        "test_split_evaluated": False,
    }
