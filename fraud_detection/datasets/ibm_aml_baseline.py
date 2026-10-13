"""Pure preparation, modelling and validation helpers for the Stage 21 IBM AML baseline."""

# %% Imports and frozen protocol
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Sequence

import lightgbm
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_detection.common.modeling import best_f1_threshold, classification_metrics, predict_scores
from fraud_detection.datasets.ibm_aml_feasibility import (
    CANONICAL_COLUMNS,
    EXPECTED_SHA256,
    FINGERPRINT_KEY_A,
    FINGERPRINT_KEY_B,
    PRIMARY_PERIOD_END_EXCLUSIVE,
    iter_valid_chunks,
)
from fraud_detection.evaluation.validation import positive_class_weight

TARGET_COLUMN: Final = "is_laundering"
TIME_COLUMN: Final = "event_timestamp"
ID_COLUMN: Final = "transaction_key_hash"
SOURCE_ROW_COLUMN: Final = "source_row_number"
AMOUNT_COLUMN: Final = "amount_paid"
CURRENCY_COLUMN: Final = "payment_currency"

CATEGORICAL_FEATURES: Final = [
    "from_bank",
    "to_bank",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]
NUMERIC_FEATURES: Final = [
    "log_amount_received",
    "log_amount_paid",
    "event_hour",
    "event_day_of_week",
    "same_bank",
    "same_currency",
    "equal_amount",
]
FEATURE_COLUMNS: Final = CATEGORICAL_FEATURES + NUMERIC_FEATURES
PROHIBITED_MODEL_FIELDS: Final = {
    TARGET_COLUMN,
    "from_account",
    "to_account",
    SOURCE_ROW_COLUMN,
    ID_COLUMN,
    TIME_COLUMN,
    "post_primary",
    "laundering_pattern",
}

DEFAULT_RANDOM_STATE: Final = 42
TRAIN_FRACTION: Final = 0.70
VALIDATION_FRACTION: Final = 0.15
TEST_FRACTION: Final = 0.15
SMOKE_ROWS: Final = 500_000
DEFAULT_CHUNK_ROWS: Final = 200_000
REVIEW_RATES: Final = (0.001, 0.005, 0.01)
MINIMUM_CURRENCY_POSITIVES: Final = 20
MINIMUM_STABILITY_POSITIVES: Final = 20
MINIMUM_AP_LIFT: Final = 5.0
MINIMUM_STABILITY_AP_LIFT: Final = 2.0
MINIMUM_TOP_1PCT_RECALL: Final = 0.25
EXPECTED_FULL_ROWS: Final = 5_078_345
EXPECTED_PRIMARY_ROWS: Final = 5_077_237
EXPECTED_PRIMARY_POSITIVES: Final = 4_522

LIGHTGBM_PARAMETERS: Final = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
MODEL_SERIALIZATION_FORMAT: Final = "cloudpickle"


# %% Result containers
@dataclass(frozen=True)
class DevelopmentSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    metrics: dict[str, Any]
    workloads: pd.DataFrame
    currency_capture: pd.DataFrame
    stability: pd.DataFrame
    scores: np.ndarray
    score_type: str


# %% Deterministic identifiers and source preparation
def deterministic_transaction_hash(source_rows: pd.Series | np.ndarray, source_sha256: str) -> np.ndarray:
    """Return stable uint64 identifiers derived from source identity and row number."""

    if len(source_sha256) != 64:
        raise ValueError("A 64-character source SHA-256 is required.")
    rows = np.asarray(source_rows, dtype=np.uint64)
    if rows.ndim != 1 or (rows == 0).any():
        raise ValueError("Source row numbers must be a one-dimensional positive array.")
    seed = np.uint64(int(source_sha256[:16], 16))
    with np.errstate(over="ignore"):
        values = rows + seed + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    return values


def _derived_chunk(raw: pd.DataFrame, source_rows: np.ndarray) -> tuple[pd.DataFrame, dict[str, int]]:
    timestamps = pd.to_datetime(raw["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    labels = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
    received = pd.to_numeric(raw["amount_received"], errors="coerce")
    paid = pd.to_numeric(raw["amount_paid"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Stage 21 found an invalid timestamp after the clean Stage 20 receipt.")
    if not labels.isin([0, 1]).all():
        raise ValueError("Stage 21 found a non-binary label after the clean Stage 20 receipt.")
    if received.isna().any() or paid.isna().any() or (received < 0).any() or (paid < 0).any():
        raise ValueError("Stage 21 requires finite, non-negative received and paid amounts.")

    primary = timestamps < PRIMARY_PERIOD_END_EXCLUSIVE
    selected = raw.loc[primary]
    selected_received = received.loc[primary]
    selected_paid = paid.loc[primary]
    selected_timestamps = timestamps.loc[primary]
    selected_source_rows = source_rows[np.asarray(primary)]
    text = selected.astype("string").fillna("")
    hash_a = pd.util.hash_pandas_object(text, index=False, hash_key=FINGERPRINT_KEY_A).to_numpy(dtype=np.uint64)
    hash_b = pd.util.hash_pandas_object(text, index=False, hash_key=FINGERPRINT_KEY_B).to_numpy(dtype=np.uint64)

    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: selected_source_rows.astype(np.int64),
            TIME_COLUMN: selected_timestamps.to_numpy(),
            "from_bank": selected["from_bank"].astype("string").str.strip().to_numpy(),
            "to_bank": selected["to_bank"].astype("string").str.strip().to_numpy(),
            "receiving_currency": selected["receiving_currency"].astype("string").str.strip().to_numpy(),
            CURRENCY_COLUMN: selected[CURRENCY_COLUMN].astype("string").str.strip().to_numpy(),
            "payment_format": selected["payment_format"].astype("string").str.strip().to_numpy(),
            AMOUNT_COLUMN: selected_paid.to_numpy(dtype=np.float64),
            "log_amount_received": np.log1p(selected_received.to_numpy(dtype=np.float64)),
            "log_amount_paid": np.log1p(selected_paid.to_numpy(dtype=np.float64)),
            "event_hour": selected_timestamps.dt.hour.to_numpy(dtype=np.int8),
            "event_day_of_week": selected_timestamps.dt.dayofweek.to_numpy(dtype=np.int8),
            "same_bank": selected["from_bank"]
            .astype("string")
            .str.strip()
            .eq(selected["to_bank"].astype("string").str.strip())
            .to_numpy(dtype=np.int8),
            "same_currency": selected["receiving_currency"]
            .astype("string")
            .str.strip()
            .eq(selected[CURRENCY_COLUMN].astype("string").str.strip())
            .to_numpy(dtype=np.int8),
            "equal_amount": selected_received.eq(selected_paid).to_numpy(dtype=np.int8),
            TARGET_COLUMN: labels.loc[primary].to_numpy(dtype=np.int8),
            "_row_hash_a": hash_a,
            "_row_hash_b": hash_b,
        }
    )
    return frame, {"primary_rows": int(primary.sum()), "primary_positives": int(labels.loc[primary].sum())}


def prepare_modeling_population(
    path: Path,
    *,
    source_sha256: str,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the fixed primary period in chunks and return no account identifiers."""

    source_sha256 = source_sha256.upper()
    if source_sha256 != EXPECTED_SHA256:
        raise ValueError("The Stage 21 source checksum does not match the Stage 20 candidate receipt.")
    chunks, structural = iter_valid_chunks(path, chunk_rows=chunk_rows, row_limit=row_limit)
    prepared: list[pd.DataFrame] = []
    source_offset = 0
    primary_rows = 0
    primary_positives = 0
    for raw in chunks:
        source_rows = np.arange(source_offset + 1, source_offset + len(raw) + 1, dtype=np.int64)
        derived, counts = _derived_chunk(raw, source_rows)
        prepared.append(derived)
        source_offset += len(raw)
        primary_rows += counts["primary_rows"]
        primary_positives += counts["primary_positives"]
    if not prepared:
        raise ValueError("Stage 21 found no rows in the fixed primary period.")
    if structural["malformed_structure_rows"]:
        raise ValueError("Stage 21 refuses to continue with malformed source rows.")

    population = pd.concat(prepared, ignore_index=True)
    duplicate_mask = population.duplicated(["_row_hash_a", "_row_hash_b"], keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    duplicate_positives_removed = int(population.loc[duplicate_mask, TARGET_COLUMN].sum())
    population = population.loc[~duplicate_mask].drop(columns=["_row_hash_a", "_row_hash_b"])
    population[ID_COLUMN] = deterministic_transaction_hash(population[SOURCE_ROW_COLUMN], source_sha256)
    if population[ID_COLUMN].duplicated().any():
        raise ValueError("The deterministic transaction hash unexpectedly contains a collision.")
    population = population.sort_values([TIME_COLUMN, SOURCE_ROW_COLUMN], kind="mergesort").reset_index(drop=True)
    for column in CATEGORICAL_FEATURES:
        population[column] = population[column].astype("category")

    full_data = row_limit is None
    if full_data:
        observed = (int(structural["physical_data_rows_seen"]), primary_rows, primary_positives)
        expected = (EXPECTED_FULL_ROWS, EXPECTED_PRIMARY_ROWS, EXPECTED_PRIMARY_POSITIVES)
        if observed != expected:
            raise ValueError(f"Stage 21 population receipt mismatch. Expected {expected}, observed {observed}.")
    receipt = {
        **structural,
        "source_sha256": source_sha256,
        "primary_period_end_exclusive": PRIMARY_PERIOD_END_EXCLUSIVE.isoformat(),
        "pre_deduplication_primary_rows": primary_rows,
        "pre_deduplication_primary_positives": primary_positives,
        "duplicates_removed": duplicates_removed,
        "duplicate_positives_removed": duplicate_positives_removed,
        "modeling_rows": int(len(population)),
        "modeling_positives": int(population[TARGET_COLUMN].sum()),
        "account_identifiers_retained": False,
        "raw_tail_rows_retained": False,
    }
    return population, receipt


# %% Frozen split and test-isolation receipt
def _split_receipt(transaction_hashes: np.ndarray, assignments: np.ndarray, source_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(source_sha256))
    digest.update(np.asarray(transaction_hashes, dtype="<u8").tobytes())
    digest.update(np.asarray(assignments, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def make_development_split(
    population: pd.DataFrame, *, source_sha256: str, random_state: int = DEFAULT_RANDOM_STATE
) -> DevelopmentSplit:
    """Materialise train/validation only; the held-out test is represented by a count and receipt."""

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN, TIME_COLUMN, AMOUNT_COLUMN, CURRENCY_COLUMN])
    missing = sorted(required.difference(population.columns))
    if missing:
        raise ValueError(f"The Stage 21 population is missing required fields: {missing}")
    if population[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The Stage 21 population requires both classes.")
    if int(population[TARGET_COLUMN].sum()) < 7:
        raise ValueError("At least seven positive rows are required to stratify train, validation and test safely.")
    positions = np.arange(len(population), dtype=np.int64)
    labels = population[TARGET_COLUMN].to_numpy(dtype=np.int8)
    train_positions, remainder = train_test_split(
        positions, test_size=VALIDATION_FRACTION + TEST_FRACTION, stratify=labels, random_state=random_state
    )
    validation_positions, test_positions = train_test_split(
        remainder,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        stratify=labels[remainder],
        random_state=random_state,
    )
    assignments = np.full(len(population), 255, dtype=np.uint8)
    assignments[train_positions] = 0
    assignments[validation_positions] = 1
    assignments[test_positions] = 2
    if (assignments == 255).any():
        raise RuntimeError("Stage 21 did not assign every row exactly once.")

    train = population.iloc[np.sort(train_positions)].copy()
    validation = population.iloc[np.sort(validation_positions)].copy()
    for column in CATEGORICAL_FEATURES:
        train[column] = train[column].cat.remove_unused_categories()
        validation[column] = validation[column].cat.remove_unused_categories()
    if train[TARGET_COLUMN].nunique() != 2 or validation[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Training and validation must both contain the two target classes.")
    manifest = {
        "random_state": random_state,
        "strategy": "stratified_random_70_15_15",
        "train_rows": int(len(train_positions)),
        "validation_rows": int(len(validation_positions)),
        "held_out_test_rows": int(len(test_positions)),
        "train_positives": int(train[TARGET_COLUMN].sum()),
        "validation_positives": int(validation[TARGET_COLUMN].sum()),
        "test_frame_materialised": False,
        "test_split_evaluated": False,
        "split_assignment_sha256": _split_receipt(population[ID_COLUMN].to_numpy(), assignments, source_sha256),
    }
    return DevelopmentSplit(train=train, validation=validation, manifest=manifest)


# %% Fixed LightGBM pipeline
def build_baseline_pipeline(class_weight: float, *, random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    if not np.isfinite(class_weight) or class_weight <= 0:
        raise ValueError("The positive class weight must be finite and greater than zero.")
    overlap = set(FEATURE_COLUMNS).intersection(PROHIBITED_MODEL_FIELDS)
    if overlap:
        raise RuntimeError(f"Prohibited fields entered the Stage 21 feature contract: {sorted(overlap)}")
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    model = lightgbm.LGBMClassifier(
        objective="binary",
        scale_pos_weight=float(class_weight),
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        **LIGHTGBM_PARAMETERS,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


# %% Validation-only workload and stability evidence
def fixed_review_workloads(
    labels: Sequence[int], scores: Sequence[float], transaction_hashes: Sequence[int]
) -> pd.DataFrame:
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=float)
    hashes = np.asarray(transaction_hashes, dtype=np.uint64)
    if not (labels_array.ndim == scores_array.ndim == hashes.ndim == 1):
        raise ValueError("Labels, scores and transaction hashes must be one-dimensional.")
    if not (len(labels_array) == len(scores_array) == len(hashes)) or len(labels_array) == 0:
        raise ValueError("Labels, scores and transaction hashes must have equal non-zero lengths.")
    if not np.isin(labels_array, [0, 1]).all() or labels_array.sum() == 0:
        raise ValueError("Workload evaluation requires binary labels with positive cases.")
    if not np.isfinite(scores_array).all():
        raise ValueError("Validation scores must be finite.")
    order = np.lexsort((hashes, -scores_array))
    total_positives = int(labels_array.sum())
    rows = []
    for rate in REVIEW_RATES:
        alert_count = max(1, int(np.floor(len(labels_array) * rate)))
        selected = order[:alert_count]
        captured = int(labels_array[selected].sum())
        rows.append(
            {
                "requested_review_rate": rate,
                "reviewed_rows": alert_count,
                "actual_review_rate": alert_count / len(labels_array),
                "laundering_rows_captured": captured,
                "legitimate_rows_reviewed": alert_count - captured,
                "precision": captured / alert_count,
                "recall": captured / total_positives,
            }
        )
    return pd.DataFrame(rows)


def currency_amount_capture(
    validation: pd.DataFrame, scores: Sequence[float], *, minimum_positives: int = MINIMUM_CURRENCY_POSITIVES
) -> pd.DataFrame:
    scores_array = np.asarray(scores, dtype=float)
    if len(validation) != len(scores_array):
        raise ValueError("Validation rows and scores must have equal lengths.")
    labels = validation[TARGET_COLUMN].to_numpy(dtype=np.int8)
    amounts = validation[AMOUNT_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Currency capture requires finite non-negative paid amounts.")
    order = np.lexsort((validation[ID_COLUMN].to_numpy(dtype=np.uint64), -scores_array))
    rows = []
    for rate in REVIEW_RATES:
        alert_count = max(1, int(np.floor(len(validation) * rate)))
        selected_mask = np.zeros(len(validation), dtype=bool)
        selected_mask[order[:alert_count]] = True
        evidence = pd.DataFrame(
            {
                "currency": validation[CURRENCY_COLUMN].astype("string").to_numpy(),
                "label": labels,
                "amount": amounts,
                "selected": selected_mask,
            }
        )
        for currency, group in evidence.groupby("currency", sort=True, observed=True):
            laundering = group.loc[group.label == 1]
            captured = laundering.loc[laundering.selected]
            positive_count = int(len(laundering))
            total_amount = float(laundering.amount.sum())
            captured_amount = float(captured.amount.sum())
            rows.append(
                {
                    "requested_review_rate": rate,
                    "currency": str(currency),
                    "laundering_rows": positive_count,
                    "eligible_for_amount_interpretation": positive_count >= minimum_positives,
                    "laundering_amount": total_amount,
                    "captured_laundering_amount": captured_amount,
                    "amount_recall_within_currency": captured_amount / total_amount if total_amount > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _stability_row(name: str, group: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positives = int(labels.sum())
    prevalence = float(labels.mean()) if len(labels) else 0.0
    if len(labels) and 0 < positives < len(labels):
        average_precision = float(average_precision_score(labels, scores))
        lift = average_precision / prevalence if prevalence > 0 else None
    else:
        average_precision, lift = None, None
    eligible = positives >= MINIMUM_STABILITY_POSITIVES and lift is not None
    return {
        "stability_dimension": name,
        "group": group,
        "rows": int(len(labels)),
        "laundering_rows": positives,
        "prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": lift,
        "eligible_for_stability_decision": eligible,
        "passes_stability_floor": bool(eligible and lift >= MINIMUM_STABILITY_AP_LIFT),
    }


def validation_stability(validation: pd.DataFrame, scores: Sequence[float]) -> pd.DataFrame:
    scores_array = np.asarray(scores, dtype=float)
    if len(validation) != len(scores_array):
        raise ValueError("Validation rows and scores must have equal lengths.")
    labels = validation[TARGET_COLUMN].to_numpy(dtype=np.int8)
    hashes = validation[ID_COLUMN].to_numpy(dtype=np.uint64)
    timestamps = pd.to_datetime(validation[TIME_COLUMN])
    boundaries = [pd.Timestamp("2022-09-03 12:00"), pd.Timestamp("2022-09-06 00:00"), pd.Timestamp("2022-09-08 12:00")]
    time_group = np.select(
        [timestamps < boundaries[0], timestamps < boundaries[1], timestamps < boundaries[2]],
        ["time_q1", "time_q2", "time_q3"],
        default="time_q4",
    )
    rows = []
    for value in (0, 1):
        mask = hashes % np.uint64(2) == value
        rows.append(_stability_row("id_hash_half", f"hash_{value}", labels[mask], scores_array[mask]))
    for group in ("time_q1", "time_q2", "time_q3", "time_q4"):
        mask = time_group == group
        rows.append(_stability_row("primary_period_time_slice", group, labels[mask], scores_array[mask]))
    return pd.DataFrame(rows)


def baseline_progression_decision(
    *, average_precision: float, prevalence: float, workloads: pd.DataFrame, stability: pd.DataFrame, smoke: bool
) -> dict[str, Any]:
    if smoke:
        return {
            "eligible": False,
            "outcome": "not_evaluated_smoke_test",
            "reason": "Smoke evidence checks software only and cannot decide IBM baseline progression.",
        }
    if not 0 < prevalence < 1 or not np.isfinite(average_precision):
        raise ValueError("A finite Average Precision and binary validation prevalence are required.")
    top_one = workloads.loc[np.isclose(workloads.requested_review_rate, 0.01)]
    if len(top_one) != 1:
        raise ValueError("The workload table must contain exactly one top-1% row.")
    ap_lift = average_precision / prevalence
    top_one_recall = float(top_one.iloc[0].recall)
    stability_passes = bool(len(stability) == 6 and stability.passes_stability_floor.all())
    if ap_lift < MINIMUM_AP_LIFT or not stability_passes:
        outcome = "stop_weak_signal"
        reason = "Average Precision lift or the fixed subgroup stability requirement was not met."
    elif top_one_recall < MINIMUM_TOP_1PCT_RECALL:
        outcome = "discuss_limited_signal"
        reason = "Average Precision lift passed, but top-1% laundering recall remained below 25%."
    else:
        outcome = "candidate_for_one_time_test"
        reason = "The validation-only lift, top-1% recall and subgroup stability gates passed."
    return {
        "eligible": True,
        "outcome": outcome,
        "validation_average_precision_lift": ap_lift,
        "validation_top_1pct_recall": top_one_recall,
        "stability_passes": stability_passes,
        "reason": reason,
        "test_split_evaluated": False,
    }


# %% Fit one model and evaluate validation only
def fit_and_evaluate_baseline(pipeline: Pipeline, train: pd.DataFrame, validation: pd.DataFrame) -> ValidationResult:
    """Fit on training and evaluate validation; this interface cannot receive a test frame."""

    if set(FEATURE_COLUMNS).intersection(PROHIBITED_MODEL_FIELDS):
        raise RuntimeError("The Stage 21 feature contract contains a prohibited field.")
    fit_started = perf_counter()
    pipeline.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    training_seconds = perf_counter() - fit_started
    inference_started = perf_counter()
    scores, _, score_type = predict_scores(pipeline, validation[FEATURE_COLUMNS])
    inference_seconds = perf_counter() - inference_started
    threshold, _ = best_f1_threshold(validation[TARGET_COLUMN], scores)
    threshold_metrics = classification_metrics(validation[TARGET_COLUMN], scores, threshold)
    prevalence = float(validation[TARGET_COLUMN].mean())
    metrics = {
        "validation_prevalence": prevalence,
        "validation_average_precision": float(average_precision_score(validation[TARGET_COLUMN], scores)),
        "validation_average_precision_lift": float(
            average_precision_score(validation[TARGET_COLUMN], scores) / prevalence
        ),
        "validation_roc_auc": float(roc_auc_score(validation[TARGET_COLUMN], scores)),
        "validation_max_f1_threshold": threshold,
        "training_seconds": training_seconds,
        "validation_inference_seconds": inference_seconds,
        **{f"validation_{key}_at_max_f1": value for key, value in threshold_metrics.items()},
    }
    workloads = fixed_review_workloads(validation[TARGET_COLUMN], scores, validation[ID_COLUMN])
    currency = currency_amount_capture(validation, scores)
    stability = validation_stability(validation, scores)
    return ValidationResult(
        metrics=metrics,
        workloads=workloads,
        currency_capture=currency,
        stability=stability,
        scores=scores,
        score_type=score_type,
    )


# %% Aggregate validation figures
def save_validation_figures(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    workloads: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, thresholds = precision_recall_curve(labels_array, scores_array)
    if len(precision) > 2_000:
        indices = np.linspace(0, len(precision) - 1, 2_000, dtype=int)
        precision_plot, recall_plot = precision[indices], recall[indices]
    else:
        precision_plot, recall_plot = precision, recall
    paths = []

    path = output_dir / "validation_precision_recall_curve.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_plot, precision_plot)
    ax.axhline(float(labels_array.mean()), color="grey", linestyle="--", label="Laundering prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="IBM AML validation precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "validation_confusion_matrix.png"
    matrix = confusion_matrix(labels_array, scores_array >= threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(f"Validation confusion matrix at threshold {threshold:.6g}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    if thresholds.size:
        denominator = precision[:-1] + recall[:-1]
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0
        )
        if len(thresholds) > 2_000:
            indices = np.linspace(0, len(thresholds) - 1, 2_000, dtype=int)
            threshold_plot, precision_threshold, recall_threshold, f1_plot = (
                thresholds[indices],
                precision[:-1][indices],
                recall[:-1][indices],
                f1[indices],
            )
        else:
            threshold_plot, precision_threshold, recall_threshold, f1_plot = thresholds, precision[:-1], recall[:-1], f1
        path = output_dir / "validation_threshold_tradeoff.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(threshold_plot, precision_threshold, label="Precision")
        ax.plot(threshold_plot, recall_threshold, label="Recall")
        ax.plot(threshold_plot, f1_plot, label="F1")
        ax.axvline(threshold, color="black", linestyle="--", label="Maximum-F1 threshold")
        ax.set(xlabel="Decision threshold", ylabel="Metric", title="IBM AML validation threshold trade-off")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    path = output_dir / "validation_fixed_review_workloads.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    review_percent = workloads["actual_review_rate"] * 100
    ax.plot(review_percent, workloads["precision"], marker="o", label="Precision")
    ax.plot(review_percent, workloads["recall"], marker="o", label="Recall")
    ax.set(xlabel="Transactions reviewed (%)", ylabel="Metric", title="IBM AML fixed review workloads")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


# %% Serialisable protocol description
def feature_contract() -> dict[str, Any]:
    return {
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "prohibited_model_fields": sorted(PROHIBITED_MODEL_FIELDS),
        "target": TARGET_COLUMN,
        "raw_account_identifiers_used": False,
        "future_information_used": False,
    }


def model_parameters(class_weight: float) -> dict[str, Any]:
    return {
        **LIGHTGBM_PARAMETERS,
        "objective": "binary",
        "scale_pos_weight": float(class_weight),
        "random_state": DEFAULT_RANDOM_STATE,
        "configuration": "untuned_stage_11_reference_reused_for_ibm_feasibility",
    }
