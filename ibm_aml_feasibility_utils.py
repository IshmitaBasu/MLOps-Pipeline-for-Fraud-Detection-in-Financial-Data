"""Chunked, aggregate-only helpers for the IBM AML feasibility audit."""

# %% Imports and frozen audit definitions
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Iterable

import numpy as np
import pandas as pd

RAW_HEADER: Final = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]
CANONICAL_COLUMNS: Final = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
CATEGORICAL_COLUMNS: Final = [
    "from_bank",
    "to_bank",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]
ENTITY_COLUMNS: Final = ["from_bank", "from_account", "to_bank", "to_account"]
EXPECTED_SHA256: Final = "B19D39F515523373F991B689C07E11E7B0B95C17A2C27A87D91584AE16C5B040"
PRIMARY_PERIOD_END_EXCLUSIVE: Final = pd.Timestamp("2022-09-11 00:00:00")
MINIMUM_POSITIVE_ROWS: Final = 1_000
MINIMUM_HISTORY_COVERAGE: Final = 0.20
DEFAULT_CHUNK_ROWS: Final = 200_000
SMOKE_ROWS: Final = 100_000
FINGERPRINT_KEY_A: Final = "ibm_aml_audit_a1"
FINGERPRINT_KEY_B: Final = "ibm_aml_audit_b2"


# %% Small aggregate containers
@dataclass
class NumericAggregate:
    valid_rows: int = 0
    invalid_rows: int = 0
    negative_rows: int = 0
    zero_rows: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    total: float = 0.0

    def update(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        self.invalid_rows += int(len(numeric) - len(finite))
        if finite.empty:
            return
        self.valid_rows += int(len(finite))
        self.negative_rows += int((finite < 0).sum())
        self.zero_rows += int((finite == 0).sum())
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        self.total += float(finite.sum())

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "negative_rows": self.negative_rows,
            "zero_rows": self.zero_rows,
            "minimum": self.minimum if self.valid_rows else None,
            "maximum": self.maximum if self.valid_rows else None,
            "mean": self.total / self.valid_rows if self.valid_rows else None,
        }


@dataclass
class HistoryAggregate:
    """Calculate strictly earlier coverage independently of CSV row order.

    For each entity, every occurrence after its minimum timestamp has prior
    history. All occurrences at that minimum timestamp are excluded so that
    simultaneous transactions never count one another as history.
    """

    sender_timelines: dict[tuple[str, str], tuple[pd.Timestamp, int, int]] = field(default_factory=dict)
    receiver_timelines: dict[tuple[str, str], tuple[pd.Timestamp, int, int]] = field(default_factory=dict)
    pair_timelines: dict[
        tuple[tuple[str, str], tuple[str, str]], tuple[pd.Timestamp, int, int]
    ] = field(default_factory=dict)
    eligible_rows: int = 0

    @staticmethod
    def _entity(bank: str, account: str) -> tuple[str, str] | None:
        bank_value, account_value = str(bank).strip(), str(account).strip()
        return (bank_value, account_value) if bank_value and account_value else None

    @staticmethod
    def _update_timeline(
        timeline: dict[Any, tuple[pd.Timestamp, int, int]],
        counts: Counter,
        timestamp: pd.Timestamp,
    ) -> None:
        for entity, count in counts.items():
            current = timeline.get(entity)
            if current is None:
                timeline[entity] = (timestamp, count, count)
                continue
            minimum, count_at_minimum, total = current
            if timestamp < minimum:
                timeline[entity] = (timestamp, count, total + count)
            elif timestamp == minimum:
                timeline[entity] = (minimum, count_at_minimum + count, total + count)
            else:
                timeline[entity] = (minimum, count_at_minimum, total + count)

    def update_timestamp_group(self, group: pd.DataFrame) -> None:
        timestamp = pd.Timestamp(group["_timestamp"].iloc[0])
        senders = [self._entity(bank, account) for bank, account in zip(group.from_bank, group.from_account)]
        receivers = [self._entity(bank, account) for bank, account in zip(group.to_bank, group.to_account)]
        pairs = [
            (sender, receiver) if sender is not None and receiver is not None else None
            for sender, receiver in zip(senders, receivers)
        ]
        self.eligible_rows += len(group)
        self._update_timeline(self.sender_timelines, Counter(sender for sender in senders if sender is not None), timestamp)
        self._update_timeline(
            self.receiver_timelines, Counter(receiver for receiver in receivers if receiver is not None), timestamp
        )
        self._update_timeline(self.pair_timelines, Counter(pair for pair in pairs if pair is not None), timestamp)

    @staticmethod
    def _prior_rows(timeline: dict[Any, tuple[pd.Timestamp, int, int]]) -> int:
        return sum(total - count_at_minimum for _, count_at_minimum, total in timeline.values())

    @property
    def sender_prior_rows(self) -> int:
        return self._prior_rows(self.sender_timelines)

    @property
    def receiver_prior_rows(self) -> int:
        return self._prior_rows(self.receiver_timelines)

    @property
    def pair_prior_rows(self) -> int:
        return self._prior_rows(self.pair_timelines)

    def as_dict(self, total_rows: int) -> dict[str, int | float]:
        denominator = max(total_rows, 1)
        sender_prior_rows = self.sender_prior_rows
        receiver_prior_rows = self.receiver_prior_rows
        pair_prior_rows = self.pair_prior_rows
        return {
            "eligible_timestamped_rows": self.eligible_rows,
            "unique_senders": len(self.sender_timelines),
            "unique_receivers": len(self.receiver_timelines),
            "unique_sender_receiver_pairs": len(self.pair_timelines),
            "sender_prior_rows": sender_prior_rows,
            "receiver_prior_rows": receiver_prior_rows,
            "pair_prior_rows": pair_prior_rows,
            "sender_prior_history_coverage": sender_prior_rows / denominator,
            "receiver_prior_history_coverage": receiver_prior_rows / denominator,
            "pair_prior_history_coverage": pair_prior_rows / denominator,
            "calculation_order_independent": True,
        }


# %% Streaming audit state
class AuditAccumulator:
    """Accumulate bounded-memory statistics without retaining transaction rows."""

    def __init__(self) -> None:
        self.rows = 0
        self.maximum_chunk_memory_bytes = 0
        self.missing = Counter()
        self.label_counts = Counter()
        self.invalid_label_rows = 0
        self.invalid_timestamp_rows = 0
        self.timestamp_min: pd.Timestamp | None = None
        self.timestamp_max: pd.Timestamp | None = None
        self.timestamp_source_order_non_decreasing = True
        self.last_timestamp: pd.Timestamp | None = None
        self.daily_counts: Counter = Counter()
        self.daily_laundering: Counter = Counter()
        self.category_counts: dict[str, Counter] = {column: Counter() for column in CATEGORICAL_COLUMNS}
        self.category_laundering: dict[str, Counter] = {column: Counter() for column in CATEGORICAL_COLUMNS}
        self.amount_received = NumericAggregate()
        self.amount_paid = NumericAggregate()
        self.same_currency_rows = 0
        self.cross_currency_rows = 0
        self.same_currency_equal_amount_rows = 0
        self.same_entity_rows = 0
        self.primary_period_rows = 0
        self.primary_period_laundering_rows = 0
        self.post_primary_rows = 0
        self.post_primary_laundering_rows = 0
        self.history = HistoryAggregate()
        self.hash_a: list[np.ndarray] = []
        self.hash_b: list[np.ndarray] = []

    def update(self, frame: pd.DataFrame) -> None:
        self.rows += len(frame)
        self.maximum_chunk_memory_bytes = max(
            self.maximum_chunk_memory_bytes, int(frame.memory_usage(index=True, deep=True).sum())
        )
        for column in CANONICAL_COLUMNS:
            self.missing[column] += int(frame[column].astype("string").str.strip().eq("").sum())

        text_frame = frame.astype("string").fillna("")
        self.hash_a.append(
            pd.util.hash_pandas_object(text_frame, index=False, hash_key=FINGERPRINT_KEY_A).to_numpy(dtype=np.uint64)
        )
        self.hash_b.append(
            pd.util.hash_pandas_object(text_frame, index=False, hash_key=FINGERPRINT_KEY_B).to_numpy(dtype=np.uint64)
        )

        labels_numeric = pd.to_numeric(frame["is_laundering"], errors="coerce")
        valid_label = labels_numeric.isin([0, 1])
        self.invalid_label_rows += int((~valid_label).sum())
        labels = labels_numeric.where(valid_label).astype("Int8")
        self.label_counts.update(labels.dropna().astype(int).tolist())

        timestamps = pd.to_datetime(frame["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
        valid_timestamp = timestamps.notna()
        self.invalid_timestamp_rows += int((~valid_timestamp).sum())
        valid_times = timestamps.loc[valid_timestamp]
        if not valid_times.empty:
            if not valid_times.is_monotonic_increasing:
                self.timestamp_source_order_non_decreasing = False
            if self.last_timestamp is not None and valid_times.iloc[0] < self.last_timestamp:
                self.timestamp_source_order_non_decreasing = False
            self.last_timestamp = valid_times.iloc[-1]
            observed_min, observed_max = valid_times.min(), valid_times.max()
            self.timestamp_min = observed_min if self.timestamp_min is None else min(self.timestamp_min, observed_min)
            self.timestamp_max = observed_max if self.timestamp_max is None else max(self.timestamp_max, observed_max)

        valid_time_and_label = valid_timestamp & valid_label
        if valid_time_and_label.any():
            daily = pd.DataFrame(
                {
                    "day": timestamps.loc[valid_time_and_label].dt.floor("D"),
                    "label": labels_numeric.loc[valid_time_and_label].astype("int8"),
                }
            ).groupby("day", sort=False).label.agg(["size", "sum"])
            self.daily_counts.update({day: int(row["size"]) for day, row in daily.iterrows()})
            self.daily_laundering.update({day: int(row["sum"]) for day, row in daily.iterrows()})

            primary_mask = timestamps.loc[valid_time_and_label] < PRIMARY_PERIOD_END_EXCLUSIVE
            valid_labels = labels_numeric.loc[valid_time_and_label].astype("int8")
            self.primary_period_rows += int(primary_mask.sum())
            self.primary_period_laundering_rows += int(valid_labels.loc[primary_mask].sum())
            self.post_primary_rows += int((~primary_mask).sum())
            self.post_primary_laundering_rows += int(valid_labels.loc[~primary_mask].sum())

        for column in CATEGORICAL_COLUMNS:
            values = frame[column].astype("string").fillna("").str.strip()
            self.category_counts[column].update(values.tolist())
            labelled = pd.DataFrame({"value": values.loc[valid_label], "label": labels_numeric.loc[valid_label]})
            grouped = labelled.groupby("value", dropna=False, sort=False).label.sum()
            self.category_laundering[column].update({str(value): int(count) for value, count in grouped.items()})

        self.amount_received.update(frame["amount_received"])
        self.amount_paid.update(frame["amount_paid"])
        received = pd.to_numeric(frame["amount_received"], errors="coerce")
        paid = pd.to_numeric(frame["amount_paid"], errors="coerce")
        receiving_currency = frame["receiving_currency"].astype("string").fillna("").str.strip()
        payment_currency = frame["payment_currency"].astype("string").fillna("").str.strip()
        same_currency = receiving_currency.eq(payment_currency)
        self.same_currency_rows += int(same_currency.sum())
        self.cross_currency_rows += int((~same_currency).sum())
        self.same_currency_equal_amount_rows += int((same_currency & received.eq(paid)).sum())

        from_entity = frame["from_bank"].astype("string") + "\x1f" + frame["from_account"].astype("string")
        to_entity = frame["to_bank"].astype("string") + "\x1f" + frame["to_account"].astype("string")
        self.same_entity_rows += int(from_entity.eq(to_entity).sum())

        history_frame = frame.loc[valid_timestamp, ENTITY_COLUMNS].copy()
        history_frame["_timestamp"] = timestamps.loc[valid_timestamp].to_numpy()
        for _, group in history_frame.groupby("_timestamp", sort=False, observed=True):
            self.history.update_timestamp_group(group)

    def duplicate_fingerprint_rows(self) -> int:
        if not self.hash_a:
            return 0
        first = np.concatenate(self.hash_a)
        second = np.concatenate(self.hash_b)
        order = np.lexsort((second, first))
        first, second = first[order], second[order]
        repeated = (first[1:] == first[:-1]) & (second[1:] == second[:-1])
        return int(repeated.sum())


# %% CSV scan and aggregate table construction
def _validate_header(header: list[str]) -> None:
    if header != RAW_HEADER:
        raise ValueError(f"Unexpected IBM AML header. Expected {RAW_HEADER}, observed {header}.")


def iter_valid_chunks(
    path: Path, *, chunk_rows: int = DEFAULT_CHUNK_ROWS, row_limit: int | None = None
) -> tuple[Iterable[pd.DataFrame], dict[str, Any]]:
    """Yield valid rows while retaining a mutable structural receipt."""

    if chunk_rows < 1_000:
        raise ValueError("chunk_rows must be at least 1,000.")
    receipt: dict[str, Any] = {
        "physical_data_rows_seen": 0,
        "valid_structure_rows": 0,
        "malformed_structure_rows": 0,
        "malformed_line_examples": [],
        "complete_file_scanned": row_limit is None,
    }

    def generator() -> Iterable[pd.DataFrame]:
        buffer: list[list[str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("The IBM AML CSV is empty.") from error
            _validate_header(header)
            for line_number, row in enumerate(reader, start=2):
                if row_limit is not None and receipt["physical_data_rows_seen"] >= row_limit:
                    break
                receipt["physical_data_rows_seen"] += 1
                if len(row) != len(CANONICAL_COLUMNS):
                    receipt["malformed_structure_rows"] += 1
                    if len(receipt["malformed_line_examples"]) < 10:
                        receipt["malformed_line_examples"].append(
                            {"line_number": line_number, "observed_fields": len(row)}
                        )
                    continue
                receipt["valid_structure_rows"] += 1
                buffer.append(row)
                if len(buffer) >= chunk_rows:
                    yield pd.DataFrame.from_records(buffer, columns=CANONICAL_COLUMNS)
                    buffer = []
            if buffer:
                yield pd.DataFrame.from_records(buffer, columns=CANONICAL_COLUMNS)

    return generator(), receipt


def _category_table(accumulator: AuditAccumulator) -> pd.DataFrame:
    rows = []
    for feature in CATEGORICAL_COLUMNS:
        for group, count in accumulator.category_counts[feature].items():
            laundering = int(accumulator.category_laundering[feature][group])
            rows.append(
                {
                    "feature": feature,
                    "group": group if group else "<EMPTY>",
                    "rows": int(count),
                    "laundering_rows": laundering,
                    "laundering_rate": laundering / count if count else 0.0,
                    "row_coverage": count / max(accumulator.rows, 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "rows", "group"], ascending=[True, False, True])


def _daily_table(accumulator: AuditAccumulator) -> pd.DataFrame:
    rows = []
    for day in sorted(accumulator.daily_counts):
        count = int(accumulator.daily_counts[day])
        laundering = int(accumulator.daily_laundering[day])
        rows.append(
            {
                "day": pd.Timestamp(day).date().isoformat(),
                "rows": count,
                "laundering_rows": laundering,
                "laundering_rate": laundering / count if count else 0.0,
                "period": "primary" if pd.Timestamp(day) < PRIMARY_PERIOD_END_EXCLUSIVE else "post_primary",
            }
        )
    return pd.DataFrame(rows)


def _decision(summary: dict[str, Any], *, full_data: bool) -> dict[str, Any]:
    if not full_data:
        return {
            "eligible": False,
            "outcome": "not_evaluated_smoke_test",
            "reason": "Smoke evidence checks software only and cannot decide dataset suitability.",
        }
    quality, time_info, history = summary["quality"], summary["time"], summary["history"]
    conditions = {
        "source_identity_verified": bool(summary["source"]["sha256_matches_download_receipt"]),
        "row_structure_usable": quality["malformed_structure_rows"] == 0,
        "timestamps_parse_reliably": quality["invalid_timestamp_rows"] == 0,
        "target_is_binary_and_parseable": quality["invalid_label_rows"] == 0,
        "positive_support_at_least_1000": summary["labels"]["laundering_rows"] >= MINIMUM_POSITIVE_ROWS,
        "target_derived_pattern_fields_excluded": True,
        "sender_history_coverage_at_least_20pct": history["sender_prior_history_coverage"] >= MINIMUM_HISTORY_COVERAGE,
        "receiver_history_coverage_at_least_20pct": history["receiver_prior_history_coverage"] >= MINIMUM_HISTORY_COVERAGE,
        "fixed_tail_rule_leaves_both_classes": (
            time_info["primary_period_laundering_rows"] > 0
            and time_info["primary_period_laundering_rows"] < time_info["primary_period_rows"]
        ),
        "past_only_history_calculated_independently_of_source_order": history["calculation_order_independent"],
        "bounded_chunk_processing_completed": True,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    if failed:
        outcome = "do_not_progress"
        reason = "One or more mandatory technical conditions failed."
    else:
        outcome = "second_dataset_candidate"
        reason = (
            "Technical conditions passed, but the target is money laundering rather than ordinary transaction fraud. "
            "Confirm thesis scope before designing a baseline or migration."
        )
    return {"eligible": not failed, "outcome": outcome, "conditions": conditions, "failed_conditions": failed, "reason": reason}


def audit_ibm_aml_csv(
    path: Path,
    *,
    source_sha256: str,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    row_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Run the aggregate audit and return no row-level data."""

    if not path.exists():
        raise FileNotFoundError(f"IBM AML candidate file not found: {path}")
    started = perf_counter()
    chunks, structural = iter_valid_chunks(path, chunk_rows=chunk_rows, row_limit=row_limit)
    accumulator = AuditAccumulator()
    for chunk in chunks:
        accumulator.update(chunk)
    if accumulator.rows == 0:
        raise ValueError("The IBM AML audit found no structurally valid data rows.")

    duplicates = accumulator.duplicate_fingerprint_rows()
    history = accumulator.history.as_dict(accumulator.rows)
    labels = {
        "legitimate_rows": int(accumulator.label_counts[0]),
        "laundering_rows": int(accumulator.label_counts[1]),
        "labelled_rows": int(accumulator.label_counts[0] + accumulator.label_counts[1]),
    }
    labels["laundering_rate"] = labels["laundering_rows"] / max(labels["labelled_rows"], 1)
    time_info = {
        "minimum_timestamp": accumulator.timestamp_min.isoformat() if accumulator.timestamp_min is not None else None,
        "maximum_timestamp": accumulator.timestamp_max.isoformat() if accumulator.timestamp_max is not None else None,
        "source_order_non_decreasing": accumulator.timestamp_source_order_non_decreasing,
        "chronological_sort_required_before_sequential_features": (
            not accumulator.timestamp_source_order_non_decreasing
        ),
        "primary_period_end_exclusive": PRIMARY_PERIOD_END_EXCLUSIVE.isoformat(),
        "primary_period_rows": accumulator.primary_period_rows,
        "primary_period_laundering_rows": accumulator.primary_period_laundering_rows,
        "post_primary_rows": accumulator.post_primary_rows,
        "post_primary_laundering_rows": accumulator.post_primary_laundering_rows,
        "post_primary_all_laundering": bool(
            accumulator.post_primary_rows > 0
            and accumulator.post_primary_rows == accumulator.post_primary_laundering_rows
        ),
    }
    quality = {
        **structural,
        "audited_valid_rows": accumulator.rows,
        "invalid_timestamp_rows": accumulator.invalid_timestamp_rows,
        "invalid_label_rows": accumulator.invalid_label_rows,
        "missing_cells_by_column": dict(accumulator.missing),
        "duplicate_row_fingerprint_rows": duplicates,
        "duplicate_method": "two_independent_64_bit_full_row_hashes",
    }
    summary: dict[str, Any] = {
        "source": {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": source_sha256.upper(),
            "expected_sha256": EXPECTED_SHA256,
            "sha256_matches_download_receipt": source_sha256.upper() == EXPECTED_SHA256,
        },
        "scope": "full_file" if row_limit is None else "smoke_prefix",
        "row_limit": row_limit,
        "quality": quality,
        "labels": labels,
        "time": time_info,
        "amounts": {
            "amount_received": accumulator.amount_received.as_dict(),
            "amount_paid": accumulator.amount_paid.as_dict(),
            "same_currency_rows": accumulator.same_currency_rows,
            "cross_currency_rows": accumulator.cross_currency_rows,
            "same_currency_equal_amount_rows": accumulator.same_currency_equal_amount_rows,
        },
        "entities": {"same_sender_receiver_entity_rows": accumulator.same_entity_rows},
        "history": history,
        "processing": {
            "chunk_rows": chunk_rows,
            "maximum_chunk_dataframe_memory_bytes": accumulator.maximum_chunk_memory_bytes,
            "elapsed_seconds": perf_counter() - started,
            "row_level_output_created": False,
            "account_identifiers_exported": False,
        },
        "target_semantics": "money_laundering_not_general_transaction_fraud",
        "model_trained": False,
        "existing_pipeline_changed": False,
    }
    summary["decision"] = _decision(summary, full_data=row_limit is None)
    return summary, {"daily_label_summary": _daily_table(accumulator), "categorical_summary": _category_table(accumulator)}


# %% Readable aggregate interpretation
def audit_report_markdown(summary: dict[str, Any]) -> str:
    labels, time_info, history, decision = summary["labels"], summary["time"], summary["history"], summary["decision"]
    return "\n".join(
        [
            "# IBM AML feasibility audit report",
            "",
            f"Scope: `{summary['scope']}`. No model was trained and no existing pipeline was changed.",
            "",
            f"Audited rows: {summary['quality']['audited_valid_rows']:,}.",
            f"Laundering rows: {labels['laundering_rows']:,} ({labels['laundering_rate']:.6%}).",
            f"Timestamp range: {time_info['minimum_timestamp']} to {time_info['maximum_timestamp']}.",
            f"Raw source order is chronological: {time_info['source_order_non_decreasing']}.",
            "Past-only coverage was calculated independently of source row order; chronological sorting is required "
            "before later sequential feature construction.",
            f"Post-primary rows: {time_info['post_primary_rows']:,}; laundering rows: "
            f"{time_info['post_primary_laundering_rows']:,}.",
            "",
            f"Strictly earlier sender-history coverage: {history['sender_prior_history_coverage']:.2%}.",
            f"Strictly earlier receiver-history coverage: {history['receiver_prior_history_coverage']:.2%}.",
            f"Strictly earlier sender–receiver-pair coverage: {history['pair_prior_history_coverage']:.2%}.",
            "",
            f"Decision: `{decision['outcome']}`.",
            decision["reason"],
            "",
            "The decision concerns dataset feasibility only. It does not compare trained models, approve a dataset "
            "replacement, or reinterpret money laundering as ordinary transaction fraud.",
        ]
    ) + "\n"
