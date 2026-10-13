"""Synthetic safeguards for the Stage 20 IBM AML feasibility audit."""

# %% Imports and fixture helpers
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from ibm_aml_feasibility_utils import (
    CANONICAL_COLUMNS,
    EXPECTED_SHA256,
    RAW_HEADER,
    HistoryAggregate,
    audit_ibm_aml_csv,
    iter_valid_chunks,
)


def transaction_row(
    timestamp: str,
    label: str = "0",
    *,
    from_bank: str = "010",
    from_account: str = "FROM-1",
    to_bank: str = "020",
    to_account: str = "TO-1",
    amount_received: str = "100.00",
    receiving_currency: str = "US Dollar",
    amount_paid: str = "100.00",
    payment_currency: str = "US Dollar",
    payment_format: str = "ACH",
) -> list[str]:
    return [
        timestamp,
        from_bank,
        from_account,
        to_bank,
        to_account,
        amount_received,
        receiving_currency,
        amount_paid,
        payment_currency,
        payment_format,
        label,
    ]


def write_fixture(path: Path, rows: list[list[str]], header: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_HEADER if header is None else header)
        writer.writerows(rows)


# %% Header, structure and canonical account handling
class StructureTests(unittest.TestCase):
    def test_duplicate_account_headers_are_mapped_by_position_and_malformed_rows_are_counted(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.csv"
            valid = transaction_row("2022/09/01 00:00")
            malformed = valid[:-1]
            write_fixture(path, [valid, malformed])
            chunks, receipt = iter_valid_chunks(path, chunk_rows=1_000)
            frames = list(chunks)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].columns.tolist(), CANONICAL_COLUMNS)
            self.assertEqual(frames[0].loc[0, "from_account"], "FROM-1")
            self.assertEqual(frames[0].loc[0, "to_account"], "TO-1")
            self.assertEqual(receipt["valid_structure_rows"], 1)
            self.assertEqual(receipt["malformed_structure_rows"], 1)
            self.assertEqual(receipt["malformed_line_examples"][0]["observed_fields"], 10)

    def test_unexpected_header_is_rejected_instead_of_guessed(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.csv"
            altered = RAW_HEADER.copy()
            altered[2] = "From Account"
            write_fixture(path, [transaction_row("2022/09/01 00:00")], header=altered)
            chunks, _ = iter_valid_chunks(path, chunk_rows=1_000)
            with self.assertRaisesRegex(ValueError, "Unexpected IBM AML header"):
                list(chunks)


# %% Strictly earlier history, including chunk-boundary equality
class HistoryTests(unittest.TestCase):
    @staticmethod
    def group(timestamp: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "from_bank": ["010"],
                "from_account": ["FROM-1"],
                "to_bank": ["020"],
                "to_account": ["TO-1"],
                "_timestamp": [pd.Timestamp(timestamp)],
            }
        )

    def test_equal_time_rows_do_not_count_as_prior_even_across_updates(self):
        history = HistoryAggregate()
        history.update_timestamp_group(self.group("2022-09-01 00:00"))
        history.update_timestamp_group(self.group("2022-09-01 00:00"))
        self.assertEqual(history.sender_prior_rows, 0)
        self.assertEqual(history.receiver_prior_rows, 0)
        self.assertEqual(history.pair_prior_rows, 0)
        history.update_timestamp_group(self.group("2022-09-01 00:01"))
        self.assertEqual(history.sender_prior_rows, 1)
        self.assertEqual(history.receiver_prior_rows, 1)
        self.assertEqual(history.pair_prior_rows, 1)

    def test_out_of_order_rows_use_minimum_timestamp_not_file_position(self):
        history = HistoryAggregate()
        history.update_timestamp_group(self.group("2022-09-01 00:02"))
        history.update_timestamp_group(self.group("2022-09-01 00:00"))
        history.update_timestamp_group(self.group("2022-09-01 00:00"))
        self.assertEqual(history.sender_prior_rows, 1)
        self.assertEqual(history.receiver_prior_rows, 1)
        self.assertEqual(history.pair_prior_rows, 1)
        self.assertTrue(history.as_dict(3)["calculation_order_independent"])


# %% Aggregate quality, tail and privacy boundaries
class AggregateAuditTests(unittest.TestCase):
    def test_invalid_values_tail_duplicates_and_aggregate_outputs_are_visible(self):
        rows = [
            transaction_row("2022/09/01 00:00", "0"),
            transaction_row("2022/09/01 00:01", "1"),
            transaction_row("2022/09/11 00:00", "1"),
            transaction_row("not-a-time", "2"),
        ]
        rows.append(rows[2].copy())
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.csv"
            write_fixture(path, rows)
            summary, tables = audit_ibm_aml_csv(path, source_sha256=EXPECTED_SHA256, chunk_rows=1_000)
        self.assertEqual(summary["quality"]["invalid_timestamp_rows"], 1)
        self.assertEqual(summary["quality"]["invalid_label_rows"], 1)
        self.assertEqual(summary["quality"]["duplicate_row_fingerprint_rows"], 1)
        self.assertEqual(summary["time"]["post_primary_rows"], 2)
        self.assertEqual(summary["time"]["post_primary_laundering_rows"], 2)
        self.assertTrue(summary["time"]["post_primary_all_laundering"])
        self.assertEqual(set(tables), {"daily_label_summary", "categorical_summary"})
        self.assertNotIn("from_account", tables["categorical_summary"].feature.unique())
        self.assertFalse(summary["processing"]["account_identifiers_exported"])
        self.assertEqual(summary["decision"]["outcome"], "do_not_progress")

    def test_smoke_scope_cannot_make_a_suitability_decision(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.csv"
            write_fixture(
                path,
                [transaction_row("2022/09/01 00:00", "0"), transaction_row("2022/09/01 00:01", "1")],
            )
            summary, _ = audit_ibm_aml_csv(
                path, source_sha256=EXPECTED_SHA256, chunk_rows=1_000, row_limit=2
            )
        self.assertEqual(summary["scope"], "smoke_prefix")
        self.assertEqual(summary["decision"]["outcome"], "not_evaluated_smoke_test")
        self.assertFalse(summary["decision"]["eligible"])


# %% A technically suitable fixture remains a second-dataset candidate
class DecisionTests(unittest.TestCase):
    def test_all_technical_conditions_lead_to_second_dataset_candidate_not_replacement(self):
        start = datetime(2022, 9, 1)
        rows = []
        for index in range(2_400):
            timestamp = (start + timedelta(minutes=index)).strftime("%Y/%m/%d %H:%M")
            rows.append(
                transaction_row(
                    timestamp,
                    str(index % 2),
                    from_account=f"FROM-{index % 10}",
                    to_account=f"TO-{index % 12}",
                    amount_received=f"{100 + index / 100:.2f}",
                    amount_paid=f"{100 + index / 100:.2f}",
                )
            )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.csv"
            write_fixture(path, rows)
            summary, _ = audit_ibm_aml_csv(path, source_sha256=EXPECTED_SHA256, chunk_rows=1_000)
        self.assertTrue(summary["decision"]["eligible"])
        self.assertEqual(summary["decision"]["failed_conditions"], [])
        self.assertEqual(summary["decision"]["outcome"], "second_dataset_candidate")
        self.assertGreater(summary["history"]["sender_prior_history_coverage"], 0.99)
        self.assertGreater(summary["history"]["receiver_prior_history_coverage"], 0.99)
        self.assertFalse(summary["model_trained"])
        self.assertFalse(summary["existing_pipeline_changed"])

    def test_unsorted_source_requires_preprocessing_but_does_not_reject_dataset(self):
        start = datetime(2022, 9, 1)
        rows = []
        for index in range(2_400):
            timestamp = (start + timedelta(minutes=index)).strftime("%Y/%m/%d %H:%M")
            rows.append(
                transaction_row(
                    timestamp,
                    str(index % 2),
                    from_account=f"FROM-{index % 10}",
                    to_account=f"TO-{index % 12}",
                )
            )
        rows[0], rows[-1] = rows[-1], rows[0]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.csv"
            write_fixture(path, rows)
            summary, _ = audit_ibm_aml_csv(path, source_sha256=EXPECTED_SHA256, chunk_rows=1_000)
        self.assertFalse(summary["time"]["source_order_non_decreasing"])
        self.assertTrue(summary["time"]["chronological_sort_required_before_sequential_features"])
        self.assertTrue(summary["history"]["calculation_order_independent"])
        self.assertTrue(summary["decision"]["eligible"])
        self.assertEqual(summary["decision"]["outcome"], "second_dataset_candidate")


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
