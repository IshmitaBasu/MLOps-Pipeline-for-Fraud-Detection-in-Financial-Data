"""Safeguards for the Stage 21 IBM AML validation-only baseline."""

# %% Imports and fixtures
from __future__ import annotations

import csv
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mlflow.sklearn
import numpy as np
import pandas as pd

from fraud_detection.datasets.ibm_aml_baseline import (
    CATEGORICAL_FEATURES,
    EXPECTED_SHA256,
    FEATURE_COLUMNS,
    ID_COLUMN,
    MODEL_SERIALIZATION_FORMAT,
    NUMERIC_FEATURES,
    PROHIBITED_MODEL_FIELDS,
    SOURCE_ROW_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    baseline_progression_decision,
    build_baseline_pipeline,
    currency_amount_capture,
    deterministic_transaction_hash,
    feature_contract,
    fit_and_evaluate_baseline,
    make_development_split,
    prepare_modeling_population,
)
from fraud_detection.datasets.ibm_aml_feasibility import RAW_HEADER
from fraud_detection.evaluation.validation import positive_class_weight


def raw_row(timestamp: str, label: str = "0", *, account: str = "FROM-1", amount: str = "100.00") -> list[str]:
    return [timestamp, "010", account, "020", "TO-1", amount, "US Dollar", amount, "US Dollar", "ACH", label]


def write_raw(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_HEADER)
        writer.writerows(rows)


def modeling_population(rows: int = 200) -> pd.DataFrame:
    labels = np.zeros(rows, dtype=np.int8)
    labels[::10] = 1
    timestamps = pd.date_range("2022-09-01", periods=rows, freq="min")
    source_rows = np.arange(1, rows + 1, dtype=np.int64)
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: source_rows,
            TIME_COLUMN: timestamps,
            ID_COLUMN: deterministic_transaction_hash(source_rows, EXPECTED_SHA256),
            "from_bank": pd.Series(np.where(source_rows % 2, "010", "011"), dtype="category"),
            "to_bank": pd.Series(np.where(source_rows % 3, "020", "021"), dtype="category"),
            "receiving_currency": pd.Series("US Dollar", index=range(rows), dtype="category"),
            "payment_currency": pd.Series("US Dollar", index=range(rows), dtype="category"),
            "payment_format": pd.Series("ACH", index=range(rows), dtype="category"),
            "amount_paid": np.linspace(10.0, 100.0, rows),
            "log_amount_received": np.log1p(np.linspace(10.0, 100.0, rows)),
            "log_amount_paid": np.log1p(np.linspace(10.0, 100.0, rows)),
            "event_hour": timestamps.hour.astype(np.int8),
            "event_day_of_week": timestamps.dayofweek.astype(np.int8),
            "same_bank": np.zeros(rows, dtype=np.int8),
            "same_currency": np.ones(rows, dtype=np.int8),
            "equal_amount": np.ones(rows, dtype=np.int8),
            TARGET_COLUMN: labels,
        }
    )
    return frame


# %% Population boundary, duplicates and privacy
class PopulationTests(unittest.TestCase):
    def test_primary_period_is_sorted_deduplicated_and_contains_no_accounts(self):
        duplicate = raw_row("2022/09/01 00:02", "1", account="SECRET-1")
        rows = [
            duplicate,
            raw_row("2022/09/01 00:00", "0", account="SECRET-2"),
            duplicate.copy(),
            raw_row("2022/09/11 00:00", "1", account="TAIL"),
        ]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "ibm.csv"
            write_raw(path, rows)
            population, receipt = prepare_modeling_population(
                path, source_sha256=EXPECTED_SHA256, chunk_rows=1_000, row_limit=len(rows)
            )
        self.assertEqual(receipt["pre_deduplication_primary_rows"], 3)
        self.assertEqual(receipt["duplicates_removed"], 1)
        self.assertEqual(receipt["modeling_rows"], 2)
        self.assertFalse(receipt["account_identifiers_retained"])
        self.assertFalse(receipt["raw_tail_rows_retained"])
        self.assertNotIn("from_account", population.columns)
        self.assertNotIn("to_account", population.columns)
        self.assertTrue(population[TIME_COLUMN].is_monotonic_increasing)
        self.assertTrue((population[TIME_COLUMN] < pd.Timestamp("2022-09-11")).all())

    def test_transaction_hash_is_deterministic_and_source_bound(self):
        rows = np.array([1, 2, 3], dtype=np.int64)
        first = deterministic_transaction_hash(rows, EXPECTED_SHA256)
        second = deterministic_transaction_hash(rows, EXPECTED_SHA256)
        changed_source = "A" * 64
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, deterministic_transaction_hash(rows, changed_source)))
        self.assertEqual(len(np.unique(first)), 3)


# %% Frozen split and test isolation
class SplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_only_development_frames_are_returned(self):
        population = modeling_population()
        first = make_development_split(population, source_sha256=EXPECTED_SHA256)
        second = make_development_split(population, source_sha256=EXPECTED_SHA256)
        self.assertEqual(len(first.train), 140)
        self.assertEqual(len(first.validation), 30)
        self.assertEqual(first.manifest["held_out_test_rows"], 30)
        self.assertFalse(first.manifest["test_frame_materialised"])
        self.assertFalse(first.manifest["test_split_evaluated"])
        self.assertEqual(first.manifest["split_assignment_sha256"], second.manifest["split_assignment_sha256"])
        self.assertEqual(set(first.__dataclass_fields__), {"train", "validation", "manifest"})

    def test_validation_interface_cannot_receive_a_test_frame(self):
        parameters = inspect.signature(fit_and_evaluate_baseline).parameters
        self.assertEqual(set(parameters), {"pipeline", "train", "validation"})

    def test_too_few_positive_rows_are_rejected_before_three_way_split(self):
        population = modeling_population(50)
        population[TARGET_COLUMN] = 0
        population.loc[:4, TARGET_COLUMN] = 1
        with self.assertRaisesRegex(ValueError, "At least seven positive rows"):
            make_development_split(population, source_sha256=EXPECTED_SHA256)


# %% Feature and class-weight contract
class ModelContractTests(unittest.TestCase):
    def test_feature_contract_excludes_labels_ids_accounts_and_time(self):
        contract = feature_contract()
        self.assertEqual(contract["feature_columns"], FEATURE_COLUMNS)
        self.assertFalse(set(FEATURE_COLUMNS).intersection(PROHIBITED_MODEL_FIELDS))
        self.assertFalse(contract["raw_account_identifiers_used"])
        self.assertFalse(contract["future_information_used"])
        self.assertEqual(set(FEATURE_COLUMNS), set(CATEGORICAL_FEATURES + NUMERIC_FEATURES))

    def test_lightgbm_receives_training_only_class_weight(self):
        split = make_development_split(modeling_population(), source_sha256=EXPECTED_SHA256)
        weight = positive_class_weight(split.train[TARGET_COLUMN])
        pipeline = build_baseline_pipeline(weight)
        self.assertAlmostEqual(pipeline.named_steps["model"].scale_pos_weight, weight)
        self.assertEqual(pipeline.named_steps["model"].n_estimators, 300)

    def test_fitted_pipeline_round_trips_with_the_pinned_serialization_format(self):
        split = make_development_split(modeling_population(), source_sha256=EXPECTED_SHA256)
        pipeline = build_baseline_pipeline(positive_class_weight(split.train[TARGET_COLUMN]))
        pipeline.fit(split.train[FEATURE_COLUMNS], split.train[TARGET_COLUMN])
        expected = pipeline.predict_proba(split.validation[FEATURE_COLUMNS].head(10))[:, 1]
        with TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model"
            mlflow.sklearn.save_model(
                pipeline,
                path=model_path,
                serialization_format=MODEL_SERIALIZATION_FORMAT,
            )
            loaded = mlflow.sklearn.load_model(model_path)
            observed = loaded.predict_proba(split.validation[FEATURE_COLUMNS].head(10))[:, 1]
        np.testing.assert_allclose(observed, expected)


# %% Currency-safe reporting and fixed decisions
class DecisionTests(unittest.TestCase):
    def test_currency_amounts_remain_separate(self):
        validation = modeling_population(100)
        validation["payment_currency"] = pd.Series(np.where(np.arange(100) < 50, "US Dollar", "Euro"), dtype="category")
        scores = np.linspace(0.0, 1.0, 100)
        table = currency_amount_capture(validation, scores, minimum_positives=1)
        self.assertEqual(set(table.currency), {"US Dollar", "Euro"})
        self.assertEqual(len(table), 6)
        self.assertNotIn("global_amount_recall", table.columns)

    @staticmethod
    def workloads(top_one_recall: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "requested_review_rate": [0.001, 0.005, 0.01],
                "recall": [0.05, 0.15, top_one_recall],
            }
        )

    @staticmethod
    def stability(passes: bool = True) -> pd.DataFrame:
        return pd.DataFrame({"passes_stability_floor": [passes] * 6})

    def test_smoke_cannot_progress_and_full_rules_are_deterministic(self):
        smoke = baseline_progression_decision(
            average_precision=0.5,
            prevalence=0.001,
            workloads=self.workloads(0.8),
            stability=self.stability(),
            smoke=True,
        )
        self.assertEqual(smoke["outcome"], "not_evaluated_smoke_test")
        weak = baseline_progression_decision(
            average_precision=0.004,
            prevalence=0.001,
            workloads=self.workloads(0.8),
            stability=self.stability(),
            smoke=False,
        )
        limited = baseline_progression_decision(
            average_precision=0.006,
            prevalence=0.001,
            workloads=self.workloads(0.2),
            stability=self.stability(),
            smoke=False,
        )
        candidate = baseline_progression_decision(
            average_precision=0.006,
            prevalence=0.001,
            workloads=self.workloads(0.3),
            stability=self.stability(),
            smoke=False,
        )
        unstable = baseline_progression_decision(
            average_precision=0.006,
            prevalence=0.001,
            workloads=self.workloads(0.3),
            stability=self.stability(False),
            smoke=False,
        )
        self.assertEqual(weak["outcome"], "stop_weak_signal")
        self.assertEqual(limited["outcome"], "discuss_limited_signal")
        self.assertEqual(candidate["outcome"], "candidate_for_one_time_test")
        self.assertEqual(unstable["outcome"], "stop_weak_signal")
        self.assertFalse(candidate["test_split_evaluated"])


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
