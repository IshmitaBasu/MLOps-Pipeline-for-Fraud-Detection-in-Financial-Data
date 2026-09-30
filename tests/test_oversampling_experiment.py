"""Safeguards for the Stage 10 oversampling experiment."""

# %% Imports
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oversampling_experiment_utils import (
    MODEL_NUMERIC_FEATURES,
    StrategyConfig,
    build_fixed_random_forest,
    categorical_indices,
    categorical_levels_from_training,
    dataframe_fingerprint,
    frozen_strategies,
    resample_training_data,
    select_validation_strategy,
    validate_full_run_identity,
)


# %% Deterministic encoded fixture
def encoded_fixture(legitimate_rows: int = 100, fraud_rows: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rows = legitimate_rows + fraud_rows
    rng = np.random.default_rng(7)
    numeric = rng.normal(size=(rows, len(MODEL_NUMERIC_FEATURES)))
    categorical = np.column_stack(
        [np.arange(rows) % 3, np.arange(rows) % 4, np.arange(rows) % 2, np.arange(rows) % 5, np.arange(rows) % 3]
    ).astype(float)
    features = np.column_stack([numeric, categorical])
    labels = np.concatenate([np.zeros(legitimate_rows, dtype=np.int8), np.ones(fraud_rows, dtype=np.int8)])
    return features, labels


def strategy(name: str) -> StrategyConfig:
    return next(item for item in frozen_strategies() if item.name == name)


# %% Strategy and resampling tests
class FrozenStrategyTests(unittest.TestCase):
    def test_strategy_order_and_ratios_are_frozen(self) -> None:
        configurations = frozen_strategies()
        self.assertEqual(
            [item.name for item in configurations],
            [
                "class_weight_reference",
                "undersample_5_to_1",
                "random_oversample_10_to_1",
                "random_oversample_5_to_1",
                "smotenc_10_to_1",
                "smotenc_5_to_1",
            ],
        )
        self.assertEqual(configurations[0].class_weight, "balanced_subsample")
        self.assertTrue(all(item.class_weight is None for item in configurations[1:]))

    def test_random_oversampling_duplicates_training_rows_only(self) -> None:
        features, labels = encoded_fixture()
        result = resample_training_data(features, labels, strategy("random_oversample_5_to_1"))
        self.assertEqual(result.diagnostics["class_counts_after"]["legitimate"], 100)
        self.assertEqual(result.diagnostics["class_counts_after"]["fraud"], 20)
        self.assertIsNotNone(result.sample_indices)
        np.testing.assert_array_equal(result.features, features[result.sample_indices])

    def test_undersampling_reaches_five_to_one(self) -> None:
        features, labels = encoded_fixture()
        result = resample_training_data(features, labels, strategy("undersample_5_to_1"))
        self.assertEqual(result.diagnostics["class_counts_after"]["legitimate"], 25)
        self.assertEqual(result.diagnostics["class_counts_after"]["fraud"], 5)

    def test_smotenc_is_deterministic_and_keeps_valid_categories(self) -> None:
        features, labels = encoded_fixture(legitimate_rows=60, fraud_rows=6)
        config = strategy("smotenc_5_to_1")
        first = resample_training_data(features, labels, config)
        second = resample_training_data(features, labels, config)
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.labels, second.labels)
        for index in categorical_indices():
            self.assertTrue(np.isin(first.features[:, index], np.unique(features[:, index])).all())
        self.assertEqual(first.diagnostics["class_counts_after"]["fraud"], 12)

    def test_fixed_model_changes_only_declared_class_weight(self) -> None:
        reference = build_fixed_random_forest(strategy("class_weight_reference"))
        oversampled = build_fixed_random_forest(strategy("random_oversample_5_to_1"))
        reference_model = reference.named_steps["model"]
        oversampled_model = oversampled.named_steps["model"]
        compared = ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "random_state"]
        for parameter in compared:
            self.assertEqual(getattr(reference_model, parameter), getattr(oversampled_model, parameter))
        self.assertEqual(reference_model.class_weight, "balanced_subsample")
        self.assertIsNone(oversampled_model.class_weight)

    def test_fixed_category_vocabulary_supports_fit_and_validation(self) -> None:
        features, labels = encoded_fixture(legitimate_rows=120, fraud_rows=12)
        levels = categorical_levels_from_training(features)
        pipeline = build_fixed_random_forest(strategy("class_weight_reference"), categorical_levels=levels)
        pipeline.fit(features, labels)
        validation = features[:3].copy()
        validation[0, categorical_indices()[0]] = -1
        probabilities = pipeline.predict_proba(validation)[:, 1]
        self.assertEqual(len(probabilities), 3)
        self.assertTrue(np.isfinite(probabilities).all())


# %% Validation and selection safeguards
class SelectionSafeguardTests(unittest.TestCase):
    @staticmethod
    def result(
        name: str, pr_auc: float, count_recall: float, value_recall: float, training_seconds: float = 10.0
    ) -> dict[str, float | str]:
        return {
            "strategy": name,
            "validation_pr_auc": pr_auc,
            "workload_5pct_fraud_count_recall": count_recall,
            "workload_5pct_fraud_value_recall": value_recall,
            "training_seconds": training_seconds,
        }

    def test_smoke_results_cannot_select_a_strategy(self) -> None:
        decision = select_validation_strategy(
            [self.result("class_weight_reference", 0.04, 0.05, 0.04)], evidence_scope="smoke_test"
        )
        self.assertFalse(decision["decision_eligible"])
        self.assertIsNone(decision["selected_strategy"])

    def test_selection_rejects_test_metrics(self) -> None:
        invalid = self.result("class_weight_reference", 0.04, 0.05, 0.04)
        invalid["test_pr_auc"] = 0.99
        with self.assertRaisesRegex(ValueError, "test evidence"):
            select_validation_strategy([invalid], evidence_scope="full_data")

    def test_full_selection_applies_operational_rule_and_tie_break(self) -> None:
        results = [
            self.result("class_weight_reference", 0.0400, 0.050, 0.040),
            self.result("candidate_a", 0.04500, 0.055, 0.045, 8.0),
            self.result("candidate_b", 0.04505, 0.056, 0.050, 12.0),
            self.result("candidate_bad_value", 0.0500, 0.060, 0.030, 4.0),
        ]
        decision = select_validation_strategy(results, evidence_scope="full_data")
        self.assertEqual(decision["selected_strategy"], "candidate_b")

    def test_validation_fingerprint_detects_mutation(self) -> None:
        validation = pd.DataFrame({"transaction_id": ["T1", "T2"], "is_fraud": [0, 1]})
        before = dataframe_fingerprint(validation)
        unchanged = validation.copy(deep=True)
        self.assertEqual(before, dataframe_fingerprint(unchanged))
        unchanged.loc[0, "is_fraud"] = 1
        self.assertNotEqual(before, dataframe_fingerprint(unchanged))

    def test_full_run_requires_complete_data_and_frozen_hash(self) -> None:
        frozen_hash = "frozen"
        validate_full_run_identity(
            source_rows=5_000_000, observed_split_hash=frozen_hash, expected_split_hash=frozen_hash
        )
        with self.assertRaisesRegex(ValueError, "5,000,000"):
            validate_full_run_identity(
                source_rows=50_000, observed_split_hash=frozen_hash, expected_split_hash=frozen_hash
            )
        with self.assertRaisesRegex(ValueError, "frozen Stage 07 split"):
            validate_full_run_identity(
                source_rows=5_000_000, observed_split_hash="changed", expected_split_hash=frozen_hash
            )


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
