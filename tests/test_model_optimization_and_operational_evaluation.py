"""Tests for stage-06 model optimisation and validation-only tuning."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from fraud_modeling_utils import FEATURE_COLUMNS, RunResult
from model_optimization_utils import (
    apply_training_imbalance_strategy,
    build_random_forest_tuning_pipeline,
    random_forest_tuning_configurations,
    select_validation_candidate,
)


def load_stage_06_module():
    module_path = (
        PROJECT_DIR
        / "06_model_optimization_and_operational_evaluation_with_mlflow.py"
    )
    specification = importlib.util.spec_from_file_location(
        "model_optimization_and_operational_evaluation",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ModelOptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_stage_06_module()

    def test_tuning_search_is_fixed_unique_and_contains_reference(self) -> None:
        configurations = random_forest_tuning_configurations()

        self.assertEqual(
            list(configurations),
            [
                "rf_reference",
                "rf_trees_200",
                "rf_trees_300",
                "rf_depth_8",
                "rf_depth_16",
                "rf_leaf_50",
                "rf_leaf_250",
                "rf_features_half",
                "rf_combined_flexible",
            ],
        )
        parameter_sets = [
            tuple(configuration.parameters.items())
            for configuration in configurations.values()
        ]
        self.assertEqual(len(parameter_sets), len(set(parameter_sets)))
        self.assertEqual(
            dict(configurations["rf_reference"].parameters),
            {
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_leaf": 100,
                "max_features": "sqrt",
            },
        )

    def test_tuning_pipeline_uses_parameters_without_class_weight(self) -> None:
        configuration = random_forest_tuning_configurations()["rf_combined_flexible"]
        pipeline = build_random_forest_tuning_pipeline(
            configuration,
            random_state=42,
        )
        model = pipeline.named_steps["model"]

        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.max_depth, 16)
        self.assertEqual(model.min_samples_leaf, 50)
        self.assertEqual(model.max_features, 0.5)
        self.assertIsNone(model.class_weight)
        self.assertEqual(model.random_state, 42)

    def test_undersampling_changes_only_training_rows(self) -> None:
        train_df = pd.DataFrame(
            {
                "row_id": range(110),
                "is_fraud": [0] * 100 + [1] * 10,
            }
        )
        validation_df = pd.DataFrame(
            {"row_id": range(20), "is_fraud": [0] * 16 + [1] * 4}
        )
        test_df = validation_df.copy()
        validation_before = validation_df.copy(deep=True)
        test_before = test_df.copy(deep=True)

        first, metadata = apply_training_imbalance_strategy(
            train_df,
            "undersample_5_to_1",
            random_state=42,
        )
        second, _ = apply_training_imbalance_strategy(
            train_df,
            "undersample_5_to_1",
            random_state=42,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["is_fraud"].value_counts().to_dict(), {0: 50, 1: 10})
        self.assertEqual(metadata["effective_rows"], 60)
        pd.testing.assert_frame_equal(validation_df, validation_before)
        pd.testing.assert_frame_equal(test_df, test_before)

    def test_candidate_selection_rejects_test_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation metric"):
            select_validation_candidate(
                [{"model": "forbidden", "test_pr_auc": 0.99}],
                metric="test_pr_auc",
            )

    def test_tuning_cli_rejects_model_or_strategy_overrides(self) -> None:
        arguments = argparse.Namespace(
            mode="tuning",
            train_fraction=0.70,
            validation_fraction=0.15,
            models=["lightgbm"],
            imbalance_strategies=None,
            tuning_configurations=None,
        )

        with self.assertRaisesRegex(ValueError, "always uses Random Forest"):
            self.runner.validate_experiment_choices(arguments)

    def test_tuning_runner_cannot_receive_or_forward_a_test_frame(self) -> None:
        signature = inspect.signature(self.runner.run_tuning_experiments)
        self.assertNotIn("test_df", signature.parameters)
        self.assertNotIn("held_out_test_df", signature.parameters)

        train_df = pd.DataFrame({"is_fraud": [0, 0, 1, 1]})
        validation_df = pd.DataFrame(
            {
                **{feature: [0, 0, 1, 1] for feature in FEATURE_COLUMNS},
                "is_fraud": [0, 0, 1, 1],
            }
        )
        arguments = argparse.Namespace(
            random_state=42,
            tuning_configurations=["rf_reference"],
            sample_rows=50_000,
            train_fraction=0.70,
            validation_fraction=0.15,
        )
        imbalance_metadata = {
            "strategy": "random_undersampling_5_to_1",
            "sampling_applied": True,
            "original_rows": 4,
            "effective_rows": 4,
            "effective_majority_rows": 2,
            "effective_minority_rows": 2,
            "effective_fraud_rate": 0.5,
        }
        threshold_summary = pd.DataFrame(
            [
                {
                    "selection": "max_f1",
                    "threshold": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "alert_rate": 0.5,
                }
            ]
        )
        run_result = RunResult(
            run_id="candidate-run",
            validation_pr_auc=0.05,
            tuned_threshold=0.5,
            validation_metrics={
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "roc_auc": 0.5,
            },
            test_metrics=None,
        )
        summary_run = SimpleNamespace(info=SimpleNamespace(run_id="summary-run"))

        with (
            patch.object(
                self.runner,
                "apply_training_imbalance_strategy",
                return_value=(train_df, imbalance_metadata),
            ),
            patch.object(
                self.runner,
                "build_random_forest_tuning_pipeline",
                return_value=object(),
            ),
            patch.object(
                self.runner,
                "fit_evaluate_and_log",
                return_value=run_result,
            ) as fit_and_log,
            patch.object(
                self.runner,
                "predict_scores",
                return_value=(np.array([0.1, 0.2, 0.8, 0.9]), 0.5, "probability"),
            ),
            patch.object(
                self.runner,
                "build_threshold_summary",
                return_value=threshold_summary,
            ),
            patch.object(self.runner, "log_operational_threshold_evidence"),
            patch.object(
                self.runner.mlflow,
                "start_run",
            ) as start_run,
            patch.object(self.runner.mlflow, "set_tags"),
            patch.object(self.runner.mlflow, "log_dict"),
            patch.object(self.runner.mlflow, "log_params"),
            patch.object(self.runner.mlflow, "log_metrics"),
        ):
            start_run.return_value.__enter__.return_value = summary_run
            selected, summary_run_id = self.runner.run_tuning_experiments(
                args=arguments,
                train_df=train_df,
                validation_df=validation_df,
                dataset_info={},
                feature_lineage={},
                source_files=[],
                source_rows=4,
                experiment_rows=4,
                evidence_scope="smoke_test",
            )

        self.assertIsNone(fit_and_log.call_args.kwargs["test_df"])
        self.assertIsNone(run_result.test_metrics)
        self.assertEqual(selected["model"], "rf_reference")
        self.assertIsNone(selected["relative_improvement_over_5_to_1_reference"])
        self.assertFalse(selected["reference_comparison_valid"])
        self.assertEqual(summary_run_id, "summary-run")


if __name__ == "__main__":
    unittest.main()
