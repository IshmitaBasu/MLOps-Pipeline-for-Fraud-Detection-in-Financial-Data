"""Check that validation reporting preserves threshold and confusion evidence."""

# %% Imports
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from validation_evaluation_utils import fit_and_evaluate_validation


# %% Reporting tests
class ValidationReportingTests(unittest.TestCase):
    def test_saved_metrics_match_the_selected_threshold_and_confusion_counts(self) -> None:
        frame = pd.DataFrame({"feature": [1, 2, 3, 4], "is_fraud": [0, 0, 1, 1], "amount": [10, 20, 30, 40]})
        pipeline = Pipeline([("preprocessing", SimpleImputer()), ("model", DummyClassifier(strategy="prior"))])
        scores = np.array([0.1, 0.8, 0.7, 0.6])
        with patch("validation_evaluation_utils.predict_scores", return_value=(scores, 0.5, "probability")):
            metrics, _, _ = fit_and_evaluate_validation(
                pipeline, frame, frame, feature_columns=["feature"], target_column="is_fraud"
            )

        self.assertAlmostEqual(metrics["validation_threshold_at_max_f1"], 0.6)
        self.assertEqual(metrics["validation_true_positives_at_max_f1"], 2)
        self.assertEqual(metrics["validation_false_positives_at_max_f1"], 1)
        self.assertEqual(metrics["validation_false_negatives_at_max_f1"], 0)
        self.assertEqual(metrics["validation_true_negatives_at_max_f1"], 1)
        self.assertAlmostEqual(metrics["validation_accuracy_at_max_f1"], 0.75)
        self.assertAlmostEqual(metrics["validation_balanced_accuracy_at_max_f1"], 0.75)
        self.assertAlmostEqual(metrics["validation_specificity_at_max_f1"], 0.5)
        self.assertAlmostEqual(metrics["validation_alert_rate_at_max_f1"], 0.75)
        self.assertAlmostEqual(metrics["validation_f1"], 0.8)


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()
