# 04 - Baseline modeling with MLflow

This note documents `04_baseline_modeling_with_mlflow.py`.

## Current structure

The file is organized as a cell-wise Python script using `# %%` markers. These markers allow notebook-style
section execution in compatible IDEs while preserving normal command-line execution through the existing
`main()` entry point.

## Purpose

This stage creates the first tracked model runs after raw EDA, minimal preprocessing, and advanced
post-cleaning EDA. It directly addresses the supervisor feedback that a baseline should be trained with
the currently available feature set before returning to the data phase for feature-engineering experiments.

The goal is not to produce the final best model yet. The goal is to create a fair reference point that later
models and engineered features can be compared against.

## Input

```text
gold_financial_fraud_detection_table.csv
```

The script expects the minimally cleaned 13-column gold table created by `02_data_pipeline_preprocessing.py`.

## Baseline feature set

The baseline uses only original cleaned predictors:

- `transaction_type`
- `merchant_category`
- `location`
- `device_used`
- `payment_channel`
- `amount`
- `time_since_last_transaction`
- `spending_deviation_score`
- `velocity_score`
- `geo_anomaly_score`

The following columns are excluded from predictors:

- `transaction_id`, because it is only for lineage;
- `event_timestamp`, because it is used for the time-aware split, not as a direct baseline feature;
- `is_fraud`, because it is the target.

Candidate engineered features such as `amount_log1p`, hour, weekday, month, and a missingness indicator are
not included in this baseline. They should be tested later as named MLflow runs.

## Split design

The data is sorted by `event_timestamp` and split in time order:

- 70% training
- 15% validation
- 15% test

This reflects the fraud-detection setting more realistically than a random split because future transactions
should be evaluated against models trained on older transactions.

## Model pipeline

Both the prior-only dummy control and logistic model use the same learned preprocessing, fitted only on the
training split:

- numeric median imputation;
- numeric standard scaling;
- categorical most-frequent imputation;
- one-hot encoding for categorical variables;
- a prior-only `DummyClassifier(strategy="prior")` non-skill baseline;
- a balanced logistic baseline trained with `SGDClassifier(loss="log_loss")`.

The models are intentionally simple and scalable. More complex candidate models are compared separately in
`05_candidate_model_comparison_with_mlflow.py`.

## Why this baseline is appropriate

There are two useful meanings of *baseline* in this study. A `DummyClassifier` using the training-set
class prior is the non-skill baseline: it shows what can be achieved without learning relationships between
the predictors and fraud. The current `SGDClassifier(loss="log_loss")` is the first predictive baseline. It
implements a regularized linear logistic model, scales to the five-million-row gold table, supports class
weighting, and produces fraud scores that can be evaluated at different decision thresholds. Its coefficients
also provide a comparatively interpretable reference for later, more complex models.

The EDA does not prove in advance that one model type is best. Instead, it motivates which candidate models
should be tested under the same split, feature set, and metrics:

| EDA or processing evidence | Modeling implication | Baseline or challenger justified |
| --- | --- | --- |
| Fraud represents only 3.5911% of transactions. | A majority prediction can achieve high accuracy while detecting no fraud. Accuracy is therefore not a suitable primary selection metric, and imbalance handling and threshold evaluation are required. | Use a prior/majority `DummyClassifier` as a non-skill control. Compare predictive models primarily using PR-AUC, then precision, recall, F1, ROC-AUC, and confusion-matrix counts. |
| The strongest absolute Pearson correlation between an original numeric predictor and the target is only 0.000370. | There is little evidence of a strong univariate linear relationship. A linear model remains necessary as a simple reference, but it should not be assumed to be sufficient. | Keep logistic SGD as the scalable, interpretable linear baseline and compare it with nonlinear challengers. |
| The EDA examines fraud lift across amount quantiles and a two-dimensional interaction surface between spending-deviation and geographic-anomaly scores. | Fraud may depend on thresholds or combinations of variables rather than additive linear effects. The plots motivate a controlled test; they do not by themselves demonstrate predictive improvement. | Test a constrained Decision Tree for an interpretable nonlinear comparison, followed by Random Forest and histogram-based gradient boosting for more flexible interactions. |
| Numeric variables have different units and ranges, while categorical variables require encoding. | Distance-based models are sensitive to scale, so raw distances would be dominated by large-scale variables. | Median-impute and standardize numeric features, and impute and one-hot encode categories before logistic SGD or optional k-NN. |
| The gold table contains five million rows and both one-hot categorical and continuous features. | Ordinary k-NN has expensive storage and prediction, and distance in a mixed high-dimensional one-hot space may not be meaningful. | Treat k-NN as an optional, explicitly sampled feasibility experiment rather than a default full-data model. |
| Weekly analysis shows temporal variation, and deployment would predict later transactions from earlier ones. | A random split could leak future distribution information and give an optimistic comparison. | Use the same chronological train, validation, and test partitions for every candidate model. |
| Missing `time_since_last_transaction` values occur in 17.93% of rows and only among non-fraud records in this dataset. | Missingness may contain predictive information but could also reflect synthetic-data generation bias. Adding it silently would make the original-feature baseline difficult to interpret. | Retain median imputation in the baseline, then test a missingness indicator as a separate, named sensitivity experiment. |
| Amount-quantile fraud lift and categorical fraud-rate differences are small in the full-data EDA. | Apparent descriptive differences may not produce useful out-of-time predictions. | Keep the original variables initially and test transformations such as `amount_log1p` separately rather than embedding them in every model run. |

Accordingly, the planned comparison has three levels:

1. **Non-skill reference:** `DummyClassifier(strategy="prior")`.
2. **Predictive reference:** the current balanced logistic SGD pipeline.
3. **Nonlinear challengers:** Random Forest and histogram-based gradient boosting as default tree-based
   candidates, with a constrained Decision Tree or sampled k-NN available only when a smaller sensitivity
   check is needed.

This ordering separates the questions "does the model learn anything?", "does a linear decision function
work?", and "do nonlinear thresholds and interactions add value?" Tree-based models do not require numeric
standardization, although their imputers and categorical encoders must still be fitted only on training data.
For a fair candidate-model comparison, all challengers must use the same original predictors and chronological
splits as the logistic baseline. Feature-engineering experiments should remain separate from model-comparison
experiments so that a performance change can be attributed to either the model or the feature set, rather
than both at once.

The validation split is used for hyperparameter and decision-threshold selection. The test split remains
untouched until a configuration has been selected. Because the positive class is rare, PR-AUC is the primary
ranking metric; precision, recall, F1, ROC-AUC, confusion-matrix counts, training time, and inference time
provide complementary evidence. A more complex model should replace the logistic baseline only when its
out-of-time improvement is meaningful relative to its computational and interpretability costs.

## MLflow tracking

The script logs:

- dataset name, location, size, modification time, and SHA-256 hash;
- feature-set name;
- split fractions and split summaries;
- excluded columns;
- postponed feature-engineering experiments;
- complete scalar model and preprocessing parameters;
- validation and test metrics;
- threshold selected on validation data;
- training and inference durations;
- PR, ROC, confusion-matrix, and threshold-trade-off figures;
- coefficients when supported;
- environment/package metadata and source-code snapshots;
- fitted scikit-learn pipeline with an input example and signature.

Tracked metrics include PR-AUC, ROC-AUC, accuracy, balanced accuracy, precision, recall, specificity,
F1-score, and confusion-matrix counts. PR-AUC remains the primary comparison metric.

## Previous smoke-run verification

A 50,000-row logistic smoke run completed successfully for the earlier single-model implementation.

```text
Experiment: financial-fraud-original-feature-baseline
Run ID: 371567dc16164bbebf3656958c109c6a
Validation PR-AUC: 0.031157
Tuned validation threshold: 0.15
Test PR-AUC: 0.029145
Test F1 at tuned threshold: 0.070369
```

These numbers are historical technical-verification results and should not be reported as final model
performance. The updated two-baseline workflow should be smoke-tested again before full execution.

## How to run

For a quick smoke run:

```bash
python "Code snippets/04_baseline_modeling_with_mlflow.py" --sample-rows 50000
```

For a quicker smoke run without hashing the complete source CSV:

```bash
python "Code snippets/04_baseline_modeling_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

For the full baseline:

```bash
python "Code snippets/04_baseline_modeling_with_mlflow.py"
```

To open the local MLflow UI:

```bash
masters_thesis\Scripts\mlflow.exe ui --workers 1 --backend-store-uri "sqlite:///D:/Germany/Documents/Magdeburg/Semester Documents/Sem 5/Thesis/Code snippets/mlflow_tracking.db"
```

The single-worker setting avoids Windows socket errors caused by MLflow's four-worker default.

Then open the URL printed by MLflow, usually:

```text
http://127.0.0.1:5000
```

## Important boundary

This script establishes the original-feature baseline. It does not test feature engineering yet. The next
script compares a compact set of candidate models with the same features. Later feature experiments should add one idea at a
time so that improvements can be attributed clearly.
