# Feature Experiment Plan

> **Use this document for:** the question, hypotheses, feature groups, controls, and decision rules defined before running the experiment. For execution instructions, see `05_feature_engineering_implementation_guide.md`. For the completed full-data findings, see `05_feature_engineering_results.md`.

## Why I am doing this

So far, I have trained the models using the ten original cleaned features. The Random Forest gave the best validation PR-AUC, so I am using it as the reference model for the next step.

Before tuning the model, I want to check whether a few features suggested by the EDA add useful information. I will test them in separate groups instead of adding everything at once. This should make it easier to see which changes help and which do not.

The main question is:

> Do the engineered features improve the Random Forest validation PR-AUC beyond the original-feature benchmark of 0.0439598?

## Starting point

The comparison will start from the frozen original-feature benchmark:

```text
Data source: Feast historical retrieval
Feature version: v1
Rows: 5,000,000
Split: chronological 70/15/15
Random state: 42
Reference model: random_forest_depth_12
Validation PR-AUC: 0.0439598
```

## Features I want to test

### Log-transformed amount

I will create:

```text
amount_log1p = log(1 + amount)
```

Transaction amount is right-skewed. The log transformation may make differences between ordinary and very large transactions easier for the model to use.

### Time-related features

I will derive the following values from the transaction timestamp:

```text
transaction_hour
transaction_day_of_week
transaction_month
transaction_is_weekend
```

These features test whether fraud behaviour changes by time of day, weekday, month, or weekend. They will be calculated only from the timestamp of the current transaction. No future transaction information will be used.

### Missing-value indicator

I will create:

```text
time_since_last_transaction_missing
```

This value will be `1` when `time_since_last_transaction` is missing and `0` otherwise. The EDA showed that this missingness has an unusual relationship with the target, so I will treat it as a separate sensitivity experiment. I will not automatically include it in the final feature set even if it improves one metric.

## Experiment groups

| Experiment name              | Features                                                |
| ---------------------------- | ------------------------------------------------------- |
| `original_v1`                | The ten original cleaned predictors                     |
| `amount_v2`                  | Original predictors plus `amount_log1p`                 |
| `temporal_v2`                | Original predictors plus the four time-related features |
| `missingness_sensitivity_v2` | Original predictors plus the missing-value indicator    |
| `combined_v2`                | Original predictors plus all engineered features        |

The separate groups are important because a combined result alone would not show which type of feature caused the change.

## What I will keep unchanged

For a fair comparison, I will not change:

- the source rows retrieved through Feast;
- the chronological train, validation, and test boundaries;
- random state 42;
- the Random Forest configuration;
- the imputation and categorical-encoding strategy;
- the class-weight strategy;
- the evaluation functions; or
- the primary selection metric.

Only the feature set will change during this experiment. Model tuning will happen later, after I have selected the strongest feature set.

## Leakage and data checks

Before training, I will confirm that:

- `is_fraud` is not used to create any feature;
- `fraud_type` remains excluded;
- the engineered features use only information available in the current transaction;
- the number of rows and transaction IDs do not change;
- `event_timestamp` itself is not passed to the model;
- no aggregate based on the full dataset or target is created;
- the original Feast v1 Parquet files remain unchanged; and
- the missingness indicator is reported separately because it may reflect a dataset-specific pattern.

I will also check the expected ranges:

```text
transaction_hour: 0 to 23
transaction_day_of_week: 0 to 6
transaction_month: 1 to 12
transaction_is_weekend: 0 or 1
time_since_last_transaction_missing: 0 or 1
```

## How I will run the experiments

I will first run every feature group on a 50,000-row smoke sample. These runs are only for checking the code, feature values, pipeline execution, and MLflow logging. I will not use the smoke-run metrics as final evidence.

After the smoke tests pass, I will run the feature groups on the full dataset using the unchanged Random Forest configuration. Every run, including an unsuccessful one, will remain in MLflow.

The order will be:

1. Check the feature calculations and row counts.
2. Run the smoke experiments.
3. Check the MLflow parameters, metrics, artifacts, and feature names.
4. Run the full-data feature experiments.
5. Compare the feature groups using validation PR-AUC.
6. Select the strongest feature set using validation evidence.
7. Evaluate the selected feature set on the test split.
8. Start model tuning only after the feature comparison is complete.

## How I will judge the result

The main comparison metric is validation PR-AUC. For each feature group, I will report:

```text
absolute change = engineered validation PR-AUC - 0.0439598

relative change = (engineered validation PR-AUC / 0.0439598 - 1) x 100
```

I will also examine precision, recall, F1, ROC-AUC, the confusion matrix, training time, and inference time. These secondary results will help identify whether a small PR-AUC improvement comes with an unreasonable computational or operational cost.

I will not describe a very small numerical difference as a meaningful improvement without checking whether the result is stable. If none of the feature groups improves the benchmark convincingly, I will keep the original feature set and report that outcome rather than selecting an engineered set simply because it was tested.

## What is outside this experiment

This experiment does not include:

- hyperparameter tuning;
- a new imbalance-handling method;
- cost-based threshold selection;
- champion model registration;
- FastAPI serving;
- monitoring; or
- deployment.

Those steps will be handled separately after the feature comparison.
