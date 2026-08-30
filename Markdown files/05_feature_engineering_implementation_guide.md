# Controlled Feature Experiment with MLflow

> **Use this document for:** understanding and running `05_feature_engineering_with_mlflow.py`. For the pre-run experimental design, see `05_feature_engineering_experiment_plan.md`. For the completed full-data findings, see `05_feature_engineering_results.md`.

## What this script is for

`05_feature_engineering_with_mlflow.py` checks whether a few feature ideas from the EDA improve the current Random Forest result.

The earlier experiment compared several models while keeping the original ten predictors unchanged. Random Forest produced the strongest validation PR-AUC, so this experiment keeps that model configuration fixed and changes only the input features. This separation matters because it lets me say whether a result changed because of the features rather than because I also changed the model.

In DSPM terms, this is an iteration from **DP07 Data Preparation** back into **DP10 Model Development** and **DP11 Model Evaluation**.

## Features being checked

The script calculates six candidate features in memory:

| Feature | Calculation | Reason for checking it |
| --- | --- | --- |
| `amount_log1p` | `log(1 + amount)` | The transaction amount is right-skewed, so a compressed view of it may be easier to use. |
| `transaction_hour` | Hour from `event_timestamp` | Fraud behaviour may vary over the day. |
| `transaction_day_of_week` | Monday = 0 through Sunday = 6 | Weekday patterns may carry information that the raw timestamp does not expose directly. |
| `transaction_month` | Month from `event_timestamp` | This checks for broad seasonal differences. |
| `transaction_is_weekend` | 1 on Saturday or Sunday, otherwise 0 | Weekend behaviour may differ from weekday behaviour. |
| `time_since_last_transaction_missing` | 1 when the original value is missing, otherwise 0 | The EDA found unusual target-related missingness, so this is treated as a sensitivity check rather than an automatically trusted feature. |

These calculations use only information from the current transaction. They do not use `is_fraud`, future transactions, or aggregates calculated from the full dataset. The persisted Feast v1 files are not rewritten.

## The five comparisons

| Group | Input columns |
| --- | --- |
| `original_v1` | The ten original cleaned predictors |
| `amount_v2` | Original predictors plus `amount_log1p` |
| `temporal_v2` | Original predictors plus the four timestamp-derived fields |
| `missingness_sensitivity_v2` | Original predictors plus the missing-value indicator |
| `combined_v2` | Original predictors plus all six candidate features |

The Random Forest configuration is identical for every group: 100 trees, maximum depth 12, minimum leaf size 100, square-root feature sampling, balanced-subsample class weighting, and random state 42.

## How selection works

Each group is trained on the same chronological training split and ranked using validation PR-AUC. During this comparison, the test split is not passed to the evaluator.

After all requested groups have finished, the script selects the group with the highest validation PR-AUC. It then trains that selected configuration once more and evaluates it on the test split. This means the feature-set decision is made without using test performance.

The selected run stores the fitted preprocessing-and-model pipeline in MLflow. The comparison runs store their parameters, feature definitions, validation metrics, plots, environment details, and source snapshots, but they do not store five unnecessary copies of the fitted model.

## Checks performed before training

The script stops with an error if:

- feature calculation changes the number of rows;
- a required candidate column is missing;
- a temporal or binary feature falls outside its expected range;
- `amount_log1p` contains a missing or infinite value; or
- a negative amount would make the chosen logarithmic transformation invalid.

The test file `tests/test_feature_engineering.py` also checks the calculations using known timestamps, amounts, and missing values.

## Smoke run

Run this first from the thesis workspace:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\05_feature_engineering_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

This run is only a technical check. It confirms that all groups train, the feature values make sense, the test split is used only after validation selection, and the expected information appears in MLflow. The resulting sample metrics are not thesis findings.

## Full experiment

After the smoke run has been inspected:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\05_feature_engineering_with_mlflow.py"
```

The full run uses the canonical Feast historical retrieval, all five million rows, the same 70/15/15 chronological split, and source-file hash validation.

The default MLflow experiment is:

```text
financial-fraud-feast-feature-engineering
```

## How I will interpret the result

The main result is the change in validation PR-AUC relative to the `original_v1` control from the same experiment. The script also keeps the frozen original-feature value of `0.0439598` in the run metadata so that the new experiment can be traced back to the earlier candidate-model comparison.

A small numerical increase will not automatically be called a meaningful improvement. I will also inspect precision, recall, F1, ROC-AUC, the tuned threshold, training time, inference time, and the operational number of transactions that would be flagged.

If none of the engineered groups gives a convincing validation improvement, keeping the original ten features is a valid result. The purpose of the experiment is to test the feature ideas fairly, not to force an engineered feature into the final model.

## What this script does not decide

This experiment does not perform hyperparameter tuning, compare new imbalance-handling methods, choose a cost-based operating threshold, register a champion model, or deploy an API. Those decisions come after the feature set has been selected.
