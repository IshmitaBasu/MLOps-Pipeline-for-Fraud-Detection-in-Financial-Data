# Oversampling Experiment Plan

## Purpose

The full Stage 08 comparison showed that sender-location history improves the
fixed Random Forest on validation data. Stage 09 then placed those retained
features behind a versioned Feast contract and verified that historical and
online retrieval agree.

The model is still not strong enough for its intended fraud-detection purpose.
The next question is:

> Can moderate oversampling of the fraud class improve validation performance
> without removing legitimate training examples?

This stage changes the imbalance treatment only. It does not compare model
families, perform open-ended tuning, evaluate the test split, or register a
model.

## Starting point

| Item | Frozen value |
| --- | --- |
| Dataset | 5,000,000 transactions |
| Split | Stratified random 70/15/15 split, random state 42 |
| Split hash | `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242` |
| Training rows | 3,500,000 |
| Validation rows | 750,000 |
| Closed test rows | 750,000 |
| Feature set | Ten original predictors plus the three retained sender-location features |
| Feature contract | `fraud_behaviour_features_v2` |
| Fixed model | Random Forest |
| Primary metric | Validation Average Precision |

The retained history definitions are:

- prior training transactions for the sender;
- prior training transactions for the sender-location pair; and
- whether a returning sender is using a location not previously seen in the
  training history.

They remain point-in-time features. The current event, equal-time events,
future events, validation events, test events, and fraud labels cannot update
their values. Sender and location identifiers remain lookup keys rather than
direct model inputs.

## Fixed Random Forest

Every strategy will use the Stage 08 Random Forest configuration:

| Parameter | Value |
| --- | ---: |
| Trees | 100 |
| Maximum depth | 12 |
| Minimum samples per leaf | 100 |
| Features considered per split | Square root |
| Random state | 42 |

The class-weighted reference uses `balanced_subsample`. All explicitly
resampled candidates use no class weighting so that weighting and resampling
are not combined in the first comparison.

## Frozen strategy comparison

| Strategy | Training treatment | Role |
| --- | --- | --- |
| `class_weight_reference` | All training rows; `balanced_subsample` | Main reference |
| `undersample_5_to_1` | Randomly retain five legitimate rows per fraud row | Existing approach reproduced with the retained features |
| `random_oversample_10_to_1` | Duplicate fraud rows until the legitimate-to-fraud ratio is 10:1 | Moderate oversampling candidate |
| `random_oversample_5_to_1` | Duplicate fraud rows until the legitimate-to-fraud ratio is 5:1 | Stronger oversampling candidate |
| `smotenc_10_to_1` | Generate mixed-type fraud examples until the ratio is 10:1 | Conditional synthetic candidate |
| `smotenc_5_to_1` | Generate mixed-type fraud examples until the ratio is 5:1 | Conditional synthetic candidate |

A 1:1 balanced dataset is excluded. It would create a much larger training set
and is not needed to answer whether moderate oversampling is useful.

## Preprocessing and resampling order

The split happens before any preprocessing or resampling. All learned
preprocessing values are fitted on the training partition only.

For a fair comparison, every strategy follows the same broad order:

1. reproduce the frozen split and remove the test partition from the
   development path;
2. construct or retrieve the retained point-in-time history features;
3. fit missing-value handling on training data only;
4. preserve categorical variables as categorical codes for the resampling
   step;
5. apply the selected imbalance treatment to training rows only;
6. one-hot encode the categorical values for the fixed Random Forest;
7. fit the model; and
8. evaluate the untouched validation partition.

Random oversampling duplicates complete training rows. SMOTENC may interpolate
numeric values, but it must select valid categorical levels rather than
interpolating one-hot columns. Ordinary SMOTE after one-hot encoding is
therefore prohibited.

## SMOTENC feasibility gate

SMOTENC will first run on the smoke sample. It may proceed to the complete
comparison only if all of the following checks pass:

- every generated categorical value maps to an existing training category;
- generated numeric values contain no new missing or infinite values;
- validation rows and validation prevalence remain unchanged;
- the output class counts match the requested ratio;
- repeating the run with seed 42 reproduces the same output checks; and
- the measured memory and runtime indicate that the full run is practical in
  the local thesis environment.

If the complete SMOTENC run fails because of measured memory or computational
limits, that result will be documented as a feasibility limitation. It will not
be replaced with ordinary SMOTE or silently removed after seeing other model
scores.

## Evaluation

Average Precision on the full validation partition is the primary selection
metric. The report will also include:

- ROC-AUC;
- precision, recall and F1 at a validation-derived maximum-F1 threshold;
- validation fraud prevalence and lift over prevalence;
- training time and validation inference time;
- training rows and fraud/legitimate counts before and after resampling; and
- estimated resampled-data memory where practical.

The maximum-F1 results provide context only because different thresholds can
produce very different alert volumes.

## Equal-workload and transaction-value evaluation

Each fitted strategy will be compared at fixed validation alert shares of 1%,
5% and 10%. Each scenario records:

- alerts generated;
- fraud cases captured;
- legitimate alerts;
- precision;
- fraud-count recall;
- fraudulent transaction value captured;
- missed fraudulent transaction value; and
- fraud-value recall.

The 5% scenario is the predeclared operational comparison point. It is a common
research workload, not a claim about a bank's actual investigation capacity.
Transaction amount is treated as a proxy for possible institutional exposure,
not as observed financial loss.

## Selection rule

An oversampling candidate may replace the class-weighted reference only when,
on the complete validation partition:

1. its Average Precision is higher;
2. its fraud-count recall at the 5% alert share is not lower;
3. its fraud-value recall at the 5% alert share is not lower; and
4. at least one of those two 5% recall measures is higher.

If more than one candidate passes, the candidate with the highest Average
Precision is selected. If scores differ by less than `0.0001`, higher 5%
fraud-value recall is the first tie-breaker and lower training time is the
second.

The 5:1 undersampling run is a sensitivity reference. It may be reported as the
strongest result, but the same progression rule still applies. If no candidate
passes, class weighting remains the fixed strategy for the later model-family
comparison.

Smoke scores cannot select a strategy or change this rule.

## Test isolation

The held-out test partition remains closed. The experiment runner may report
its row count from the frozen split manifest, but development functions must
not receive test features, labels, amounts, predictions, or metrics.

The test set will be evaluated once only, after the feature set, imbalance
strategy, model family, model configuration, and operating rule have all been
frozen from validation evidence and reviewed.

## MLflow evidence

Every candidate run will record:

- evidence scope: smoke or full data;
- frozen split and source hashes;
- feature service and feature definitions;
- sampler name, ratio, seed and library version;
- class counts before and after resampling;
- complete Random Forest and preprocessing parameters;
- validation predictive and equal-workload metrics;
- fraud-count and fraud-value results;
- fit and inference timings;
- environment and source snapshots; and
- a clear statement that the test split was not evaluated.

The summary run will contain the frozen rule, all candidate results, feasibility
exclusions, and the resulting imbalance-strategy decision. No run from this
stage will be registered as a production model.

## Automated safeguards

The implementation tests must prove that:

- the frozen split hash is reproduced;
- development APIs cannot accept a test frame;
- resampling changes training rows only;
- validation rows, labels, amounts and prevalence remain unchanged;
- random oversampling contains only duplicated training rows;
- SMOTENC categorical outputs are valid training categories;
- requested class ratios are achieved deterministically;
- point-in-time history definitions and leakage boundaries remain unchanged;
- all strategies use the same fixed Random Forest parameters apart from the
  declared class-weight difference; and
- candidate selection cannot read test metrics.

## Planned files

```text
10_oversampling_experiment_with_mlflow.py
oversampling_experiment_utils.py
tests/test_oversampling_experiment.py
Markdown files/10_oversampling_experiment_implementation_guide.md
Markdown files/10_oversampling_experiment_results.md
```

These files do not exist yet. The next implementation step is to create the
shared resampling safeguards and smoke workflow. No complete experiment should
be run before the automated tests and smoke checks pass.
