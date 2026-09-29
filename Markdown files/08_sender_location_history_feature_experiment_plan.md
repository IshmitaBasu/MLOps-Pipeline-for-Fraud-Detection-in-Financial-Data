# Sender-Location History Feature Experiment Plan

## Purpose

The Stage 07 full-data diagnostic found that sender-location history was the
only examined entity relationship to pass the predefined feasibility gate. A
previous sender-location relationship existed for 20.844% of training rows,
and the fraud rates of new and returning combinations differed by 15.638% in
relative terms.

That association does not prove that the information improves prediction. This
controlled experiment will answer a narrower question:

> Does adding a small group of point-in-time sender-location history features
> improve the fixed Random Forest over `original_v1` on validation data?

## Fixed comparison

Only the feature columns may change. Both runs will use the same:

- five-million-row source dataset;
- frozen stratified random 70/15/15 split;
- random state 42;
- training and validation transactions;
- preprocessing for the ten original features;
- class-weighted Random Forest;
- 100 trees, maximum depth 12, minimum leaf size 100, and square-root feature sampling;
- validation metrics and fixed alert volumes; and
- MLflow experiment structure.

The control is `original_v1`. The candidate is
`sender_location_history_v1`, which contains `original_v1` plus three numeric
features.

## Feature definitions

| Feature | Definition |
| --- | --- |
| `sender_prior_transaction_count` | Number of strictly earlier training transactions belonging to the sender |
| `sender_location_prior_transaction_count` | Number of strictly earlier training transactions with the same sender and location |
| `is_new_location_for_returning_sender` | 1 when the sender has earlier training history but the sender-location pair does not; otherwise 0 |

The third feature distinguishes a genuinely new location for an existing sender
from the first transaction of a completely new sender.

The current transaction, transactions with the same timestamp, future events,
validation events, test events, and fraud labels cannot update these features.
For a validation transaction, only strictly earlier training transactions may
contribute to its history. Raw sender identifiers are aggregation keys and are
not model inputs.

## Split and test isolation

The complete run must reproduce Stage 07 split hash:

```text
e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242
```

If the hash differs, the full experiment must stop. Test rows will be counted in
the split manifest, then removed before sender keys are joined or historical
features are calculated. The comparison function accepts training and
validation frames only. No model will be registered or promoted in this stage.

## Evaluation

The primary metric is validation Average Precision. Precision, recall, F1 and
ROC-AUC at the validation-derived maximum-F1 threshold provide supporting
context.

Both models will also be compared at identical alert volumes of 1%, 5% and 10%
of validation transactions. Each workload scenario records:

- number of alerts;
- fraud cases captured;
- legitimate alerts;
- precision;
- fraud-count recall;
- fraudulent transaction value captured; and
- fraud-value recall.

Transaction amount is treated as a proxy for possible institutional exposure,
not as the bank's actual loss.

## Predeclared progression rule

The candidate may proceed to later model experiments only if all of the
following are true on the full validation partition:

1. validation Average Precision is higher than the control;
2. at the 5% alert volume, fraud-count recall is not lower;
3. at the 5% alert volume, fraud-value recall is not lower; and
4. at least one of the two 5% workload recall measures is higher.

A smoke result cannot satisfy this rule. If the full candidate fails, the three
features will be documented as a negative experiment and `original_v1` will be
retained for the following oversampling and model-family comparisons.

Passing permits the feature group to be used in later validation experiments.
It does not permit test evaluation, Feast online serving claims, model
registration, or deployment.
