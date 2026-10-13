# Original-feature benchmark v1

This document fixes the reference point for all later feature-engineering and tuning experiments. “Frozen” means that the data source, original feature set, chronological split, random seed, selection metric, and benchmark model are recorded and should remain unchanged when a new experiment claims an improvement.

The benchmark protocol was frozen on 29 July 2026 and reproduced with the consolidated baseline-comparison workflow on 9 August 2026. It is a research baseline, not a registered champion and not a deployment candidate. The canonical MLflow experiment is <code>financial-fraud-feast-original-feature-baseline-comparison</code>, with comparison summary run <code>f1fc1ab8932143b389f04c4fabc09c7e</code>.

## Data and feature lineage

The modeling table was reconstructed through Feast using feature service <code>fraud_model_features_v1</code>. It contains 5,000,000 transactions, including 179,553 fraud cases, for an overall fraud rate of 3.59106%. The transaction entity is joined through <code>transaction_id</code>, event order is defined by <code>event_timestamp</code>, and <code>is_fraud</code> is the supervised target. All experiments use random state 42.

The source artifacts are identified by these hashes:

~~~text
Feature table SHA-256:
db1c01acc7c6425113a4d8d41df753f4c9fc49b5c1a0c37479cc258ac4e5354c

Label table SHA-256:
72e3a80f04d6d1bf917bd88588b7bc9820c39c703acdd8496ea136ec7e8f8e78
~~~

The benchmark uses five categorical and five numerical predictors:

| Categorical predictors | Numerical predictors |
| --- | --- |
| <code>transaction_type</code> | <code>amount</code> |
| <code>merchant_category</code> | <code>time_since_last_transaction</code> |
| <code>location</code> | <code>spending_deviation_score</code> |
| <code>device_used</code> | <code>velocity_score</code> |
| <code>payment_channel</code> | <code>geo_anomaly_score</code> |

The transaction identifier remains available for lineage, and the event timestamp determines the split. Neither enters the model. The fraud label is also excluded from the predictors, as is the target-derived <code>fraud_type</code> field.

## Chronological evaluation split

Transactions are ordered by event time and divided into 70% training, 15% validation, and 15% test data.

| Split | Rows | Fraud cases | Fraud rate | Time range |
| --- | ---: | ---: | ---: | --- |
| Train | 3,500,000 | 125,453 | 0.035844 | 2023-01-01 00:09:26 UTC to 2023-09-14 00:46:55 UTC |
| Validation | 750,000 | 27,019 | 0.036025 | 2023-09-14 00:46:56 UTC to 2023-11-07 17:06:46 UTC |
| Test | 750,000 | 27,081 | 0.036108 | 2023-11-07 17:07:02 UTC to 2024-01-01 22:58:30 UTC |

Imputation, encoding, scaling, class weighting, and model fitting are learned from training data only. Each decision threshold is chosen by maximizing F1 on validation data and is then applied unchanged to the test period.

## Model comparison

The benchmark selection rule is the highest validation PR-AUC among full-data baseline configurations. Test metrics are included to describe out-of-time behavior; they were not used by the comparison script to choose the model.

| Model | MLflow run ID | Validation PR-AUC | Test PR-AUC | Threshold | Test precision | Test recall | Test F1 | Test ROC-AUC | FP | FN | Training time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy prior | <code>9e005ddd3e46460c9bad2affb31474ac</code> | 0.036025 | 0.036108 | 0.035844 | 0.036108 | 1.000000 | 0.069699 | 0.500000 | 722,919 | 0 | 14.62 s |
| Logistic SGD | <code>fb2029b4be274f94a1f101c6a4864b5b</code> | 0.032996 | 0.031627 | 0.427277 | 0.036089 | 0.998634 | 0.069660 | 0.459112 | 722,332 | 37 | 28.71 s |
| Random Forest, depth 12 | <code>06c112748e634444879b3892089fb310</code> | **0.043960** | **0.044113** | 0.428590 | **0.043974** | **1.000000** | **0.084244** | **0.594210** | 588,754 | 0 | 1,114.06 s |
| Histogram Gradient Boosting | <code>d1b2e7dafd0349ee8b06fa44625199e8</code> | 0.043842 | 0.044044 | 0.272284 | 0.043971 | 0.999668 | 0.084237 | 0.593752 | 588,608 | 9 | 84.09 s |

## Why Random Forest is the reference benchmark

Random Forest achieved the highest validation PR-AUC, 0.0439598, and is therefore frozen as the original-feature benchmark. Relative to the validation fraud prevalence of 0.0360253, this represents an improvement of approximately 22.0%:

~~~text
(0.0439598 / 0.0360253 − 1) × 100 = 22.025%
~~~

The result should not be overstated. Histogram Gradient Boosting was only about 0.000118 lower in validation PR-AUC and trained in roughly 84 seconds rather than 1,114 seconds. Random Forest wins under the predefined selection rule, while Histogram Gradient Boosting remains the more computationally efficient near-equivalent. Later decisions should consider both predictive and operational cost.

## Why this model is not ready for deployment

At the validation-F1-selected threshold, Random Forest classified 615,835 of the 750,000 test transactions as fraud. This is approximately 82.1% of the test period:

~~~text
(27,081 true positives + 588,754 false positives) / 750,000 × 100 = 82.111%
~~~

Recall reached 1.0, but only because the threshold produced 588,754 false alerts. Such an alert volume would overwhelm a realistic investigation process. The benchmark therefore demonstrates a modest improvement in ranking quality, not a usable operating policy.

Later work needs to improve the separation of fraud and legitimate transactions and choose thresholds using explicit business costs or investigation capacity. Maximizing validation F1 alone is not sufficient for operational deployment.

## Rules for later experiments

A result should be compared with this benchmark only when it:

1. uses the same Feast v1 source rows and chronological partitions;
2. retains random state 42;
3. changes the feature set or model configuration in a documented, isolated way;
4. uses validation PR-AUC as the primary comparison metric;
5. selects models and thresholds from training and validation evidence rather than test results;
6. reports the already observed test benchmark transparently without optimizing against it;
7. logs feature definitions, parameters, metrics, source snapshots, and lineage in MLflow; and
8. passes a smoke run before the complete five-million-row experiment.

These controls make the benchmark useful: future improvements can be attributed to a documented change rather than to an altered dataset or evaluation protocol.
