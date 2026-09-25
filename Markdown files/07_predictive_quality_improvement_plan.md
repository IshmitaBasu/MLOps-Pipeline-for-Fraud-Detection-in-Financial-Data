# Plan for Predictive-Quality Improvement

## Purpose

Stage 06 showed that the existing pipeline is reproducible, but it did not produce a model that is useful enough for fraud detection. The strongest tuning result, `rf_leaf_50`, reached a validation Average Precision of 0.044293 and only slightly improved the previous Random Forest reference. Its fraud ranking and fixed-workload results remained weak.

Following supervisor feedback in September 2026, the modelling stage is reopened before any final test evaluation or model registration. This stage will investigate whether a different split protocol, better behavioural information, oversampling, or another model family can provide a meaningful improvement. At the same time, the model-independent parts of the serving and monitoring architecture can be designed.

This document is the pre-run plan. Results must be recorded in a separate results document without rewriting the original decisions after scores are known.

## Starting point

| Item | Current position |
| --- | --- |
| Dataset | 5,000,000 transactions with 179,553 fraud cases |
| Overall fraud rate | 3.59106% |
| Existing feature set | `original_v1`, containing ten transaction-level predictors |
| Previous protocol | Chronological 70% training, 15% validation, 15% test |
| Previous validation reference | Random Forest with class weighting, Average Precision 0.0439598 |
| Previous strongest tuning score | `rf_leaf_50`, Average Precision 0.0442925 |
| Previous operational finding | Precision remained close to prevalence; maximum-F1 operation alerted on about 82% of validation transactions |
| Current model status | Research reference only; not accepted, registered, or promoted |
| Test status | No new candidate may use test evidence during this stage |

The earlier chronological results remain valid records of those experiments. They will not be compared directly with scores from the revised random-split protocol because the evaluation populations are different.

## Questions for this stage

The experiments will answer five questions:

1. Does a stratified random split reveal stronger learnable relationships than the earlier chronological split?
2. Do account, counterparty, device, IP, location, or transaction-history fields contain enough repeated structure to justify behavioural feature engineering?
3. Does oversampling the fraud class improve ranking while preserving the legitimate examples removed by undersampling?
4. Do XGBoost, CatBoost, or a scalable linear SVM improve upon the revised Random Forest reference?
5. Can any improvement be shown not only in Average Precision, but also in fraud-count and fraud-value capture at a documented review workload?

KNN will be treated as a sampled feasibility study rather than a likely full-data candidate because its storage and inference costs grow with the training set.

## Revised split protocol

Stage 07 will use one deterministic stratified random split:

| Partition | Share | Purpose |
| --- | ---: | --- |
| Training | 70% | Fit preprocessing, resampling, feature transformations, and models |
| Validation | 15% | Compare features, imbalance strategies, models, and thresholds |
| Test | 15% | Remain closed until one candidate and operating rule are approved |

The split will use random state 42 and stratification by `is_fraud`. Transaction identifiers assigned to each partition, row counts, fraud counts, fraud rates, and a stable split hash will be recorded so that every Stage 07 experiment uses exactly the same rows.

Changing the split creates a new experimental protocol. The Dummy, linear, Random Forest, and other reference values required for Stage 07 must therefore be reproduced under this split. Previous chronological results will be discussed as robustness and temporal-generalisation evidence, not used as like-for-like rankings.

Historical features must still respect event time. They may use transaction attributes observed before the current event, but they must not use future transactions or past fraud labels. A random modelling split does not permit future information to enter a row's feature values.

## Test-isolation rules

The Stage 07 test partition will not be passed to diagnostic, feature-selection, resampling, model-screening, or tuning functions. During development, scripts may report only its row count and class count from the frozen split manifest. They must not calculate test predictions, metrics, thresholds, feature associations, or cost results.

A final test path will be implemented separately only after:

1. one feature set is frozen;
2. one imbalance strategy is frozen;
3. one model configuration is frozen;
4. one validation-derived operating rule is frozen;
5. the validation evidence is documented; and
6. the supervisor has reviewed the decision.

## Experiment sequence

### Phase 1: reproduce references under the random split

The first run will establish the new comparison baseline using `original_v1`. It will include:

- Dummy prior classifier;
- Logistic SGD;
- the fixed Random Forest reference;
- Histogram Gradient Boosting; and
- LightGBM.

These runs answer only how the established models behave under the revised split. They will not reuse thresholds or scores from the chronological experiments.

The reference report will include validation Average Precision, ROC-AUC, precision-recall curves, training time, inference time, probability or decision-score distributions, and class prevalence.

### Phase 2: diagnose the available signal

Before creating historical features, the training partition will be examined for:

- repetition counts for sender accounts, receiver accounts, devices, IP addresses, and sender-receiver pairs;
- the share of transactions for which a prior history would exist;
- changes in feature and fraud distributions across time periods;
- categories that are absent from earlier periods or unusually concentrated in fraud;
- overlap between fraud and legitimate prediction-score distributions; and
- the stability of simple feature-target associations across training subperiods.

This phase is diagnostic. It must not inspect the test partition or select a model from test evidence.

### Phase 3: behavioural-feature feasibility gate

Historical feature engineering will proceed only if the diagnostic work shows that identifiers repeat often enough and that simple, causally calculated candidates show a stable relationship with fraud.

Possible candidates include:

- number of previous sender transactions;
- sender transaction count in a recent time window;
- sender historical mean or median amount;
- current amount relative to sender history;
- time since the sender's previous transaction;
- number of devices, locations, or IP addresses previously used by a sender;
- whether the current device, location, IP address, or receiver is new for the sender;
- previous sender-receiver interaction count; and
- receiver, device, or IP transaction frequency calculated from earlier events.

All values must be calculated in timestamp order and shifted so that the current transaction cannot contribute to its own history. Target labels and future events are prohibited. Raw high-cardinality identifiers will be retained only as keys for aggregation and Feast lookup, not passed directly to a model.

At most two small feature groups will proceed to validation modelling. Each will first be tested with a fixed reference model so that the feature effect is not confused with a simultaneous model change. If repetition, coverage, stability, or pilot performance is weak, the result will be documented and the full behavioural feature pipeline will not be built.

### Phase 4: Feast serving decision

If a behavioural feature group provides useful and stable validation evidence, it will receive a named version and a documented feature contract. The implementation will then separate two responsibilities:

1. a point-in-time feature pipeline calculates and updates the aggregates; and
2. Feast provides consistent historical retrieval for training and online retrieval for inference.

The same feature names, types, definitions, and versions must be used in training and serving. The serving design will measure feature-fetch latency and verify that a sample transaction receives the same values through the offline and online paths where the timestamps and source state are equivalent.

If the candidate features do not help, no online feature-store claim will be made merely for architectural completeness. The negative feasibility result will be retained.

### Phase 5: oversampling comparison

The oversampling experiment will keep the feature set and Random Forest configuration fixed. It will compare:

- class weighting;
- the existing 5:1 training-only undersampling approach reproduced under the random split;
- random oversampling to a final legitimate-to-fraud ratio of 10:1;
- random oversampling to a final legitimate-to-fraud ratio of 5:1;
- SMOTENC to 10:1, if the smoke test confirms valid categorical handling; and
- SMOTENC to 5:1, if the smoke test confirms valid categorical handling.

Resampling will occur inside the training workflow after the split. Validation and test distributions will remain unchanged. Oversampling will not be combined with class weighting in the first comparison.

Ordinary SMOTE will not be applied after one-hot encoding because interpolated one-hot columns can describe invalid category combinations. SMOTENC must use a preprocessing order that preserves categorical meaning. Generated rows will be checked for valid category levels, numeric ranges, missing-value handling, and reproducibility.

A fully balanced 1:1 dataset is excluded from the initial comparison. Moderate ratios are less expensive and provide a direct comparison with the earlier imbalance experiments. A more aggressive ratio will be considered only if the predeclared comparison produces a clear reason.

### Phase 6: additional model families

The selected feature and imbalance treatment will be used for a bounded model-family comparison.

| Model | Role |
| --- | --- |
| Random Forest | Fixed nonlinear reference |
| Histogram Gradient Boosting | Efficient existing boosting reference |
| LightGBM | Existing external boosting reference |
| XGBoost | Additional scalable gradient-boosting family |
| CatBoost | Additional candidate with native treatment of categorical information |
| Linear SVM | Scalable margin-based comparison using a linear implementation |
| KNN | Sampled feasibility study only |

Each new full-data model will begin with one documented configuration. Extensive tuning of every family is prohibited. The model comparison must record predictive results, fit time, inference latency, peak memory where practical, preprocessing requirements, and compatibility with the serving design.

A nonlinear kernel SVM is excluded from the initial full-data comparison because its computational scaling is unsuitable for five million rows. KNN will use documented training and evaluation limits and cannot win the full-data selection from sampled evidence.

### Phase 7: limited tuning

Only the strongest full-validation approach may receive a small, fixed tuning search. Tuning will begin only if the candidate improves the revised Random Forest reference in Average Precision and shows a useful direction in either fraud-count or fraud-value capture.

The search space, number of candidates, primary metric, tie-break rule, and stopping condition must be written into the implementation guide before the first tuning result is observed. No configuration may be added because an early score is disappointing.

## Evaluation measures

### Predictive measures

The primary selection measure remains validation Average Precision. The following will provide context:

- ROC-AUC;
- precision, recall, and F1 at documented thresholds;
- precision-recall curve;
- confusion matrix;
- lift over validation fraud prevalence; and
- probability or decision-score distributions by class.

Accuracy will not be used for selection because legitimate transactions dominate the data.

### Workload scenarios

Fixed review volumes will be retained as sensitivity scenarios rather than described as real institutional capacities. Results will show both the share and number of transactions reviewed. No 1%, 5%, or 10% scenario will be called an approved policy without an external workload assumption.

For equal review volumes, every model will report:

- alerts created;
- fraud cases found;
- legitimate alerts;
- precision;
- fraud-count recall; and
- alerts per evaluation period.

### Transaction-value evaluation

The fixed false-negative ratios of 10, 50, and 100 from Stage 06 will not be used as the primary cost analysis. Every candidate will instead report:

- total fraudulent transaction value;
- detected fraudulent transaction value;
- missed fraudulent transaction value;
- fraud-count recall; and
- fraud-value recall.

Fraud-value recall is defined as:

```text
detected fraudulent transaction value / total fraudulent transaction value
```

If a combined sensitivity calculation is included, it will use:

```text
estimated cost = investigation cost per alert × number of alerts
               + assumed loss rate × missed fraudulent transaction value
```

Transaction value is only a proxy for institutional loss. Review cost, recovery rate, reimbursement, legal cost, and customer impact are not available in the public dataset. Any combined cost result must therefore be labelled as a sensitivity scenario rather than an estimate of actual bank loss.

## Smoke runs and full evidence

Every new resampling method, behavioural feature pipeline, and model family will first receive a reproducible smoke run. Smoke runs verify schema, resampling boundaries, fit and prediction behavior, artifacts, and MLflow logging. They do not select features, models, ratios, or thresholds.

Only results from the complete Stage 07 training and validation partitions count as model-selection evidence. A smoke leader must not change the predeclared full-data comparison.

## Selection and stopping rules

The revised Random Forest result under the new split will be the main reference. A new approach must be judged on more than a numerically higher score.

A candidate can proceed only when it:

1. improves validation Average Precision over the revised reference;
2. shows an operationally relevant improvement in fraud-count or fraud-value capture at an equal review volume;
3. is technically feasible for the local prototype;
4. has reproducible preprocessing, feature, resampling, and model definitions;
5. stores complete lineage, parameters, metrics, runtime, and artifacts in MLflow; and
6. passes the automated safeguards.

Small or conflicting differences will be reported rather than hidden. If no approach provides a convincing improvement, the stage will stop with a documented negative result instead of continuing open-ended feature or parameter searches.

## MLflow evidence

Each full run will record:

- split protocol and split hash;
- dataset and source hashes;
- feature version and definitions;
- point-in-time history rules where applicable;
- preprocessing and resampling parameters;
- effective class counts after resampling;
- model parameters and library versions;
- validation predictive metrics;
- review-volume results;
- fraud-value results;
- training and inference timings;
- diagnostic plots;
- environment details; and
- source snapshots.

Smoke runs, diagnostic studies, sampled feasibility runs, full comparisons, and any tuning runs must be distinguishable through MLflow tags and names.

## Automated safeguards

Tests must verify that:

- the random split is deterministic and stratified;
- the frozen split assigns each transaction to exactly one partition;
- the test partition cannot enter development functions;
- resampling changes training data only;
- oversampling does not change validation prevalence;
- SMOTENC preserves categorical validity;
- historical features use only earlier events;
- the current row cannot contribute to its own history;
- target labels are not used in behavioural features;
- sampled KNN results cannot become a full-data winner;
- value-based metrics use fraudulent transaction amounts correctly; and
- repeated runs with the same seed reproduce the same configuration.

## Parallel MLOps design

The following work can proceed without a final model:

- define MLflow registered-model names and candidate/champion promotion rules;
- define FastAPI request, prediction, health, and model-information contracts;
- define versioned prediction-log fields;
- define delayed-label joins;
- define monitoring report schemas;
- define data-quality, drift, performance, and fraud-value indicators;
- define rule-based investigation and retraining recommendations;
- design the Streamlit pages; and
- design Docker services, ports, volumes, and health checks.

These components may use a fixture or explicitly labelled research model during integration testing. A champion alias, final threshold, and final monitoring reference profile must wait for the revised model decision.

## Planned implementation files

```text
07_predictive_quality_improvement_with_mlflow.py
predictive_quality_utils.py
tests/test_predictive_quality_improvement.py
Markdown files/07_predictive_quality_improvement_implementation_guide.md
Markdown files/07_predictive_quality_improvement_results.md
```

The first implementation task is the deterministic split and behavioural-feature feasibility analysis. Oversampling, new model libraries, and online Feast serving will not be implemented until their preceding checks justify them.

## Immediate next action

Create the Stage 07 implementation skeleton for:

1. the deterministic stratified split and split manifest;
2. training-only entity repetition and feature-signal diagnostics;
3. a report stating whether behavioural feature engineering should proceed; and
4. tests that prove the test partition remains inaccessible to the diagnostic workflow.

No model selection, final test evaluation, registry promotion, or deployment claim is part of this immediate step.
