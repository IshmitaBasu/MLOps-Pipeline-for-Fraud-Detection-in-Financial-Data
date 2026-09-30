# Model-Family Comparison Plan

## Purpose

Stage 10 determines how the retained feature set should handle class imbalance.
Stage 11 will then keep that selected treatment fixed and change only the model
family.

The comparison asks:

> Can another scalable model family improve fraud ranking and fraud-value
> capture over the revised Random Forest reference?

The held-out test partition remains closed. This stage uses training data for
fitting and validation data for comparison.

## Why a new comparison is needed

Random Forest, Histogram Gradient Boosting and LightGBM appeared in the earlier
Stage 06 screening, but those results were produced under an older experimental
setup. At that point, the models did not use the retained sender-location
history features or the imbalance treatment being selected in Stage 10. The
evaluation also did not make fraudulent transaction value part of the model
progression decision.

Stage 11 therefore repeats those three model families under the revised and
consistent conditions:

- the frozen stratified random split;
- the original predictors plus the three retained sender-location history
  features;
- one imbalance treatment selected from the full Stage 10 validation evidence;
- identical training and validation populations for all full-data candidates;
- fraud-count and fraud-value capture at equal alert workloads; and
- no access to the held-out test partition.

Their earlier scores cannot be compared fairly with new XGBoost, CatBoost or
SVM scores produced under these revised conditions. Re-running them provides a
valid reference and shows whether any improvement comes from the new model
family rather than from different data, features or imbalance handling.

A separate Stage 11 implementation preserves Stage 06 as historical evidence.
Stage 06 asked which initial model and imbalance combination looked promising.
Stage 11 asks which model family performs best after the feature set and
imbalance treatment have been revised and fixed. Combining both questions in
the old script would mix two experimental protocols and make the results harder
to reproduce and explain.

## Entry conditions

The full Stage 11 comparison cannot begin until the Stage 10 full-data summary
has recorded:

- the selected imbalance strategy;
- its complete validation evidence;
- the frozen split hash;
- any feasibility exclusions; and
- confirmation that the test split was not evaluated.

Stage 11 must reproduce split hash:

```text
e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242
```

It will use the ten original predictors and three retained sender-location
features represented by `fraud_behaviour_features_v2`.

## Full-data model candidates

Each family receives one documented starting configuration. Extensive tuning
is not part of this comparison.

| Candidate | Role and initial configuration |
| --- | --- |
| Random Forest | Reference: 100 trees, depth 12, minimum leaf size 100, square-root feature sampling |
| Histogram Gradient Boosting | 150 iterations, learning rate 0.1, 31 leaves, minimum leaf size 100 |
| LightGBM | 300 estimators, learning rate 0.05, 31 leaves, row and feature subsampling of 0.8 |
| XGBoost | 300 estimators, learning rate 0.05, depth 6, row and feature subsampling of 0.8 |
| CatBoost | 300 iterations, learning rate 0.05, depth 6, silent deterministic execution |
| Linear SVM | Scalable SGD hinge-loss classifier with training-only scaling and deterministic stopping rules |

The Random Forest is the main reference because it was used for the Stage 08
feature and Stage 10 imbalance experiments. Histogram Gradient Boosting and
LightGBM remain useful efficiency references. XGBoost and CatBoost add distinct
boosting implementations. The linear SVM tests whether a margin-based decision
boundary is useful without attempting an infeasible full nonlinear kernel SVM.

The model set is intentionally limited. It covers bagged trees, three boosting
implementations and a scalable linear margin model without turning the stage
into an unrestricted model search. KNN is included only because it offers a
different distance-based approach; its scalability is tested separately rather
than assumed.

Model-specific preprocessing is allowed only where technically required:

- tree models receive the documented numeric imputation and categorical
  representation;
- CatBoost receives the same compact train-fitted ordinal representation used
  for dense boosting; and
- the linear SVM receives one-hot encoding and numeric scaling fitted on
  training data only.

Every difference must be logged. No model may receive additional predictors or
a different validation population.

## KNN feasibility study

KNN is not a full-data selection candidate because prediction requires distance
comparisons with stored training examples. With millions of transactions this
would create excessive memory use and inference latency.

One separately labelled feasibility run will use:

- a deterministic stratified sample of 100,000 training rows;
- a deterministic stratified sample of 25,000 validation rows;
- 25 neighbours;
- distance weighting;
- Euclidean distance on the common transformed feature representation; and
- fit, inference, memory and predictive measurements.

The sampled KNN score cannot win the full-data comparison, trigger final test
evaluation, or be compared as if it used the complete validation partition.

## Imbalance-treatment handoff

The Stage 10 full-data decision is immutable inside Stage 11. The chosen
treatment will be translated consistently across model APIs:

- if a resampling strategy wins, the same resampled training population is
  supplied to all full-data candidates;
- if class weighting wins, each model uses its documented equivalent positive
  weighting while training rows remain unchanged; and
- KNN uses the selected resampling rule within its sampled training population,
  but no class-weight substitute because KNN does not support one.

The implementation must stop if the Stage 10 decision is missing, comes from a
smoke run, or does not match the frozen strategy names.

## Evaluation

Validation Average Precision remains the primary metric. Each full-data model
also records:

- ROC-AUC;
- precision, recall and F1 at the validation-derived maximum-F1 threshold;
- lift over validation fraud prevalence;
- training and inference time;
- transformed feature dimensions and effective training rows;
- model or pipeline size where practical; and
- probability or decision-score distributions.

At fixed validation alert shares of 1%, 5% and 10%, the report records:

- fraud cases captured;
- legitimate alerts;
- precision;
- fraud-count recall;
- fraudulent transaction value captured;
- missed fraudulent transaction value; and
- fraud-value recall.

The 5% alert share is the predeclared comparison point, not a claim about an
institution's real investigation capacity.

## Progression rule

A full-data candidate may replace the Random Forest reference only when:

1. validation Average Precision is higher;
2. fraud-count recall at the 5% alert share is not lower;
3. fraud-value recall at the 5% alert share is not lower;
4. at least one of the two 5% recall measures is higher; and
5. training and inference complete within the local prototype environment.

If several candidates pass, the highest Average Precision wins. Differences
below `0.0001` are treated as a tie; higher 5% fraud-value recall is the first
tie-breaker and lower validation inference time is the second.

If no candidate passes, Random Forest remains the reference. Smoke runs and the
sampled KNN run cannot make this decision.

## Smoke and dependency gates

Before any full comparison:

- required XGBoost and CatBoost versions must be pinned;
- each optional library must import successfully;
- every candidate must fit and score the 50,000-row smoke population;
- output scores must be finite and have the expected row count;
- model-specific categorical handling must accept unseen validation levels;
- the selected Stage 10 imbalance treatment must affect training only; and
- MLflow logging must distinguish smoke, full-data and sampled-feasibility
  evidence.

An unavailable or computationally infeasible candidate is recorded as a
feasibility exclusion. It is not silently replaced after other scores are
known.

## Test isolation

Stage 11 development APIs accept training and validation inputs only. Test
features, labels, amounts, predictions and metrics are prohibited. The test
partition may appear only as an untouched row count in the split manifest.

One final test evaluation becomes possible only after the feature set,
imbalance strategy, model family, limited tuning decision and operating rule
have been frozen from validation evidence and documented.

## MLflow evidence

Each run records:

- evidence scope and model role;
- Stage 10 decision and summary run ID;
- source, feature and split hashes;
- preprocessing, imbalance and model parameters;
- library versions;
- effective training rows and class counts;
- predictive and equal-workload validation metrics;
- fraud-value results;
- fit and inference timings;
- feasibility exclusions; and
- source and environment snapshots.

No Stage 11 screening model will be registered or promoted.

## Implementation files

```text
11_model_family_comparison_with_mlflow.py
model_family_comparison_utils.py
tests/test_model_family_comparison.py
Markdown files/11_model_family_comparison_implementation_guide.md
Markdown files/11_model_family_comparison_results.md
```

These files are implemented. The full Stage 10 summary retained
`class_weight_reference`, and the Stage 11 runner verifies that handoff directly
from MLflow before it loads modeling data. The next action is the Stage 11 smoke
workflow; no Stage 11 model run has yet been performed.
