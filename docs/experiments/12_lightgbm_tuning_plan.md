# Stage 12: Limited LightGBM Tuning Plan

## Why this stage is needed

The model-family comparison selected LightGBM for the next experiment. That comparison used one reasonable configuration per model family; it did not establish that the chosen LightGBM settings were the best settings for this dataset. This stage therefore makes a small, controlled adjustment to LightGBM before an operating rule is frozen.

The search is deliberately limited. It tests a few understandable changes instead of trying a large number of combinations until one happens to fit the validation data.

## What stays fixed

- The stratified random split and its recorded hash stay unchanged.
- The held-out test partition remains closed.
- The retained behavioural feature set stays unchanged.
- Class weighting remains the imbalance treatment.
- Average Precision remains the main ranking metric.
- Fraud-count recall and fraud-value recall at a 5% alert rate remain operational checks.
- Random state 42 remains fixed.

## Configurations

The Stage 11 LightGBM settings form the reference. Seven limited alternatives examine:

- more boosting trees;
- more trees with a smaller learning rate;
- fewer or more leaves per tree;
- smaller or larger minimum leaf populations; and
- stronger L1/L2 regularisation.

This is not a Cartesian grid. Each configuration answers one focused question, apart from the conventional paired change of more trees with a smaller learning rate.

## Decision rule

A full-data candidate may replace the Stage 11 LightGBM reference only when all of the following are true on validation data:

1. Average Precision is higher.
2. Fraud-count recall at the 5% alert rate is not lower.
3. Fraud-value recall at the 5% alert rate is not lower.
4. At least one of the two recall measures is higher.

If qualifying candidates have nearly equal Average Precision, fraud-value recall is preferred, followed by lower validation inference time. Smoke results cannot select a configuration.

## Expected result

The output will be either a defensible tuned LightGBM configuration or evidence that the Stage 11 reference should remain unchanged. It will not be a final model approval because the test set is still closed.
