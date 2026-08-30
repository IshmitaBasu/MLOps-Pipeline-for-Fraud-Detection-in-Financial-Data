# Controlled Feature-Engineering Results

> **Use this document for:** the completed full-data results, their interpretation, and the current feature-set decision. For the pre-run design, see `05_feature_engineering_experiment_plan.md`. For execution instructions, see `05_feature_engineering_implementation_guide.md`.

## Result status

The controlled feature experiment was completed on 29 July 2026 using all 5,000,000 transactions. The same Random Forest configuration was used for every feature group so that differences could be attributed to the input features rather than to a model change.

The experiment is recorded in MLflow under:

```text
Experiment: financial-fraud-feast-feature-engineering
Comparison summary run: 99183d4edc81406281ff5204e216f013
```

These are full-data results, not smoke-test results.

## What the version labels mean

`original_v1` is the frozen set of ten original predictors. Names ending in `_v2` identify candidate engineered-feature variants tested against that reference.

The `_v2` suffix does **not** mean that the Feast repository or stored dataset was upgraded to version 2. All engineered features were calculated temporarily in memory. The persisted Feast handoff remained unchanged at `v1`.

## Controlled experiment conditions

The following conditions were held constant:

- Feast `v1` source rows and labels;
- five-million-row dataset;
- chronological 70% training, 15% validation, and 15% test split;
- Random Forest with 100 trees, maximum depth 12, minimum leaf size 100, and square-root feature sampling;
- balanced-subsample class weighting;
- training-only preprocessing;
- random state 42; and
- validation Average Precision as the feature-group selection metric.

The test split was withheld during the five-group comparison. It was used only after `temporal_v2` had been selected from validation evidence.

## Validation results

The implementation logs scikit-learn Average Precision under the metric name `pr_auc`. This document calls it **Average Precision** for precision.

| Feature group | Features | Validation Average Precision | Absolute change | Relative change | MLflow run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| `original_v1` | 10 | 0.043960 | Reference | Reference | `cb0ca81735d248069b43d544e0d79a9c` |
| `amount_v2` | 11 | 0.043872 | -0.000088 | -0.20% | `7280a15b4d4d428cb90e3e0a03fafbff` |
| `temporal_v2` | 14 | **0.044342** | **+0.000382** | **+0.87%** | `64febfc02ac44f5b9bd413a4c1522582` |
| `missingness_sensitivity_v2` | 11 | 0.043982 | +0.000022 | +0.05% | `fa145b733e3e45c382d5e0f2290550f3` |
| `combined_v2` | 16 | 0.044130 | +0.000171 | +0.39% | `58f8b273e9ce4475b750124634fe8ed3` |

The predefined selection rule chose `temporal_v2` because it had the highest validation Average Precision. It added:

- `transaction_hour`;
- `transaction_day_of_week`;
- `transaction_month`; and
- `transaction_is_weekend`.

The validation increase was small: approximately 0.87% relative to the original-feature control.

## Test-period check

After validation selection, `temporal_v2` was trained and evaluated once on the test period. The selected test run was:

```text
Run ID: 2635155e7db2461c8591f7eb9d9e8c58
```

| Metric | Original-feature benchmark | Selected `temporal_v2` | Direction |
| --- | ---: | ---: | --- |
| Test Average Precision | **0.044113** | 0.043947 | Worse by 0.000167 (-0.38%) |
| Test ROC-AUC | **0.594210** | 0.593542 | Slightly worse |
| Precision | **0.043974** | 0.043974 | Effectively unchanged |
| Recall | 1.000000 | 1.000000 | Unchanged |
| F1 | **0.084244** | 0.084243 | Effectively unchanged |
| False positives | **588,754** | 588,764 | 10 more false alerts |

The temporal feature group did not confirm its validation improvement on the later test period. Its ranking metrics were slightly worse, and it did not reduce the false-positive burden.

## Interpretation of each feature group

- `amount_v2` did not improve validation performance. The log transformation should not be added on the basis of this experiment.
- `temporal_v2` produced the largest validation improvement, but the improvement did not persist on the test period.
- `missingness_sensitivity_v2` changed performance by only about 0.05%. Its missingness pattern may also reflect synthetic-data generation, so it should remain a sensitivity result rather than a trusted production feature.
- `combined_v2` was better than the original control on validation but weaker than `temporal_v2`. Adding all candidate features did not create a stronger result.

## Current feature-set decision

This experiment does not provide convincing evidence that any engineered group is more reliable than the original ten predictors. `temporal_v2` was the validation winner, but it is not accepted as a canonical feature-set replacement because the test result did not confirm the improvement.

The current defensible decision is therefore:

1. retain `original_v1` as the stable reference feature set;
2. report `temporal_v2` as a promising but unconfirmed sensitivity result;
3. do not create a persisted Feast `v2` handoff from these features; and
4. move to controlled model, imbalance-handling, and cost-aware threshold experiments.

Neither the original Random Forest nor the temporal variant is ready for deployment. Both flag approximately 82% of test transactions at the validation-F1-selected threshold, which would create an impractical investigation workload.

## Reproducibility evidence

MLflow retains the group parameters, feature definitions, validation metrics, evaluation plots, environment details, source snapshots, and run IDs. The selected temporal run also stores the fitted preprocessing-and-model pipeline and its final test evidence.

The maintained repository contains:

- `05_feature_engineering_with_mlflow.py` — executable experiment;
- `05_feature_engineering_experiment_plan.md` — pre-run design and controls;
- `05_feature_engineering_implementation_guide.md` — execution and implementation explanation;
- `05_feature_engineering_results.md` — this completed result and decision record; and
- `tests/test_feature_engineering.py` — deterministic feature-calculation safeguards.
