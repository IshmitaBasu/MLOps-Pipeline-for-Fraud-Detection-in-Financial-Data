# Stage 06 Implementation Guide

The stage-06 runner handles two related workflows in one Python file: completed model and imbalance screening, and the new Random Forest tuning search. The file remains a normal `.py` script and uses `# %%` markers so its sections can also be opened as cells in VS Code or Spyder.

> **Execution status, 17 September 2026:** Screening, the tuning smoke test, and full-data validation tuning are complete. Full tuning summary run `0653535d894b47e4af7c7db43a51c3cd` selected `rf_leaf_50` by validation Average Precision. The result remains a research candidate because the improvement was small and fixed-capacity behaviour did not improve. No final-test evaluation, deployment threshold, model registration, or champion promotion has occurred.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `06_model_optimization_and_operational_evaluation_with_mlflow.py` | Loads data, starts MLflow runs, executes screening or tuning, and writes summaries. |
| `model_optimization_utils.py` | Builds models, performs deterministic training-only undersampling, defines tuning configurations, and calculates threshold and cost scenarios. |
| `fraud_modeling_utils.py` | Provides the shared chronological split, metrics, plots, lineage logging, and fitted-model evaluation contract. |
| `tests/test_model_optimization_and_operational_evaluation.py` | Checks tuning configuration, resampling, selection, and test-isolation rules. |
| `06_model_optimization_and_operational_evaluation_plan.md` | Records the questions, fixed controls, search design, and decision rules. |
| `06_model_optimization_and_operational_evaluation_results.md` | Records commands, run IDs, values, interpretation, and the current decision. |

## Screening mode

`screening` is the default mode. It was used for the model-family and imbalance comparisons. Models and imbalance strategies are supplied explicitly when a comparison other than the two default controls is required.

Example:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --models random_forest_depth_12 `
  --imbalance-strategies class_weight undersample_10_to_1 undersample_5_to_1
```

Running that command again would duplicate the completed imbalance experiment. Screening mode is retained for reproducibility, not as the current next action.

## Tuning mode

Tuning is activated with:

```text
--mode tuning
```

This mode fixes the choices already supported by full-data evidence:

```text
Model family: Random Forest
Feature set: original_v1
Imbalance strategy: 5:1 training-only undersampling
Random state: 42
Selection metric: validation Average Precision
```

The mode rejects `--models` and `--imbalance-strategies`. This prevents a tuning command from quietly changing the selected model family or resampling method. An optional `--tuning-configurations` argument can run a named subset for debugging, but the official smoke and full runs should use all nine predeclared configurations.

## Predeclared tuning configurations

| Name | Trees | Depth | Minimum leaf | Maximum features |
| --- | ---: | ---: | ---: | --- |
| `rf_reference` | 100 | 12 | 100 | `sqrt` |
| `rf_trees_200` | 200 | 12 | 100 | `sqrt` |
| `rf_trees_300` | 300 | 12 | 100 | `sqrt` |
| `rf_depth_8` | 100 | 8 | 100 | `sqrt` |
| `rf_depth_16` | 100 | 16 | 100 | `sqrt` |
| `rf_leaf_50` | 100 | 12 | 50 | `sqrt` |
| `rf_leaf_250` | 100 | 12 | 250 | `sqrt` |
| `rf_features_half` | 100 | 12 | 100 | `0.5` |
| `rf_combined_flexible` | 200 | 16 | 50 | `0.5` |

The reference configuration must remain in the search. It checks that the earlier 5:1 result can be reproduced inside the tuning workflow. Seven configurations change one setting at a time. The combined configuration is the only predeclared interaction candidate.

## Run the tests

From the thesis workspace:

```powershell
.\masters_thesis\Scripts\python.exe -m unittest discover -s ".\Code snippets\tests" -v
```

The tuning smoke run should start only if the tests pass.

## Run the tuning smoke test

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --mode tuning `
  --sample-rows 50000 `
  --skip-data-hash
```

Expected behavior:

- the terminal reports 7,500 held-out test rows and states that they will not be evaluated;
- 5:1 undersampling is applied to the 35,000-row smoke training split;
- nine tuning runs are created;
- one tuning summary run is created;
- each candidate contains validation metrics and an operational threshold table;
- the summary labels the highest smoke score as diagnostic rather than selecting it; and
- the search definition is not changed in response to smoke scores.

## Inspect the smoke artifacts

Start MLflow with:

```powershell
.\masters_thesis\Scripts\mlflow.exe ui --workers 1 --backend-store-uri "sqlite:///D:/Germany/Documents/Magdeburg/Semester Documents/Sem 5/Thesis/Code snippets/mlflow_tracking.db"
```

Open the experiment:

```text
financial-fraud-feast-model-optimization-operational-evaluation
```

Each tuning candidate should contain:

```text
operational_evaluation/validation_threshold_summary.csv
operational_evaluation/validation_threshold_summary.json
evaluation/metrics_summary.json
reproducibility/split_summary.json
reproducibility/feature_set.json
reproducibility/environment.json
```

The summary run is named:

```text
random_forest_5_to_1_tuning_summary
```

Its `decision_status` should be `technical_smoke_no_selection` for the smoke run.

## Reproducing the completed full-data tuning run

The full run was started only after the smoke output and artifacts had been checked. It can be reproduced with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" --mode tuning
```

Without `--sample-rows`, the original five-million-row dataset is used. The 5:1 strategy reduces the effective training data to 752,718 rows, but all nine forests still need to be fitted and evaluated. The completed run therefore took considerably longer than the smoke test. Re-running it will create duplicate MLflow runs and is not the current next action.

## What tuning does not do

Tuning mode does not:

- evaluate the held-out test period;
- use smoke scores as thesis evidence;
- add class weighting after undersampling;
- change the feature set;
- run XGBoost or LightGBM;
- register a champion model; or
- choose a deployment threshold automatically.

The full tuning output selected `rf_leaf_50`, but no operating threshold has been frozen. The current next action is supervisor review of the provisional acceptance target, alert-capacity choice, and cost assumptions. Only after the candidate and a validation-derived threshold are explicitly frozen should a separate final-test path be considered.
