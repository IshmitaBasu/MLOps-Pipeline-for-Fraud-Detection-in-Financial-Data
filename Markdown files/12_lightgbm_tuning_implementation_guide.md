# Stage 12: LightGBM Tuning Implementation Guide

## What the script does

`12_lightgbm_tuning_with_mlflow.py` loads the completed Stage 11 decision from MLflow and verifies that LightGBM was selected using the frozen full-data split. It then recreates the same training and validation partitions, fits eight predeclared LightGBM configurations, and compares them using validation evidence only.

In simple terms, the model type and data are no longer changing. The script is checking whether a few careful adjustments to LightGBM make it rank fraud cases better without reducing the number or value of frauds found within a fixed 5% review workload.

Each candidate run records its parameters, validation metrics, workload table, timings, data lineage, split manifest, library versions, and source snapshot in MLflow. A separate summary run records the final validation-only decision.

From the reporting update on 5 October 2026, validation metrics include the numerical maximum-F1 threshold, precision, recall, F1, accuracy, balanced accuracy, specificity, confusion counts (TN, FP, FN, TP), and alert rate, as well as Average Precision and ROC-AUC. Classification fields labelled `at_max_f1` refer to that validation-selected threshold. `validation_f1` also refers to the maximum-F1 rule; the existing field name is preserved. Fixed-workload fraud-count and fraud-value metrics describe separate batch ranking scenarios.

Completed historical runs did not save all those fields. Reconstructed historical values and the missing numerical thresholds are explicitly identified in the results document. Historical artifacts remain unchanged, and a rerun is not required to read the documented results.

## Safeguards

- A valid full-data Stage 11 summary is mandatory.
- The Stage 11 handoff must show that LightGBM was selected.
- Full mode requires exactly 5,000,000 source rows and the frozen split hash.
- The tuning function has no test-frame parameter.
- The held-out test rows are counted but not materialised or scored.
- Smoke mode is only a technical check and cannot select a configuration.

## Run the smoke check

From the thesis workspace:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\12_lightgbm_tuning_with_mlflow.py" --mode smoke
```

Expected outcome: all eight configurations run, MLflow stores their evidence, and the final message says that no configuration was selected because this was a smoke run.

## Run the full comparison

Run this only after the smoke output has been checked:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\12_lightgbm_tuning_with_mlflow.py" --mode full
```

Expected outcome: the script prints validation Average Precision for each configuration, creates a full-data summary run, and either retains the Stage 11 reference or selects one tuning candidate. The test split remains unevaluated.

## Where to look in MLflow

Open the experiment `financial-fraud-stage-12-lightgbm-tuning`. Candidate runs contain `validation_workloads.csv` and `tuning_result.json`. The summary run contains `lightgbm_tuning_smoke_test_summary.json` or `lightgbm_tuning_full_data_summary.json` plus a combined CSV table.
