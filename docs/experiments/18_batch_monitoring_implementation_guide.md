# Batch monitoring implementation guide

## Files

| File | Role |
| --- | --- |
| `fraud_detection/monitoring/batch.py` | Profile construction, validation, PSI, delayed-label metrics, fraud-value measures and recommendation rules |
| `stages/18_batch_monitoring_with_mlflow.py` | Cell-runnable command-line workflow and MLflow logging |
| `fraud_detection/common/training_data.py` | Shared frozen-split reconstruction that retains training features only |
| `fraud_detection/monitoring/reference.py` | Strictly earlier history preparation and aggregate receipt |
| `stages/18_prepare_monitoring_reference.py` | Cell-runnable preparation workflow; currently stops before model scoring |
| `tests/test_batch_monitoring.py` | Synthetic arithmetic, drift, label, decision and read-only-ledger checks |

## Smoke check

From the thesis root:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\18_batch_monitoring_with_mlflow.py" --mode smoke
```

The smoke workflow creates an invented reference and an intentionally shifted invented batch. It should issue an investigation recommendation while proving that automatic retraining remains off. Its metrics are not thesis model-performance evidence.

The separate reference-preparation smoke check reconstructs a training-only sample and adds the three history features:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\18_prepare_monitoring_reference.py" --mode smoke
```

This command deliberately stops before loading or scoring the model. It exports only aggregate lineage and a history checksum to MLflow. Its full mode is available for the later canonical run, but should not be run merely to repeat Stage 17 evidence; model scoring and profile freezing are connected first.

## Creating an explicit reference

The input CSV must already contain the 13 frozen model features and a `score` column produced by the frozen model. It may also contain `transaction_id`, `is_fraud` and `amount`. Use a new empty output directory:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\18_batch_monitoring_with_mlflow.py" `
  --mode create-reference `
  --input-csv ".\path\to\scored_reference.csv" `
  --input-scope training_reference `
  --output ".\monitoring_runs\reference_v1"
```

`operational_reference` is also accepted for a suitable pre-deployment reference. Validation and test scopes are deliberately not available. The runner refuses to replace files in a non-empty output directory.

The project does not currently export a canonical scored reference CSV. That export needs its own provenance-controlled step; do not use the completed final test predictions or manually assembled values to fill the gap.

## Monitoring a scored CSV

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\18_batch_monitoring_with_mlflow.py" `
  --mode monitor-csv `
  --reference-profile ".\monitoring_runs\reference_v1\monitoring_reference_profile.json" `
  --input-csv ".\path\to\new_scored_batch.csv" `
  --input-scope operational_batch `
  --output ".\monitoring_runs\batch_001"
```

## Monitoring the private API ledger

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\18_batch_monitoring_with_mlflow.py" `
  --mode monitor-ledger `
  --reference-profile ".\monitoring_runs\reference_v1\monitoring_reference_profile.json" `
  --ledger-path ".\Code snippets\serving_artifacts\research_v1\predictions.sqlite" `
  --output ".\monitoring_runs\api_batch_001"
```

The ledger is opened read-only. It normally has no fraud labels, so the report will say `unavailable_no_labels` for performance. Full request rows can contain sensitive identifiers and amounts; generated monitoring directories remain local and should not be committed without checking their contents.

## Outputs

Each comparison logs `monitoring_report.json`, `feature_drift.csv`, the exact reference-profile snapshot and library versions to `financial-fraud-stage-18-batch-monitoring` in MLflow. The report separates quality, drift, label availability, performance, cost assumptions and the recommendation.

This endpoint does not let anyone run training scripts. It reads a scored batch or the private prediction ledger and produces aggregate monitoring evidence.
