# Stage 14: Final Test Evaluation Guide

## Files and safeguards

`14_final_test_evaluation_with_mlflow.py` is the cell-based runner. `final_test_evaluation_utils.py` validates the Stage 13 handoff, freezes the protocol, calculates training-only history for test queries, and evaluates the fixed rules. Automated safeguards are in `tests/test_final_test_evaluation.py`.

The runner checks the model definition and key library versions against the executed Stage 13 reference, then logs the frozen protocol before loading and scoring test data. Full mode verifies the five-million-row source, raw and model-data hashes, and the split hash against Stage 13. It refits the same LightGBM on the original training partition. Test queries do not update history; validation rows are omitted.

## Execution

From the thesis workspace, the technical smoke option is:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\14_final_test_evaluation_with_mlflow.py" --mode smoke
```

This evaluates a sampled technical partition at the already frozen rules. It is not the canonical final test and cannot support a new decision.

The authorised final evaluation command is:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\14_final_test_evaluation_with_mlflow.py" --mode full
```

Do not rerun the full command once final evaluation is complete. A local receipt in the ignored `final_test_evaluation/` directory and an MLflow canonical-test tag block repeated full evaluation. The receipt is created exclusively before scoring and updated after results are obtained. An interrupted receipt must be inspected before any recovery; it is not automatically removed.

## Artifacts

Experiment: `financial-fraud-stage-14-final-test-evaluation`.

The final run stores `frozen_evaluation_protocol.json`, `final_test_evaluation.json`, `test_workloads.csv`, `test_confusion_matrix.json`, data/feature lineage, library versions, and source snapshots. The exact evaluated preprocessing/model pipeline is saved under `evaluated_model/evaluated_lightgbm_pipeline.joblib`, for later registration without fitting a different model.

The model remains unregistered. Saved pipeline artifacts should only be loaded from trusted local experiment storage. Test results and interpretation are recorded in `14_final_test_evaluation_results.md`.
