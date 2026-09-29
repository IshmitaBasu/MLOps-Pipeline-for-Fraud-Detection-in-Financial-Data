# Sender-Location History Feature Experiment: Implementation Guide

## What the experiment does

The script trains the same Random Forest twice. The first run uses the ten
original transaction features. The second adds three descriptions of the
sender's earlier behaviour. Any difference therefore comes from the added
history rather than a different model configuration.

All Stage 08 Python files use `# %%` cell markers and can be run cell by cell in
VS Code or as complete scripts.

## Files

| File | Purpose |
| --- | --- |
| `08_sender_location_history_feature_experiment_with_mlflow.py` | Loads the canonical table, recreates the split, runs the two models, and logs the validation comparison |
| `sender_location_history_utils.py` | Calculates past-only features, equal-workload measures, and the progression decision |
| `tests/test_sender_location_history_feature_experiment.py` | Tests history boundaries, test isolation, model equality, and value-based workload calculations |
| `08_sender_location_history_feature_experiment_results.md` | Records smoke and full results without changing the pre-run rules |

## Simple example

Suppose sender A made a transaction from Berlin on Monday.

- A later transaction from Berlin has sender history and sender-Berlin history.
- A later transaction from Hamburg has sender history but no sender-Hamburg
  history, so `is_new_location_for_returning_sender` is 1.
- The first transaction ever observed for sender B is not labelled as a new
  location for a returning sender because B has no earlier history.

If a training and validation transaction share the same timestamp, neither is
treated as earlier than the other. Validation transactions can read earlier
training history but cannot update the history seen by later validation rows.

## MLflow runs and artifacts

The experiment produces three runs:

1. `original_v1` control;
2. `sender_location_history_v1` candidate; and
3. a comparison-summary run.

Each model run stores validation metrics, plots, feature definitions, source
snapshots, and `validation_fixed_workload` evidence. The summary stores both
run IDs, the split manifest, the predeclared decision, and a combined workload
CSV. All runs state that the test split was not evaluated and that they are not
eligible for model promotion.

## Run the safeguards

```powershell
.\masters_thesis\Scripts\python.exe -m unittest ".\Code snippets\tests\test_sender_location_history_feature_experiment.py" -v
```

## Run the smoke comparison

```powershell
.\masters_thesis\Scripts\python.exe `
  ".\Code snippets\08_sender_location_history_feature_experiment_with_mlflow.py" `
  --mode comparison `
  --sample-rows 50000 `
  --skip-data-hash
```

The smoke run still retrieves the canonical Feast table and scans the raw CSV
for matching sender keys. It may therefore take some time even though only
50,000 sampled rows are fitted. Its scores verify the implementation only.

## Run the complete comparison

Run this only after the smoke output and MLflow artifacts have been checked:

```powershell
.\masters_thesis\Scripts\python.exe `
  ".\Code snippets\08_sender_location_history_feature_experiment_with_mlflow.py" `
  --mode comparison
```

The full run constructs histories for 3.5 million training rows and 750,000
validation rows and fits two Random Forest models. It will require substantial
time and memory.

Before recreating the split, the script restores the canonical raw transaction
order. The source was verified to contain the continuous sequence `T100000` to
`T5099999`, so the numeric part of `transaction_id` provides that order without
an additional five-million-row merge. Feast contains the same transaction IDs
but does not guarantee the same complete row order as the raw CSV. Because the
frozen Stage 07 random split was created from raw order, this restoration is
required for the split hash to match. The script stops before feature
construction or model fitting if the hash is different.

## Reading the result

Start with validation Average Precision. Then compare fraud-count and
fraud-value recall at the 5% alert volume. The summary JSON applies the frozen
rule automatically.

If the candidate passes, keep the feature group for the next validation-only
oversampling and model-family experiments. If it fails, retain `original_v1`.
Do not evaluate the test partition after this experiment.
