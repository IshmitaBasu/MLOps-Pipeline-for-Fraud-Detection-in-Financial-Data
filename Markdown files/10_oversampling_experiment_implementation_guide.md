# Oversampling Experiment: Implementation Guide

## What this stage tests

Stage 10 keeps the retained sender-location features and Random Forest fixed.
It changes only how the training data deals with the shortage of fraud cases.

For example, a training set might contain 100 legitimate transactions and five
fraudulent transactions. The strategies treat that imbalance differently:

- class weighting keeps all 105 rows but makes fraud errors more important;
- 5:1 undersampling keeps the five fraud rows and only 25 legitimate rows;
- 10:1 random oversampling duplicates fraud rows until there are ten fraud
  rows; and
- 5:1 random oversampling duplicates fraud rows until there are 20 fraud rows.

SMOTENC also increases the fraud group, but it creates new numeric combinations
while choosing existing categorical values. It is checked separately because a
synthetic transaction must not contain an impossible category created by
interpolating one-hot columns.

## Files

| File | Purpose |
| --- | --- |
| `10_oversampling_experiment_with_mlflow.py` | Runs and logs the controlled smoke and full-data workflows |
| `oversampling_experiment_utils.py` | Defines strategies, preprocessing, resampling, model construction and selection safeguards |
| `tests/test_oversampling_experiment.py` | Checks ratios, duplication, SMOTENC validity, fixed model settings and test isolation |
| `10_oversampling_experiment_plan.md` | Stores the question and decision rule fixed before results |
| `10_oversampling_experiment_results.md` | Records smoke and later full-data outputs |

Both Python files use `# %%` cell markers.

## Processing order

The implementation follows this order:

1. reproduce the frozen stratified random split;
2. remove test rows before joining sender keys or calculating history;
3. create the three point-in-time sender-location features;
4. fit imputation and category coding on training rows only;
5. freeze the category vocabulary from the original training rows;
6. transform validation rows without changing them;
7. resample only the encoded training matrix;
8. one-hot encode with the frozen vocabulary and fit the fixed Random Forest;
9. score the original validation distribution; and
10. log predictive, fixed-workload and fraud-value results in MLflow.

Validation rows are fingerprinted before the strategy loop. The run stops if a
strategy changes the validation values, labels, amounts, order, columns or
data types.

## Fixed strategies

The smoke run executes:

```text
class_weight_reference
undersample_5_to_1
random_oversample_10_to_1
random_oversample_5_to_1
smotenc_10_to_1
smotenc_5_to_1
```

The class-weighted model uses all training rows and
`class_weight="balanced_subsample"`. Resampled models use no class weighting.
All other Random Forest settings remain identical.

Random oversampling is verified against the sampler's source-row indices, so
it cannot silently generate modified transactions. SMOTENC is repeated with
seed 42 and must produce identical output with valid categorical codes and
finite numeric values.

## Automated tests

From the thesis directory, run:

```powershell
.\masters_thesis\Scripts\python.exe -m unittest ".\Code snippets\tests\test_oversampling_experiment.py" -v
```

The tests do not train on the complete dataset.

## Smoke command

After the tests pass, run:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\10_oversampling_experiment_with_mlflow.py" --mode smoke
```

The smoke workflow is fixed to a stratified sample of 50,000 transactions. It
uses 35,000 training rows, 7,500 validation rows and holds aside 7,500 test
rows. It may take longer than earlier smoke runs because six forests are fitted
and both SMOTENC configurations are repeated to verify reproducibility.

## What the smoke result means

A successful smoke result proves that:

- every strategy can process the feature set;
- requested class ratios are produced;
- validation data remains unchanged;
- SMOTENC generates technically valid values;
- the fixed model can fit and score; and
- MLflow receives the required evidence.

The highest smoke score does not select a strategy. The complete comparison
is enabled only after the smoke output has been reviewed and documented. The
held-out test split remains closed.

## Full-data command

After the smoke gate has passed, run:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\10_oversampling_experiment_with_mlflow.py" --mode full
```

Full mode refuses `--sample-rows`. Before any model is fitted, it requires:

- exactly 5,000,000 source rows;
- the frozen 3,500,000/750,000/750,000 split; and
- split hash
  `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242`.

The strategies run sequentially and large intermediate frames are released
before fitting to reduce memory pressure. SMOTENC is not repeated twice on full
data because reproducibility already passed the smoke gate. If a full SMOTENC
configuration raises a measured `MemoryError`, the script logs a feasibility
exclusion in MLflow and continues with the remaining predeclared strategies.
It does not replace SMOTENC with ordinary SMOTE.

The full summary applies the frozen Average Precision and 5% workload rule.
Only the full validation results may select the imbalance strategy for the
later model-family comparison. No test predictions or metrics are produced.
