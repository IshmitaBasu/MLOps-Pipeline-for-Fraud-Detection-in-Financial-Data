# Stage 07 Predictive-Quality Diagnostic: Implementation Guide

## What this first script does

The first Stage 07 script does not train another model. It checks whether the
raw identifiers contain enough repeated behaviour to justify spending time on
historical feature engineering.

It creates the new stratified random split described in the Stage 07 plan and
then gives only the training rows to the diagnostic functions. Validation and
test rows are counted in the split manifest but are not explored or evaluated.

All Python files added for this stage contain `# %%` markers and can therefore
be run as cells in VS Code as well as from the terminal.

## Files involved

| File | Purpose |
| --- | --- |
| `07_predictive_quality_improvement_with_mlflow.py` | Loads the required raw columns, freezes the split, runs the training-only checks, and logs the evidence in MLflow |
| `predictive_quality_utils.py` | Contains the split safeguards and past-only diagnostic calculations |
| `tests/test_predictive_quality_improvement.py` | Verifies the random split, test isolation, and exclusion of current and future transactions |
| `07_predictive_quality_improvement_results.md` | Records each run, its values, and the resulting decision |

## Data used by the diagnostic

Only the following raw columns are loaded:

```text
transaction_id
timestamp
sender_account
receiver_account
amount
location
ip_address
device_hash
is_fraud
```

The raw identifiers are used only to study repetition and construct aggregate
history. They are not supplied directly to a model.

## Split behaviour

The default split is deterministic and stratified by `is_fraud`:

- 70% training;
- 15% validation; and
- 15% test.

The random seed is 42. MLflow stores a split manifest containing row counts,
fraud counts, fraud rates, timestamp ranges, and a SHA-256 hash of the split
assignment. The diagnostic code receives only the materialised training table.

## Checks performed

For sender, receiver, device, IP address, and selected combinations, the script
records:

- how many distinct entities exist;
- how often they repeat;
- how many transactions have an earlier observation for the same entity; and
- whether fraud rates differ between new and returning entities.

It also compares the current amount with the sender's mean amount from strictly
earlier transactions. The current row is excluded. Future rows and previous
fraud labels are never used to construct the diagnostic feature.

## Simple examples

The following examples are hypothetical. They explain what the checks mean;
they are not findings from the dataset.

### An unusual transaction amount

A EUR 2,000 payment is not automatically suspicious. It may be normal for a
customer who regularly transfers similar amounts, but unusual for a customer
whose earlier transactions were normally between EUR 20 and EUR 50. Comparing
the current amount with that sender's past amounts gives the model this context.

### A new device or IP address

A customer may normally make transactions from the same phone and IP address.
If a new transaction comes from a device or IP address that has never previously
been associated with that sender, this change could be useful information. The
diagnostic first checks whether devices and IP addresses repeat often enough for
such a feature to be meaningful.

### A new receiver

A sender may repeatedly transfer money to the same small group of receivers. A
payment to a receiver that the sender has never used before may be different
from the sender's normal behaviour. The script therefore checks both receiver
history and sender-receiver history.

### A new location

If a sender's previous transactions consistently came from one location, a
transaction from a previously unseen location might be relevant. This idea is
useful only when the dataset contains enough repeated sender-location history.

### Why repetition matters

Historical features cannot be calculated for an entity that appears only once.
For example, if every sender account occurs in only one transaction, there is no
earlier sender behaviour to compare with the current transaction. That is why
the script measures repetition and history coverage before a complete feature
pipeline is built.

None of these conditions proves that a transaction is fraudulent. They are
possible warning signals whose value must later be tested with a fixed model on
the untouched validation partition.

## Initial feasibility gate

An entity passes the initial screen only when all three conditions hold:

- at least 5% of training transactions have prior history;
- both the new and returning groups contain at least 100 fraud cases; and
- their fraud rates differ by at least 10% in relative terms.

This is a screening rule, not proof that the feature improves a model. Passing
means that a small point-in-time feature group is worth implementing and
validating next. Failing means that the negative result should be documented
before more engineering time is spent.

## MLflow artifacts

Each run logs:

- `split_manifest.json`;
- `feature_feasibility_decision.json`;
- `entity_repetition_summary.csv`;
- `new_vs_returning_fraud_rates.csv`;
- `sender_amount_deviation_summary.csv`; and
- snapshots of the Stage 07 source and documentation files.

No model is trained, and the run is marked as ineligible for promotion.

## Run the automated safeguards

From the thesis folder:

```powershell
.\masters_thesis\Scripts\python.exe -m unittest ".\Code snippets\tests\test_predictive_quality_improvement.py" -v
```

## Run a smoke diagnostic

```powershell
.\masters_thesis\Scripts\python.exe `
  ".\Code snippets\07_predictive_quality_improvement_with_mlflow.py" `
  --mode diagnostics `
  --sample-rows 50000 `
  --skip-data-hash
```

This checks that the pipeline and MLflow artifacts work. Its figures are not
thesis evidence and must not determine which features are kept.

The smoke option reads the first 50,000 rows so that it remains quick. Those
rows may not represent the class distribution or repetition found across the
complete dataset. Low coverage, few fraud cases, or an apparent relationship in
the smoke output must therefore not be treated as a feature decision.

## Run the full diagnostic

```powershell
.\masters_thesis\Scripts\python.exe `
  ".\Code snippets\07_predictive_quality_improvement_with_mlflow.py" `
  --mode diagnostics
```

The full run reads several high-cardinality identifier columns from all five
million transactions, so it will take longer and use substantially more memory
than the smoke run.

## How to use the result

First confirm that the run contains all five artifacts and that it states
`test_split_evaluated=false`. Then copy the run ID and the important values into
the results document.

If at least one entity passes the gate, the next script should implement a
small, versioned, point-in-time feature group using only the promising entities.
That group should first be compared against `original_v1` with a fixed model.
If no entity passes, move to the random-split reference models and oversampling
experiments without building a large historical feature pipeline.
