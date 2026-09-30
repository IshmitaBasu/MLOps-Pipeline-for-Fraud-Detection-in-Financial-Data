# Sender-Location Feature Store: Implementation Guide

## Purpose

Stage 08 showed that three sender-location history features improved the fixed
Random Forest on validation data. Stage 09 makes those features available
through a versioned Feast contract so that training and prediction use the same
definitions.

This stage does not train another model and does not evaluate the test split.

## Files

| File | Purpose |
| --- | --- |
| `09_sender_location_feature_store.py` | Builds, registers, materializes, and validates the versioned contract |
| `sender_location_feature_store.py` | Contains point-in-time calculations, Feast objects, persistence, and consistency checks |
| `fraud_feature_definitions.py` | Retains the original `v1` contract and registers the additive `v2` behavioural definitions |
| `feature_store.yaml` | Configures the local SQLite online store |
| `tests/test_sender_location_feature_store.py` | Checks calculations, leakage boundaries, entity keys, and offline/online equality |

All new Python files use `# %%` cell markers.

## Versioned contract

The original `fraud_model_features_v1` service remains unchanged. Stage 09 adds
the separate service:

```text
fraud_behaviour_features_v2
```

It contains:

| Feature | Feast source |
| --- | --- |
| `sender_prior_transaction_count` | Sender feature view keyed by `sender_account` |
| `sender_location_prior_transaction_count` | Sender-location feature view keyed by `sender_location_key` |
| `is_new_location_for_returning_sender` | Shared adapter derived from the two Feast-served counts |

The sender-location key uses a length-prefixed format such as
`9:ACC123456|Berlin`. This prevents two different sender/location combinations
from accidentally producing the same key.

Raw sender and location values are lookup keys. They are not passed to the
model as high-cardinality predictors.

## Why the source contains state snapshots

Feast retrieves the most recent feature value available at or before a request
timestamp. The source tables therefore store cumulative state after each group
of training events.

Each state becomes effective one microsecond after its source transaction
timestamp. Consequently:

- a historical query at the transaction timestamp cannot see that transaction;
- transactions sharing a timestamp cannot see one another;
- the next transaction can see the updated state; and
- the newest snapshot can be materialized for online inference.

Missing counts mean that no history exists and are consistently treated as
zero. `is_new_location_for_returning_sender` is true only when the sender count
is greater than zero and the sender-location count is zero.

The source includes a zero-state baseline for every sender and sender-location
key found in the training and validation development population. Feast's local
historical join can otherwise omit a first-seen entity row when no earlier
source record exists. These baseline rows contain no activity and no label
information; validation events still never update a count.

## Data boundary

The canonical source uses the frozen Stage 07 split. Only the 3.5 million
training events update state. Validation rows may request point-in-time values
but do not update the feature source. Test rows neither update state nor appear
in development queries.

The target is required only to reproduce the stratified split assignment. It is
not included in either Feast source and is not used in any feature calculation.
The build stops if the split hash differs from:

```text
e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242
```

## Local online store

`feature_store.yaml` now uses:

```yaml
online_store:
  type: sqlite
  path: feature_repo/data/online_store.db
```

The generated database is local and ignored by Git. It is a thesis prototype,
not a production-scale online store.

## Verification

The consistency check selects entity rows at an identical source cutoff and
retrieves their features through:

1. Feast historical retrieval; and
2. Feast SQLite online retrieval.

It fails unless both paths return the same two counts and derived flag.

The sender and sender-location views are retrieved separately and joined back
to each request row before the flag is calculated. This matters when two
transactions have the same sender and timestamp: a single mixed-granularity
Feast request can collapse those repeated sender lookups. The adapter preserves
each transaction and uses the same derivation for historical and online data.

## Run the automated test

```powershell
.\masters_thesis\Scripts\python.exe -m unittest ".\Code snippets\tests\test_sender_location_feature_store.py" -v
```

## Run an isolated smoke test

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\09_sender_location_feature_store.py" --mode smoke
```

The smoke workflow uses the first 50,000 rows and a temporary Feast repository.
It registers, materializes, and compares both retrieval paths without changing
the canonical registry, feature tables, or online database. Its values are not
thesis evidence.

## Build and validate the complete contract

After checking the smoke result:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\09_sender_location_feature_store.py" --mode all
```

This command:

1. recreates and verifies the frozen split;
2. writes the two versioned Parquet state tables;
3. writes schema and lineage metadata;
4. applies the `v2` objects without replacing the `v1` service;
5. materializes the two state views to SQLite;
6. checks offline/online equality; and
7. records the evidence in MLflow.

The full build processes five million rows and may take substantial time and
memory.

## Operational limitation

The canonical prototype materializes the frozen training state required to
reproduce Stage 08. A deployed system would additionally update the sender and
sender-location snapshots after each accepted transaction or micro-batch. That
production update mechanism is outside this stage and must not be claimed as
implemented.
