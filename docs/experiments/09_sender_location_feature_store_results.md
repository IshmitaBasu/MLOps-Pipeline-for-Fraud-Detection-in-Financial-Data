# Sender-Location Feature Store Results

## Status

The versioned Stage 09 feature-store workflow is complete. The repository
integration test, 50,000-row smoke run, and five-million-row canonical build
all passed.

## Implemented contract

- Feature version: `v2`
- Feature service: `fraud_behaviour_features_v2`
- Sender feature view: `sender_history_features_v2`
- Sender-location feature view: `sender_location_history_features_v2`
- Shared derivation adapter: `derive_new_location_flag_v2`
- Online store: local SQLite
- Original `v1` service retained
- Target excluded from feature sources
- Test events excluded from feature state and development queries

## Automated verification

The focused integration test confirms that:

- only training events update feature state;
- test transactions are excluded from development queries;
- changing fraud labels without changing the frozen partition does not change
  feature values;
- composite sender-location keys are unambiguous;
- current and equal-time transactions are excluded through effective timestamps;
- a known sender at a new location produces the expected flag; and
- historical and SQLite online retrieval return identical values.

## Smoke run

Status: passed.

| Item | Value |
| --- | --- |
| MLflow run ID | `eaed396b9c11472b9a9908f498140b08` |
| Sample rows | 50,000 |
| Frozen smoke split | 35,000 train / 7,500 validation / 7,500 test |
| Sender snapshots | 76,513 |
| Sender-location snapshots | 77,376 |
| Consistency rows | 50 |
| Technical result | Passed |

The smoke workflow materialized both feature views in a temporary SQLite
online store. Historical and online retrieval returned matching values for all
50 checked entity rows. The temporary Feast artifacts were then removed, so
the canonical `v2` tables, registry, and online store were not changed.

This run verifies that the workflow operates correctly on a small sample. Its
split hash is sample-specific and it is not thesis evidence.

## Complete canonical build

Status: passed.

| Item | Value |
| --- | --- |
| MLflow run ID | `177d6af656724b35a8179e669f15edcc` |
| Split hash | `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242` |
| Training events | 3,500,000 |
| Validation events excluded from updates | 750,000 |
| Test events excluded | 750,000 |
| Sender snapshots | 4,392,006 |
| Sender-location snapshots | 6,708,923 |
| Offline/online consistency | Passed for 50 entity rows |

The canonical run applied `fraud_behaviour_features_v2`, materialized both
feature views to the local SQLite online store, and confirmed that historical
and online retrieval produced equal values for all checked rows. The frozen
Stage 07 split hash matched before the artifacts were accepted.

## Interpretation

The completed runs show that the retained features have a versioned definition
and can be registered, materialized, and retrieved consistently through the
tested historical and online paths. Stage 09 created serving infrastructure;
it did not train or improve a model. This remains a local prototype rather than
a production deployment, and the held-out test split was not evaluated.
