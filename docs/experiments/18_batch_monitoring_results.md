# Batch monitoring results

## Status

The monitoring engine and rule-based recommendation workflow were implemented and smoke-checked on 10 October 2026. A canonical reference and real post-deployment monitoring run are still pending because no sufficiently large real future batch exists.

## Automated checks

Ten focused tests cover identical-population PSI, shifted-population detection, bin mismatch rejection, missing-schema rejection, invalid-reference rejection, delayed-label and fraud-value metrics, missing-label reporting, human-gated critical decisions, read-only SQLite ledger loading, and training-only history integration.

The monitor does not train, register or deploy a model. It does not read or rescore the consumed test split. Critical drift never triggers automatic retraining.

## Synthetic smoke run

| Item | Result |
| --- | --- |
| MLflow experiment | `financial-fraud-stage-18-batch-monitoring` |
| Run | `fb6bf3f939f645ad89ea2bc161aacf74` |
| Reference / current rows | 10,000 / 5,000 invented rows |
| Maximum PSI | 2.348048 |
| Recommendation | `investigate_before_retraining_decision` |
| Delayed-label calculation | Available for the invented batch |
| Automatic retraining | False |

The shift was deliberately inserted. Geo-anomaly score, time since last transaction, velocity, sender history and spending deviation crossed the critical PSI threshold; amount and model score crossed the warning threshold. The invented fraud-value recall was 51.99%, below the configured 80% research floor. These values show that the rules respond to a known change; they say nothing about the real fraud model or a real operating population.

## Training-history preparation smoke run

| Item | Result |
| --- | --- |
| Initial / reproduced MLflow runs | `4ef6554919034959a1aa69b8ae08a73f` / `e71177954aa04b9585d19a50b3df4f08` |
| Retained training rows | 35,000 |
| Training fraud rows | 1,257 |
| Maximum sender / sender-location prior counts | 2 / 1 |
| New-location rows | 606 |
| Model loaded, fitted or scored | No |
| Reference profile frozen | No |
| Validation/test features retained | No |

The low history counts reflect sparse history in the 50,000-row smoke population and must not be interpreted as the full-data distribution. The runs verify that the canonical-reference path reuses the frozen training-only extraction and strictly earlier, label-independent history logic. Both produced the same aggregate history checksum, `516bd9731caf880e1c467a328db7757c38346fd0db9bd635ba81feacbbf5dc76`, confirming deterministic reproduction of this smoke preparation.

## Current decision

The monitoring component is ready for a properly scoped reference and later operational batch. The current manual API records are too few to support a meaningful distribution comparison. Therefore, no actual drift conclusion, performance-degradation claim or retraining decision has been made.

The next implementation step is to load the unchanged evaluated LightGBM and score the prepared training rows without fitting it. After model identity, feature order and score validity are verified, the aggregate canonical profile can be frozen. The consumed test partition remains excluded. After that, the local API ledger can be monitored in batches, and a Streamlit page can display the aggregate report without exposing raw identifiers.
