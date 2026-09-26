# Stage 07 Predictive-Quality Improvement Results

## Status

The training-only behavioural-feature diagnostic has been implemented, and its
50,000-row smoke run completed successfully on 26 September 2026. No Stage 07
feature result has been accepted yet because the full-data diagnostic has not
been run.

## Why this diagnostic comes first

The supervisor asked whether the planned historical features show a noticeable
relationship with fraud before substantial engineering work is invested. The
diagnostic therefore measures repetition, prior-history coverage, new-versus-
returning fraud rates, and sender amount deviation on training data only.

## Implemented safeguards

- The revised protocol uses one deterministic, stratified 70/15/15 random split.
- The diagnostic functions receive only the training partition.
- The validation and test partitions are not used for feature exploration.
- Historical counts use strictly earlier transactions and exclude the current row.
- Fraud labels are not used to construct historical features.
- Split lineage and diagnostic tables are stored as MLflow artifacts.
- Smoke runs are labelled as technical checks and cannot be treated as thesis evidence.

## Smoke run

Status: completed successfully on 26 September 2026.

| Item | Value |
| --- | --- |
| Command | `--mode diagnostics --sample-rows 50000 --skip-data-hash` |
| MLflow run ID | `2358850955a7496c993879ae3c947b2f` |
| Split assignment hash | `d2fa01494bddbe169a360a4f8523c5cb58a11dd4486efe0b93df64a597a38083` |
| Training rows | 35,000, including 36 fraud cases |
| Validation rows held aside | 7,500, including 8 fraud cases |
| Test rows held aside | 7,500, including 8 fraud cases |
| Test evaluated | No |
| Technical outcome | Passed; the split, diagnostic calculations, and MLflow artifact logging completed |

The first 50,000 source rows contained only 52 fraud cases, a fraud rate of
0.104%. This is far below the complete dataset's fraud rate of approximately
3.59%. The smoke sample is therefore not representative and its feature gate
must not be used for the thesis decision.

The following history coverage values were observed in the smoke training
partition:

| Entity | Rows with prior history | History coverage | Passed gate |
| --- | ---: | ---: | --- |
| Sender account | 663 | 1.8943% | No |
| Receiver account | 624 | 1.7829% | No |
| Device hash | 68 | 0.1943% | No |
| IP address | 0 | 0.0000% | No |
| Sender-receiver pair | 0 | 0.0000% | No |
| Sender-device pair | 0 | 0.0000% | No |
| Sender-IP pair | 0 | 0.0000% | No |
| Sender-location pair | 87 | 0.2486% | No |

No entity passed the smoke gate. Sender and receiver histories showed large
apparent relative fraud-rate differences, but the returning groups contained
only six and two fraud cases respectively. The minimum evidence requirement is
100 fraud cases in both groups, so these differences are too unstable to
interpret. Entities with no returning rows had no calculable rate difference.

The sender amount-deviation bands were based on only 663 rows with earlier
sender history and six fraud cases. Their fraud-rate differences are likewise
too sparse to support a conclusion.

This run demonstrated that all expected MLflow artifacts were created:
`split_manifest.json`, `feature_feasibility_decision.json`,
`entity_repetition_summary.csv`, `new_vs_returning_fraud_rates.csv`, and
`sender_amount_deviation_summary.csv`.

## Full-data diagnostic

Status: completed successfully on 26 September 2026.

| Item | Value |
| --- | --- |
| MLflow run ID | `95a914f6b68f4faba00dc2132e6ec790` |
| Split assignment hash | `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242` |
| Training rows | 3,500,000, including 125,687 fraud cases |
| Validation rows held aside | 750,000, including 26,933 fraud cases |
| Test rows held aside | 750,000, including 26,933 fraud cases |
| Test evaluated | No |
| Promising entities | `sender_location_pair` |
| Recommendation | Engineer and validate the selected history features |

## Entity results

| Entity | Prior-history rows | Coverage | New fraud rate | Returning fraud rate | Relative difference | Evidence sufficient | Passed gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Sender account | 2,618,491 | 74.8140% | 3.3929% | 3.6578% | 7.2401% | Yes | No |
| Receiver account | 2,618,217 | 74.8062% | 3.5633% | 3.6004% | 1.0288% | Yes | No |
| Device hash | 600,810 | 17.1660% | 3.5930% | 3.5818% | 0.3100% | Yes | No |
| IP address | 1,424 | 0.0407% | 3.5911% | 3.3708% | 6.1362% | No | No |
| Sender-receiver pair | 6 | 0.0002% | 3.5911% | 0.0000% | 100.0000% | No | No |
| Sender-device pair | 0 | 0.0000% | 3.5911% | Not available | Not available | No | No |
| Sender-IP pair | 0 | 0.0000% | 3.5911% | Not available | Not available | No | No |
| Sender-location pair | 729,540 | 20.8440% | 3.7121% | 3.1316% | 15.6380% | Yes | Yes |

Sender and receiver accounts repeat frequently, but the difference in fraud
rate between new and returning entities did not reach the predefined 10%
relative threshold. Device history showed almost no separation. IP and the
other high-cardinality pairs were too sparse to support useful history.

The sender-location pair was the only entity to meet all three requirements. A
prior sender-location relationship existed for 20.844% of training rows, both
groups contained far more than 100 fraud cases, and their fraud rates differed
by 15.638% in relative terms. Transactions from a sender-location combination
without prior history had the higher fraud rate: 3.7121%, compared with 3.1316%
when the pair had appeared earlier.

## Amount-deviation result

The sender-relative-amount diagnostic covered 2,618,491 transactions with
earlier sender history. Fraud rates across its ten equal-frequency bands ranged
from approximately 3.5837% to 3.7086%. The variation was small and did not show
a clear monotonic pattern as relative amount increased. Amount deviation will
therefore not be prioritised from this diagnostic alone.

This is exploratory training evidence, not validation or test performance.

## Inference and next decision

The full-data result supports a small sender-location history experiment. The
next feature group should distinguish among:

- the sender's number of strictly earlier transactions;
- the sender-location pair's number of strictly earlier transactions; and
- whether a previously observed sender is using a location not previously seen
  for that sender.

These features must be calculated point in time without current rows, future
transactions, or fraud labels. They should first be compared with `original_v1`
using the same fixed reference model. Validation evidence will determine whether
they improve prediction; passing this diagnostic does not itself prove model
improvement.

The validation and test partitions were not analysed during this diagnostic.
The held-out test remains closed.
