# IBM AML dataset feasibility audit

## Status

The audit question and decision rules were fixed on 13 October 2026. The chunked aggregate runner and automated safeguards are implemented, and the corrected complete-file audit is finished. IBM HI-Small passed the technical conditions and is retained as a second-dataset candidate. It has not replaced the current dataset, and no IBM model has been trained.

## Dataset under consideration

| Item | Recorded value |
| --- | --- |
| Dataset | IBM Transactions for Anti Money Laundering, HI-Small transaction file |
| Local file | `data_candidates/ibm_aml/HI-Small_Trans.csv` |
| Local size | 475,664,283 bytes (453.63 MiB) |
| SHA-256 | `B19D39F515523373F991B689C07E11E7B0B95C17A2C27A87D91584AE16C5B040` |
| Source | [IBM AML-Data](https://github.com/IBM/AML-Data) and its linked Kaggle distribution |
| Research description | [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models](https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models) |
| Data licence | Community Data License Agreement – Sharing 1.0 |
| Repository handling | Local candidate data; excluded from Git through `/data_candidates/` |

The file is synthetic and labels money-laundering transactions, not ordinary payment fraud. That distinction affects the thesis wording and must remain visible throughout the comparison.

## Audit question

Is the IBM HI-Small transaction dataset sufficiently documented, internally consistent, computationally manageable and compatible with the planned MLOps system to justify a small modelling feasibility experiment?

Passing this audit does not automatically replace the current dataset. It only permits the next controlled step: a small baseline evaluated under a separately fixed protocol.

## Evidence boundary

The audit is read-only and descriptive. It may calculate aggregate values from the IBM CSV, but it will not:

- change the existing fraud dataset or its pipeline;
- train, tune, register or deploy a model;
- change the current Feast repository, API or monitoring reference;
- compare final test-set results across datasets;
- load the laundering-pattern file as a predictor;
- infer that synthetic labels represent real institutional detection performance.

The current Stage 14 test result and registered research model remain historical evidence for the existing dataset. A later IBM experiment, if approved, starts with new lineage, splits and model evidence.

## Schema handling fixed before inspection

The raw header contains the name `Account` twice. Columns will therefore be interpreted by their documented positions and renamed explicitly:

| Raw position | Audit name | Meaning |
| ---: | --- | --- |
| 1 | `timestamp` | Transaction time |
| 2 | `from_bank` | Sending bank |
| 3 | `from_account` | Sending account |
| 4 | `to_bank` | Receiving bank |
| 5 | `to_account` | Receiving account |
| 6 | `amount_received` | Amount received |
| 7 | `receiving_currency` | Currency received |
| 8 | `amount_paid` | Amount paid |
| 9 | `payment_currency` | Currency paid |
| 10 | `payment_format` | Payment method or format |
| 11 | `is_laundering` | Binary target |

No stable transaction identifier is supplied. If later stages need one, it must be generated deterministically from the verified source checksum and original row position. Account identifiers must never be treated as numeric quantities.

## Checks to be performed

### 1. Reproducibility and scope

- Recalculate the file checksum and size.
- Record the source, version, licence and citation.
- Confirm that the research question can legitimately include money-laundering detection rather than silently relabelling it as ordinary fraud detection.

### 2. Structural integrity

- Count rows and columns without loading the whole file into memory.
- Require exactly eleven columns in every parsed row.
- Confirm that the two account columns are preserved separately.
- Count empty, malformed and exact duplicate rows.
- Confirm that the target contains only `0` and `1`.

### 3. Time and split suitability

- Parse every timestamp and record the minimum and maximum.
- Record whether the raw rows are chronological. If they are not, calculate feasibility independently of row order and require explicit chronological sorting before any later sequential feature construction.
- Summarise rows and laundering prevalence by day.
- Examine transactions outside the documented main activity period because a laundering-only late tail would make time an unintended shortcut.
- Determine whether a random split, chronological split or restricted date window would be defensible. No split will be selected during this audit.

### 4. Label support and imbalance

- Count laundering and legitimate transactions and calculate prevalence.
- Require at least 1,000 positive rows before a full supervised experiment is considered.
- Record that accuracy will be unsuitable if the expected extreme imbalance is confirmed.
- Plan later comparisons around Average Precision relative to prevalence and fixed review workloads, not raw accuracy or ROC-AUC alone.

### 5. Feature and leakage risks

- Summarise amount ranges, currencies and payment formats.
- Check whether received and paid amount/currency relationships provide legitimate information or encode generator rules too directly.
- Exclude `is_laundering` from predictors and keep laundering-pattern membership outside the modelling table.
- Identify constant, nearly constant and implausibly target-deterministic fields.

### 6. Behavioural and graph feasibility

- Measure repeated sender, receiver and sender–receiver activity.
- Calculate the coverage of strictly earlier sender, receiver and pair history.
- Check whether account and bank keys can support Feast entities without exposing them in reports.
- Record whether the short date range is sufficient for meaningful behavioural windows and monitoring examples.

### 7. Local MLOps feasibility

- Record chunk-processing time and peak-memory-safe design assumptions.
- Confirm that a deterministic transaction key, event timestamp, feature schema and binary label can be produced.
- Map the likely offline features, online request fields and history-derived features at a design level only.
- Verify that the raw CSV and generated row-level derivatives remain excluded from Git.

## Predeclared decision rules

### Mandatory conditions for progressing to a baseline

All of the following must hold:

1. The source checksum is stable and the licence/citation are recorded.
2. All data rows have the expected eleven-field structure, or any malformed rows are isolated and fully quantified rather than silently dropped.
3. Timestamps and the binary target can be parsed reliably.
4. At least 1,000 laundering rows are available.
5. No predictor directly contains the laundering label or laundering-pattern membership.
6. Sender and receiver identities provide enough repeated activity to construct past-only features for at least 20% of rows.
7. Any laundering-only late-period tail can be excluded through a rule fixed before modelling.
8. The dataset can be processed on the available computer using bounded chunks and reproducible lineage.
9. The change from fraud detection to money-laundering detection is explicitly accepted within the thesis scope before migration.

### Decision outcomes

| Outcome | Meaning |
| --- | --- |
| `do_not_progress` | A mandatory integrity, leakage, computational or thesis-scope condition fails. The current dataset remains the implementation dataset. |
| `progress_to_small_baseline` | The audit passes. One fixed, validation-only IBM baseline may be designed; no migration is implied. |
| `second_dataset_candidate` | The data is technically useful but the AML target is too different for direct replacement. It may demonstrate architectural reuse if time permits. |
| `replacement_candidate_after_baseline` | The audit and later baseline both support replacement, and the scope change is accepted. A separate migration plan is then required. |

The audit itself can produce only the first three outcomes. `replacement_candidate_after_baseline` requires later predictive evidence and an explicit thesis-scope decision.

## Expected outputs

The implementation will produce aggregate JSON/CSV summaries and a readable Markdown interpretation. It will not export account-level tables or transaction-level labels. The final section of this document will record the command, runtime, observed values, limitations and decision once the audit has run.

Implementation files:

- `stages/20_ibm_aml_dataset_feasibility_audit.py`: cell-runnable smoke/full workflow and MLflow logging;
- `fraud_detection/datasets/ibm_aml_feasibility.py`: strict header handling, chunked aggregates, past-only history coverage and decision rules;
- `tests/test_ibm_aml_dataset_feasibility.py`: synthetic schema, history, quality, privacy and decision safeguards.

## Automated verification

Eight focused Stage 20 tests passed. They verify positional handling of the duplicate `Account` headers, explicit malformed-row accounting, rejection of an unexpected header, exclusion of equal-time events from history, correct past-only coverage when source rows are out of order, visibility of invalid values and a laundering-only tail, aggregate-only outputs, smoke-decision isolation, and the distinction between technical feasibility and dataset replacement.

The complete repository suite also passed after the correction: 131 tests and 20 subtests. Two existing dependency deprecation warnings were reported; neither caused a failure.

## Results

### Smoke audit

The smoke audit was run with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\20_ibm_aml_dataset_feasibility_audit.py" --mode smoke
```

| Item | Smoke result |
| --- | ---: |
| Rows audited | 100,000 |
| Laundering rows | 5 |
| Laundering prevalence | 0.005000% |
| Earlier sender-history coverage | 24.40% |
| Earlier receiver-history coverage | 26.27% |
| Earlier sender-receiver-pair coverage | 19.09% |
| MLflow run | `0a7b2e1b38a643908bd66fb7ac33157d` |
| Recorded decision | `not_evaluated_smoke_test` |

The corrected smoke run confirms that the local file checksum can be verified, the CSV can be processed through the bounded audit workflow, and past-only history can be calculated without relying on CSV row order. Sender and receiver history coverage exceeded the predeclared 20% technical threshold within this sample. The five laundering examples are far too few to judge label support, temporal behaviour or dataset suitability, so none of these observations is treated as thesis evidence or a dataset-selection result. The earlier smoke run `4847e821c606478eb4a7795bef5467d4` is retained in MLflow for traceability but is superseded by this corrected software check.

### Initial full-run diagnostic

The first complete-file execution audited 5,078,345 rows and found 5,177 laundering rows, a prevalence of 0.101943%. The file had no malformed structures, missing cells, invalid timestamps or invalid labels, and all technical checks passed except `source_order_supports_past_only_history`. This produced `do_not_progress` in MLflow run `e13564126b56401e8a057cb6b7320eed`.

Inspection showed that the outcome was caused only by the raw file not being chronologically ordered. Raw row order is a preprocessing concern rather than a reason to reject an otherwise usable timestamped dataset. More importantly, the first implementation had calculated history by source position, so its reported full-file history percentages were not valid strictly-earlier-time evidence.

The history calculation now uses each entity's true minimum timestamp, independently of source position. Transactions sharing that minimum timestamp do not count one another as prior history. The audit also records that chronological sorting will be mandatory before later sequential feature engineering. The initial full run remains documented as a diagnostic and will not be used as the final feasibility decision.

### Final full audit

The corrected complete-file audit produced MLflow run `5bdc8b842b5e4ac5a88925e5935ef6ea`.

| Item | Full-audit result |
| --- | ---: |
| Audited rows | 5,078,345 |
| Structurally malformed rows | 0 |
| Missing cells | 0 |
| Invalid timestamps | 0 |
| Invalid labels | 0 |
| Duplicate full-row fingerprints | 9 |
| Laundering rows | 5,177 |
| Laundering prevalence | 0.101943% |
| Timestamp range | 1 September to 18 September 2022 |
| Primary-period rows | 5,077,237 |
| Primary-period laundering rows | 4,522 |
| Post-primary rows | 1,108 |
| Post-primary laundering rows | 655 |
| Earlier sender-history coverage | 90.1025% |
| Earlier receiver-history coverage | 91.6320% |
| Earlier sender-receiver-pair coverage | 79.7530% |
| Processing time | 277.27 seconds |
| Largest in-memory chunk | 142,843,205 bytes |
| Decision | `second_dataset_candidate` |

All mandatory technical conditions passed. The file identity was verified, all rows had the expected structure, timestamps and labels parsed cleanly, positive-label support exceeded the minimum, and exact past-only history coverage was high. The raw rows are not chronological, so later sequential preparation must explicitly sort by event timestamp. The order-independent audit prevents raw file position from being mistaken for transaction history.

The target remains a major scope limitation: `Is Laundering` represents synthetic money-laundering activity, not general transaction fraud. The 0.101943% prevalence is also substantially more imbalanced than the current dataset. In addition, 655 of the 1,108 post-primary transactions are labelled laundering, making the late period unusually enriched. A baseline must therefore use the already fixed primary-period boundary and must not allow the late tail to become an easy time-based shortcut.

This result permits a separately planned, small IBM baseline. It does not justify replacing the current dataset, comparing the IBM result directly with the completed fraud test result, registering a new model, or changing the API and Feast pipeline.

The next controlled action is to define the IBM baseline question, primary-period population, split, predictors, leakage exclusions, metrics and stopping rule before writing modelling code. No model has been trained, and the existing dataset pipeline remains unchanged.
