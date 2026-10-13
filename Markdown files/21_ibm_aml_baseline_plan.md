# IBM AML small-baseline plan

## Current position

The Stage 20 full audit accepted IBM HI-Small as a technically feasible second-dataset candidate. It did not approve replacement of the existing fraud dataset. Stage 21 asks a deliberately narrower question: does this AML dataset contain enough transaction-level predictive signal to justify any further implementation work? The validation-only runner, safeguards, smoke check and corrected full validation run are complete. The fixed outcome is `discuss_limited_signal`; the IBM test partition remains untouched.

This document fixes the baseline protocol before model results are available. Stage 21 will not tune hyperparameters, compare many model families, add behavioural history, change Feast, register a model or modify the existing API.

## Baseline question

Can one fixed, class-weighted LightGBM model distinguish laundering from legitimate transactions on the IBM HI-Small primary period substantially better than the no-skill prevalence reference, using only transaction-level fields available at prediction time?

The result will describe synthetic money-laundering detection. It must not be presented as a direct improvement over the completed general-fraud experiment because the datasets, targets and prevalence differ.

## Frozen population

| Rule | Fixed handling |
| --- | --- |
| Source | Checksum-verified `HI-Small_Trans.csv` used in Stage 20 |
| Included period | Timestamps before `2022-09-11 00:00:00` |
| Excluded period | All 1,108 later rows; this tail contains 655 laundering labels and could become an artificial shortcut |
| Invalid rows | Reject execution if the Stage 20 structure, timestamp or binary-label guarantees no longer hold |
| Exact duplicates | Retain the first physical occurrence and record every removed duplicate |
| Row order | Sort deterministically by event timestamp and original source-row number before creating the modelling handoff |
| Transaction identity | Hash of the verified source checksum and original source-row number; never derived from the label |

The Stage 20 audit found 5,077,237 rows and 4,522 laundering labels inside the primary period. Stage 21 must reproduce these pre-deduplication counts before continuing, then separately record the exact rows and labels remaining after duplicate handling.

## Frozen split

A stratified random split with random seed `42` will be assigned after the population rules above:

- training: 70%;
- validation: 15%;
- held-out test: 15%.

The exact assignment and its SHA-256 receipt must be saved. Smoke mode may use a smaller stratified sample only to verify software. It cannot select a model or produce thesis evidence.

The test partition must remain unmaterialised during baseline development. Stage 21 stops after validation reporting. A later one-time test evaluation requires an explicit progression decision based only on the frozen validation result.

## Predictors

The baseline uses fields that would be known when a transaction arrives:

- sending and receiving bank;
- `log1p` received and paid amounts, calculated only after confirming non-negative values;
- receiving and payment currencies;
- payment format;
- hour of day and day of week derived from the event timestamp;
- same-bank, same-currency and equal-amount indicators;

Raw sending and receiving account identifiers are excluded from this first baseline. They are high-cardinality identities and could encourage memorisation. Their usefulness may be tested later only through point-in-time behavioural features if the baseline justifies continuing.

The following are prohibited:

- `Is Laundering` or any transformation of it;
- laundering-pattern files or pattern membership;
- raw source-row position;
- a post-primary or late-tail indicator;
- features calculated from future transactions;
- validation or test labels during preprocessing or fitting.

All learned preprocessing, including category handling and any imputation, must be fitted on training rows only.

## Fixed reference and model

The no-skill reference is validation laundering prevalence. One predictive model will be fitted: a class-weighted LightGBM with the already documented reference parameters, reused without tuning:

| Parameter | Value |
| --- | ---: |
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `min_child_samples` | 100 |
| `subsample` | 0.8 |
| `subsample_freq` | 1 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.0 |
| `random_state` | 42 |

The positive-class weight must be calculated from the training partition only. Reusing a known configuration keeps the experiment small; it does not imply that parameters optimal for the first dataset are optimal for IBM AML.

## Validation evidence

The primary metric is Average Precision because the target prevalence is approximately one case per thousand. The report must show:

- validation prevalence and Average Precision lift over prevalence;
- ROC-AUC as secondary context;
- maximum-F1 threshold selected on validation, with precision, recall, F1, balanced accuracy and confusion counts;
- precision and recall at fixed top 0.1%, 0.5% and 1% review shares;
- laundering-amount capture within each currency containing at least 20 validation laundering rows, without summing unlike currencies into a misleading global amount;
- precision-recall and threshold/workload figures generated from saved validation predictions;
- results for two deterministic ID-hash halves and four fixed 2.5-day slices within the primary period, so one narrow subgroup cannot create a misleading overall result;
- training and scoring time, dependency versions, source checksum, split receipt and feature contract.

Accuracy must not be used as evidence of success. With this imbalance, predicting every row as legitimate would appear highly accurate while detecting no laundering.

## Interpretation and stopping rule

The outcome will be classified without changing the thresholds after seeing results:

| Outcome | Validation interpretation | Next action |
| --- | --- | --- |
| `stop_weak_signal` | Average Precision is below five times validation prevalence, or a fixed stability group has fewer than 20 laundering rows or less than twice-prevalence Average Precision | Stop IBM modelling and retain the existing dataset/system implementation |
| `discuss_limited_signal` | Average Precision reaches five times prevalence but top-1% recall is below 25% | Document the limited gain and discuss whether further work is worthwhile |
| `candidate_for_one_time_test` | Average Precision reaches five times prevalence, top-1% recall is at least 25%, and subgroup evidence does not reveal a collapse | Present the validation result before any one-time IBM test evaluation or pipeline migration |

These are feasibility gates, not claims of production readiness. Passing them does not automatically trigger tuning, registration, deployment or dataset replacement.

## Implementation

Stage 21 adds:

- `21_ibm_aml_baseline_with_mlflow.py`, a cell-runnable entry point with `# %%` cells and `smoke`/`full` modes;
- `ibm_aml_baseline_utils.py`, containing source preparation, split, feature, model, metric, stability and decision functions;
- `tests/test_ibm_aml_baseline.py`, covering the time boundary, duplicate handling, deterministic identity, split isolation, rare-class rejection, train-only class weighting, prohibited fields, currency separation, smoke ineligibility and fixed decisions;
- aggregate MLflow summaries, workload and stability tables, figures, feature importance and a full-run validation model artifact without registration.

The raw IBM file, split-level rows, account identifiers and prediction-level outputs must remain local and excluded from Git.

## Automated verification

Ten focused Stage 21 tests pass. They cover the primary-period boundary, duplicate removal, account-ID exclusion, chronological preparation, deterministic transaction identity, reproducible 70/15/15 assignment, the absence of a test-frame return path, minimum rare-class support, training-only class weighting, prohibited predictors, currency-separated amount reporting, every fixed progression outcome, and fitted-model save/load prediction parity using the pinned serialization format.

The complete repository suite also passes: 141 tests and 20 subtests. Two existing dependency deprecation warnings remain; neither caused a failure.

## Commands

From the thesis directory, the smoke workflow is:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\21_ibm_aml_baseline_with_mlflow.py" --mode smoke
```

The full validation workflow is:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\21_ibm_aml_baseline_with_mlflow.py" --mode full
```

## Smoke result

An initial 100,000-row integration attempt stopped before model fitting because the prefix contained only five laundering cases, which could not support a safe three-way stratified split. The smoke prefix was therefore increased to 500,000 rows, and an explicit minimum-positive guard was added. This changed only the software-check sample size, not the full-data protocol.

The corrected smoke run used 350,000 training rows, 75,000 validation rows and 75,000 untouched test rows. It produced Average Precision `0.010888`, a `28.16x` lift over the smoke validation prevalence, and `44.83%` recall within the top 1% of validation scores. MLflow run `64b7837816b942b69bfd9bfdaeaa6305` recorded the aggregate evidence. These values only show that the workflow operates end to end: the validation set contained 29 laundering cases, so the smoke metrics are not thesis evidence and cannot trigger progression.

## First full-run attempt and logging correction

The first full attempt fitted and evaluated the fixed model on 3,554,059 training rows and 761,584 validation rows. The 761,585-row test partition remained unmaterialised and unevaluated. Aggregate evidence was written to MLflow run `c1251808dace4b65b15910ec99d9a272` before model-artifact saving failed. MLflow therefore marked the run as failed, so it is retained as diagnostic evidence rather than the final Stage 21 run.

The recovered validation values were Average Precision `0.006142`, `6.90x` lift over prevalence, ROC-AUC `0.853245`, and `9.00%` recall within the top 1% of scores. All six stability groups passed, but top-1% recall did not reach the fixed 25% gate; the calculated outcome was `discuss_limited_signal`. A clean rerun was required and is recorded below.

The failure occurred after evaluation because MLflow 3.14 defaulted to `skops`, which rejected the LightGBM-specific types as untrusted. Model logging now uses the repository's pinned `cloudpickle` format and the non-deprecated `name` argument. Integer-derived signature fields are represented as doubles to avoid nullable-integer schema problems. An automated round-trip test confirms that the fitted pipeline can be saved, loaded and used to reproduce predictions.

## Final full-validation result

The corrected full workflow completed successfully in MLflow run `94ccbdf8542f41f88b8b9fbb6d694db2`.

### Population and split

| Item | Result |
| --- | ---: |
| Rows before the fixed time boundary | 5,077,237 |
| Exact duplicates removed | 9 |
| Final modelling rows | 5,077,228 |
| Laundering rows | 4,522 |
| Training rows | 3,554,059 |
| Training laundering rows | 3,165 |
| Validation rows | 761,584 |
| Validation laundering rows | 678 |
| Held-out test rows | 761,585 |
| Test evaluated | No |
| Split receipt | `d968641c3003779c01ba197b9635a09aae7e99e7e101dea9c4cc58aadeafd5cf` |

### Validation metrics

| Metric | Result |
| --- | ---: |
| Laundering prevalence | 0.089025% |
| Average Precision | 0.006142 |
| Average Precision lift over prevalence | 6.90x |
| ROC-AUC | 0.853245 |
| Maximum-F1 threshold | 1.000000 |
| Precision at maximum F1 | 0.7335% |
| Recall at maximum F1 | 71.2389% |
| F1 | 0.014521 |
| Accuracy | 91.3920% |
| Balanced accuracy | 81.3245% |
| True negatives | 695,544 |
| False positives | 65,362 |
| False negatives | 195 |
| True positives | 483 |

The apparently strong accuracy is mostly a consequence of the extreme imbalance and is not evidence of an effective detector. At the maximum-F1 rule, the model finds 483 of 678 laundering cases but produces 65,362 false alerts. Put differently, fewer than 1 in 136 alerts is a laundering case.

### Fixed review workloads

| Transactions reviewed | Rows reviewed | Laundering cases captured | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.1% | 761 | 6 | 0.7884% | 0.8850% |
| 0.5% | 3,807 | 31 | 0.8143% | 4.5723% |
| 1.0% | 7,615 | 61 | 0.8011% | 8.9971% |

The top-1% review workload captures only 61 of 678 laundering cases. This is much better than random ranking for such a rare target, but it remains below the predeclared 25% recall requirement. Currency-specific amount capture varied sharply and is retained only as within-currency evidence; amounts in different currencies were never summed into a global value claim.

### Stability and decision

Both deterministic ID-hash halves achieved about `6.9x` Average Precision lift. The four fixed time slices achieved lifts between `5.96x` and `7.03x`. All six stability groups passed the twice-prevalence floor, so the result is not driven by one narrow partition.

The final outcome is `discuss_limited_signal`:

- the `5x` Average Precision lift gate passed;
- all fixed stability checks passed;
- the top-1% recall gate failed: `9.00%` observed versus `25%` required.

This supports a limited-signal conclusion, not a claim that IBM HI-Small solves the predictive-quality problem. It also does not permit automatic test evaluation, tuning, dataset replacement or MLOps migration.

### Saved model boundary

The fitted validation model is stored as MLflow logged model `m-8b08d36dbeee461e98c3cf627755fb00`, linked to the successful run. It uses `cloudpickle` because MLflow's default safe serializer did not accept LightGBM-specific types. The warning shown during saving is expected: pickle-based artifacts must be loaded only from this trusted local experiment. The model is not registered, approved or deployed.

## Immediate next action

Share the completed validation evidence before deciding whether any further IBM experiment is justified. Do not evaluate the IBM test partition, tune the model or modify the existing pipeline while that decision is open.
