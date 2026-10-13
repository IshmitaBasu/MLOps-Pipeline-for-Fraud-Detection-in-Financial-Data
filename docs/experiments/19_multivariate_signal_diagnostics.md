# Training-only multivariate predictive-signal diagnostic

## Why this stage exists

Stage 17 found weak marginal separation in most individual fields, a clear association with missing `time_since_last_transaction`, and modest behavioural-history associations. A weak one-variable result does not rule out a useful combination of variables. Stage 19 therefore checks whether several fields used together separate fraud from legitimate transactions, and whether a few plausible interactions add information.

This is a focused data investigation rather than another broad model search. The fixed LightGBM configuration is used as a measuring instrument because it can represent nonlinear combinations. It is not tuned, compared for deployment, registered or sent to the API.

## What is compared

All rows come from the frozen training partition. That partition is divided reproducibly into an 80% internal development subset and a 20% internal audit subset, with the fraud proportion preserved. The words *development* and *audit* refer only to subdivisions of training; neither the project validation partition nor the consumed final test partition is loaded for feature analysis.

The comparison contains:

1. Fraud prevalence, representing no discrimination.
2. Whether `time_since_last_transaction` is observed, representing the clearest Stage 17 marginal pattern.
3. Core transaction, category and past-only behavioural fields, excluding the three supplied risk scores.
4. All supplied main fields, adding `spending_deviation_score`, `velocity_score` and `geo_anomaly_score`. Their source and calculation remain undocumented, so their contribution is shown separately.
5. The same fields plus four predeclared combinations: location × device, location × payment channel, transaction type × payment channel, and transaction type × time-since-last status.
6. A shuffled-label control. It keeps the same features and class balance but breaks the true feature–label relationship. Performance close to this control would indicate that apparent discrimination is weak.

For example, `location = Berlin` may have little meaning alone and `device = mobile` may have little meaning alone, while the combination `Berlin | mobile` could occur at a different fraud rate. The interaction columns let the diagnostic test that possibility directly. Fraud labels are never used to construct them.

Average Precision is the primary measure because the data are imbalanced. ROC-AUC, precision at a fixed 5% review workload, fraud-count recall and fraud-value recall are also recorded. The two multivariate candidates are checked across four time slices and two deterministic transaction-ID halves of the internal audit subset. These slices are stability checks inside training, not independent validation.

The prevalence and time-observed baselines contain tied scores, so selecting an exact top 5% from them would depend on arbitrary row order. Their fixed-workload cells are therefore left undefined; the workload metrics are reported only for model scores that provide a meaningful ranking.

## Safeguards and interpretation

- The full run verifies the raw source checksum and frozen split hash.
- Only frozen training features are retained by the chunked loader.
- `fraud_type`, the fraud label, IDs, timestamps and sender identifiers cannot enter a candidate feature set.
- Earlier-event history is label-free, excludes events at the same timestamp, and counts only earlier internal-development events. Internal-audit events cannot feed back into development history.
- Preprocessing is fitted only on the internal development subset.
- The audit subset is untouched during fitting. No threshold is selected.
- Only aggregate metrics, feature importances and figures are saved; transaction-level predictions and identifiers are not exported.
- The registered model, frozen operating rule, completed test evidence and API remain unchanged.

An explicit-interaction AP increase below 0.001 is described as no clear incremental signal; 0.001–0.005 as small exploratory signal; and at least 0.005 as a larger exploratory signal that would still require a separately approved validation design. These labels make the report readable; they are not model-selection rules.

## How to run

From the thesis root, run the software check first:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\19_training_multivariate_signal_diagnostics_with_mlflow.py" --mode smoke
```

If that succeeds, run the complete frozen training partition:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\19_training_multivariate_signal_diagnostics_with_mlflow.py" --mode full
```

The MLflow experiment is `financial-fraud-stage-19-training-multivariate-signal`. Each run saves `candidate_metrics.csv`, `stability_slice_metrics.csv`, transformed feature importances, the split and lineage receipts, a concise report, two figures, library versions and source snapshots.

## Results log

### Smoke run

The corrected smoke workflow completed successfully on 12 October 2026. It used 35,000 frozen training rows, divided into 28,000 internal development rows and 7,000 internal audit rows. The audit subset contained 251 fraud cases, a rate of 3.5857%. Validation and test features were not retained or analysed.

| Candidate | Internal-audit Average Precision | ROC-AUC | Interpretation |
| --- | ---: | ---: | --- |
| Fraud prevalence | 0.035857 | 0.500000 | No-discrimination reference |
| Time-since-last value is observed | 0.043523 | 0.591347 | Strongest Stage 17 marginal pattern |
| Core fields without supplied risk scores | 0.044837 | 0.587738 | Small improvement over the marginal pattern |
| All supplied main fields | 0.048657 | 0.601785 | Higher AP after adding the supplied risk scores |
| All supplied fields plus explicit interactions | 0.051328 | 0.619574 | Highest smoke AP; increase of 0.002671 over main fields |
| Shuffled-label control | 0.044019 | 0.525167 | Weak discrimination after breaking the true label relationship |

The multivariate main fields exceeded the time-missingness pattern by 0.005133 AP in this sample. The explicit interactions added 0.002671 AP and improved on the main-fields model in five of six internal stability slices. The true-label interaction model exceeded its shuffled-label control by 0.007309 AP. This is a small exploratory interaction signal, not evidence of a deployable improvement. The sample is too small to establish its size or stability.

The final smoke run is `a33f65f9f3d54678a1e990339b4bdfbe` in `financial-fraud-stage-19-training-multivariate-signal`. It builds both development and audit history from earlier internal-development events only. Internal-audit events cannot affect development history. Earlier development runs `85a6cb4c56fc4fcc8e81b2c06720442c` and `19034eb150c94eef804dd9c9d572f655` led to two safeguards: tied baseline workload values are now left undefined, and audit events are excluded from all history counts. Neither the earlier runs nor the final smoke run is thesis evidence.

The complete repository suite passed after the final implementation changes: 123 tests and 20 subtests. Two existing dependency deprecation warnings were reported; neither caused a failure. `pytest==9.0.2` is pinned so the documented test command works in a fresh installation.

### Full run

The full workflow completed on 12 October 2026 as MLflow run `8b3b5e25606c45d987cd6ef4dce8a657`. It used all 3,500,000 frozen training rows: 2,800,000 internal-development rows and 700,000 internal-audit rows. The audit subset contained 25,137 fraud cases, a prevalence of 3.5910%. The raw source checksum, frozen split hash and training-membership hash matched the earlier evidence. Validation and test features were excluded.

| Candidate | Average Precision | ROC-AUC | Precision at 5% | Fraud-count recall at 5% | Fraud-value recall at 5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fraud prevalence | 0.035910 | 0.500000 | — | — | — |
| Time-since-last value is observed | 0.043753 | 0.592968 | — | — | — |
| Core fields without supplied risk scores | **0.046129** | 0.608847 | **4.8171%** | **6.7072%** | **7.3410%** |
| All supplied main fields | 0.046004 | **0.609567** | 4.6800% | 6.5163% | 6.6596% |
| All supplied fields plus explicit interactions | 0.046112 | 0.609385 | 4.7543% | 6.6197% | 6.3262% |
| Shuffled-label control | 0.035756 | 0.496709 | 3.6486% | 5.0802% | 6.2838% |

The real-label models clearly exceed the shuffled-label control, so the dataset contains multivariate predictive signal. The signal remains weak: the best AP is 0.046129 against 0.035910 prevalence, and a fixed 5% review workload finds only 6.71% of fraud cases.

The supplied risk-score fields did not improve AP or the 5% workload results. Adding them reduced AP by 0.000125 compared with the core fields. Adding the four explicit interactions then recovered only 0.000108 AP relative to all supplied main fields and remained 0.000017 below the simpler core model. The interaction candidate exceeded the main-fields candidate in four of six internal stability slices, which is not consistent enough to support a robust interaction claim.

The full-data conclusion is therefore **no clear incremental interaction signal** for the four tested combinations. This answers the multivariate-pattern question without claiming that every possible interaction has been exhausted. It also supports the earlier explanation for weak model performance: the available predictors contain some genuine ranking information, but not enough separation to produce strong fraud precision or recall.

This training-only diagnostic does not reopen model selection. The retained LightGBM, frozen operating rule, completed final test evaluation, registered research model and API remain unchanged.
