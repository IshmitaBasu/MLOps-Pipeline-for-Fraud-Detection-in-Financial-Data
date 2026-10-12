# Training-only predictive-signal diagnostics

## Question and relationship to earlier work

Does the training data contain practically useful, reasonably stable fraud/non-fraud associations, and are apparent patterns affected by data quality or unverified generation rules?

Earlier EDA described the full dataset before modelling. It found small categorical and numerical differences, and a much stronger association with missing `time_since_last_transaction`. Stage 07 focused on entity-history feasibility; Stage 08 validated a selected history feature set. Stage 17 revisits these questions within the frozen training partition, extends the missingness and stability investigation, and adds only two predeclared interactions. It does not repeat model screening or assume that another classifier will resolve weak performance.

The training-only scope describes this new diagnostic. It does not undo the earlier full-dataset EDA or imply that the project's historical data exploration was confined to training rows.

## Files

- `17_training_predictive_signal_diagnostics_with_mlflow.py`: executable `# %%` runner and MLflow logging.
- `frozen_training_data.py`: shared split/source verification and chunked extraction that retains training features only; Stage 18 reference preparation reuses this boundary.
- `predictive_signal_utils.py`: descriptive calculations, label-free earlier-event history and aggregate plots.
- `tests/test_predictive_signal.py`: synthetic arithmetic, timing, extraction/isolation, malformed-input and plotting checks.
- This plan: scope, methods, commands and interpretation rules.
- `17_predictive_signal_diagnostics_results.md`: actual execution evidence and interpretation; populated after runs, not with anticipated findings.

## Scope and safeguards

Full mode requires the existing five-million-row raw source checksum and frozen stratified 70/15/15 split hash. Only transaction IDs and labels are used to reconstruct canonical membership. Raw CSV feature columns are read in bounded chunks; only selected training rows are retained or analysed. No validation/test diagnostic frame, predictions, feature distributions, associations or costs are produced. Partition row totals are administrative metadata, not held-out analysis.

Smoke mode uses a deterministic stratified 50,000-row sample selected from membership metadata before creating a sample partition. It is not the earlier first-N-rows diagnostic and need not use the same sampled IDs as other stages. Sample history is sparse and not representative of full-population history. Smoke outputs are software checks only.

The three existing history features count strictly earlier training events only. Fraud labels do not enter their construction, and equal-time events do not count as prior history. No models are fitted, tuned, registered or deployed. Stage 14 evidence and the Stage 15/16 model remain unchanged.

## Methods and examples

1. **Quality and label screens:** count missing values, malformed numeric text, infinities, negative amounts, invalid timestamps and repeated selected transaction content. Different labels for the same selected observables are a screen, not proof of wrong labels. Invalid IDs/labels prevent safe splitting and cause a failure rather than silent row deletion. Invalid timestamps/sender keys prevent reliable history construction and also stop the run. `fraud_type` is audited as an outcome-related field and excluded from predictors.
2. **Category and missingness rates:** show group counts, fraud counts, coverage, fraud lift, rate differences and nominal 95% Wilson intervals. For example, a 5% fraud rate against a 3.6% baseline is about 1.39 times the baseline, but counts and uncertainty still matter. Examine time-since-last missing versus observed groups explicitly.
3. **Numerical separation:** report class means, standard deviations and quantiles, finite-only Pearson/point-biserial correlation, and total variation between class distributions across up to ten training-derived quantile bins. Binary and low-cardinality numeric values remain separate groups; constant and entirely missing features are handled explicitly. Plots show class quantile/CDF approximations over the 0.1–99.9% finite range, not every empirical observation. Missing/nonfinite groups remain visible in rate tables rather than being silently imputed.
4. **Bounded combinations:** transaction type × amount quintile and payment channel × amount quintile only. All bin boundaries are derived from training amounts and recorded. No exhaustive interaction search or automated feature selection occurs.
5. **Stability:** compare each predeclared grouping across four equal-duration training time slices and two deterministic transaction-ID hash halves. Record slice-specific baseline rates and group support. Training-wide bin edges stay fixed across slices; this is exploratory description, not nested validation or independent replication. Check whether the same groups have similar differences, not just which slice has the largest rate.
6. **Practical effects and uncertainty:** uncorrected Cramér's V describes categorical/binned association; numerical binned total variation ignores ordering within bins and is not a complete nonlinear-information measure. Groups below 1,000 rows or 30 fraud cases are flagged, not discarded. Zero fraud among many rows can still be informative: interpret its interval upper bound, not the low-fraud-count flag alone. No p-value-driven selection is performed. Wilson intervals are nominal marginal intervals; dependence, rare groups and multiple comparisons limit their interpretation.

Missingness could be informative, a generation artifact, or both. For example, zero fraud among missing entries does not prove that missing entries in real banking data are safe. Local documentation calls the source synthetic, but no generator or authoritative source specification is verified by this script. Original label rules and derivations/availability of supplied risk scores remain explicit provenance questions; they are not inferred from correlations.

## Run and inspect

From the thesis root:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\17_training_predictive_signal_diagnostics_with_mlflow.py" --mode smoke
```

After checking the smoke run:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\17_training_predictive_signal_diagnostics_with_mlflow.py" --mode full
```

Both commands scan the raw CSV to identify training rows; smoke mode avoids full source hashing but still reads membership metadata and feature chunks. Full mode verifies the raw checksum against the existing pre-test Stage 13 source record, reconstructs the frozen split, and checks the source again after extraction. These scans do not score or analyse held-out feature values. No new dependency is required.

MLflow experiment: `financial-fraud-stage-17-training-predictive-signal`. Artifacts include the summary/quality audit, training lineage, provenance audit, group-rate CSV, numeric class quantiles, numeric effects, categorical/binned associations, stability group rates, a readable interpretation report, four aggregate figures, library versions and source snapshots. Transaction-level records and account identifiers are not exported. Results remain descriptive and ineligible for automatic model selection.

## Decision boundary

The intended output is a reasoned account of promising associations, weak separation, support/stability limitations and unknown provenance. It does not prove robust prediction, causality, dataset inadequacy or a particular generation rule. Any feature/missingness sensitivity experiment comes later under a separately defined validation protocol. Dataset replacement, new final-performance claims, registration and API switching are separate decisions. The consumed test partition remains closed to further tuning and repeated evaluation.
