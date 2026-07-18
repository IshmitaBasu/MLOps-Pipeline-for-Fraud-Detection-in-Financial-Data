# Advanced Post-Cleaning EDA Report

## Purpose

This report summarizes the executed analysis in `03_exploratory_data_analysis.ipynb`. The notebook runs
after the initial raw-data EDA and the aligned cleaning pipeline. It validates the cleaned artifact and
investigates more complex patterns before baseline modeling.

## Verified source

- Input: `gold_financial_fraud_detection_table.csv`
- Rows: 5,000,000
- Columns: 13
- Fraud cases: 179,553
- Fraud rate: 3.5911%
- Preserved missing values: 896,513
- Duplicate transaction IDs: 0
- Missing expected columns: 0
- Unexpected columns: 0

The row count, schema, target distribution, and missing-value count agree with the aligned data-quality
report. Unlike the earlier artifact, the gold table is complete and includes the final raw transactions.

## Main findings

### Missingness requires special treatment

`time_since_last_transaction` is missing in 896,513 rows, or 17.9303% of the dataset. None of those rows
are labelled as fraud, producing a 0.0000% fraud rate for the missing group. The observed group has a
4.3756% fraud rate.

This is a strong signal, but its perfect separation pattern suggests possible dataset-generation bias.
A missingness indicator must therefore be evaluated as a separate sensitivity experiment rather than
included automatically in the baseline.

### Amount shows little target separation

The fraud and non-fraud empirical amount distributions are visually very similar. Across twenty
equal-frequency amount groups, the range of fraud lift is only 0.0482. Raw amount should remain in the
baseline, while a log-amount feature should be tested separately.

### Categorical differences are small

Across transaction type, merchant category, location, device, and payment channel, the complete range
of fraud rates is only 0.0613 percentage points. Wilson confidence intervals are displayed because raw
rankings alone can make these small differences appear more meaningful than they are.

### Temporal rates are broadly stable

The four-week rolling fraud-rate range is 0.1838 percentage points. The notebook also shows the final
partial week alongside transaction volume so that a small incomplete period is not mistaken for drift.
The temporal structure supports using a time-aware model split, while hour and weekday remain candidate
features to test after the baseline.

### Raw numeric correlations are negligible

The strongest absolute Pearson correlation between an available raw numeric predictor and the target is
`velocity_score`, at only 0.000370. This does not prove that the variables lack nonlinear predictive
value, but it shows that simple full-data linear relationships are extremely weak. Feature selection must
not be based on these full-data correlations.

## Advanced notebook visualizations

The executed notebook contains eight inline figures:

1. population share and fraud rate by missingness status;
2. amount empirical CDFs by target class on a logarithmic scale;
3. fraud lift across twenty amount quantiles;
4. categorical fraud rates with 95% Wilson confidence intervals;
5. hour-by-weekday fraud-rate heatmap;
6. weekly fraud rate, four-week rolling rate, and transaction volume;
7. annotated numeric correlation heatmap;
8. spending-deviation by geo-anomaly decile interaction surface.

No PNG, SVG, CSV, or other chart files are saved. All charts and supporting tables are embedded directly
inside `03_exploratory_data_analysis.ipynb`.

## Modeling implications

- Use the verified 5,000,000-row gold table for the original-feature baseline.
- Use a time-aware train/validation/test design.
- Fit imputation, categorical encoding, and any scaling on training data only.
- Keep `transaction_id` for lineage but exclude it from predictors.
- Evaluate a missingness indicator through a separate sensitivity run.
- Test log amount and temporal fields as named feature experiments after the baseline.
- Prioritize PR-AUC, precision, recall, F1, confusion matrices, and threshold-aware evaluation over raw accuracy.

## Leakage boundary

The notebook creates temporary bins and temporal fields only for analysis. It does not save them to the
gold table, fit a model-dependent transformation, select features, oversample classes, train a model, or
tune a decision threshold.
