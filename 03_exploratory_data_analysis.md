# 03 - Advanced post-cleaning exploratory data analysis

This note documents `03_exploratory_data_analysis.ipynb`.

## Position in the workflow

This notebook runs after:

1. `01_initial_raw_data_eda.ipynb`, which explores the untouched source;
2. `02_data_pipeline_preprocessing.py`, which creates the minimally cleaned gold table.

Its purpose is to validate the cleaned artifact and investigate more complex relationships before the
baseline-model stage. It does not replace the initial EDA and does not train or select a model.

## Input

```text
gold_financial_fraud_detection_table.csv
```

The aligned gold table contains the original usable predictors, `transaction_id` for lineage,
`event_timestamp` for time-aware analysis, and the target. It deliberately preserves missing
`time_since_last_transaction` values and contains no engineered model features.

## Notebook-only analysis variables

The notebook temporarily creates hour, weekday, week, quantile, decile, and missingness-status fields.
These exist only in memory for aggregation and visualization. They are not written to the gold table and
must not be confused with approved model features.

## Advanced analyses and charts

All charts appear inline in the executed notebook. No PNG, SVG, CSV, or other chart files are saved.

The notebook includes:

### Missingness bias analysis

A paired view compares population share and fraud rate for rows where
`time_since_last_transaction` is observed or missing. This highlights the suspicious fact that missingness
occurs only among non-fraud records.

### Target-class empirical distribution functions

An empirical CDF compares the full shape of transaction-amount distributions for fraud and non-fraud
samples on a logarithmic x-axis. This avoids relying on a single histogram binning choice.

### Amount-quantile fraud lift

Twenty equal-frequency amount groups are compared using lift relative to the overall fraud rate. This
tests whether the amount distribution contains practically meaningful fraud concentration.

### Categorical rates with Wilson intervals

Fraud rates for transaction, merchant, location, device, and payment categories are shown with 95%
Wilson confidence intervals. The intervals prevent small rate differences from being presented as strong
effects without uncertainty.

### Hour-by-weekday heatmap

A two-dimensional heatmap examines temporal interactions that separate hourly and weekday summaries
could miss.

### Weekly stability and volume

Weekly fraud rates, a four-week rolling mean, and transaction volume are displayed together to assess
stability, drift, and partial-period effects. These findings support a time-aware model split.

### Numeric correlation matrix

A labeled Pearson matrix examines linear structure among the original numeric predictors and target. It
is descriptive only and is not used for full-data feature selection.

### Anomaly-score interaction surface

A decile-by-decile heatmap examines whether combinations of spending deviation and geographic anomaly
show fraud-rate structure that is hidden in one-dimensional summaries.

## Leakage and experiment-control boundary

No imputer, encoder, scaler, sampler, feature selector, or model is fitted. No analysis-derived parameter
is saved for later model use. Candidate features such as log amount, temporal fields, and a missingness
indicator remain separate tracked experiments after the original-feature baseline.

## Output

The executed `.ipynb` file is the complete analysis artifact. Its tables, charts, and final evidence-to-
implication table are visible directly below the corresponding cells.
