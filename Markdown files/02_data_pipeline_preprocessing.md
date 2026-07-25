# 02 - Data cleaning and preparation

This note documents `02_data_pipeline_preprocessing.py` and connects its decisions directly to the
findings in `01_initial_raw_data_eda.ipynb`.

## Position in the workflow

The initial EDA happens first. This second stage applies only cleaning and column-selection decisions
supported by that analysis, domain constraints, or prediction-time availability. It does not perform
model feature engineering.

The output is a minimally cleaned handoff table for:

- post-cleaning exploratory analysis;
- the original-feature baseline;
- later tracked feature-engineering experiments.

## Input and outputs

Input:

```text
financial_fraud_detection_dataset.csv
```

Outputs:

```text
gold_financial_fraud_detection_table.csv
data_quality_report_v1.md
```

The gold CSV is written to a temporary file first and replaces the final file only after the write
finishes successfully. This prevents an interrupted run from intentionally replacing the final table
with a partially written file.

## How the initial EDA informs preparation

| Initial EDA finding | Preparation action |
| --- | --- |
| All timestamps are valid ISO-8601 values, with mixed fractional-second precision. | Parse with `format="ISO8601"` and preserve all valid rows. |
| No exact duplicates or duplicate transaction IDs were found. | Keep the duplicate checks, but do not claim that rows were removed when the count is zero. |
| No negative or zero transaction amounts were found. | Keep the validation check; no amount rows should be removed in the current data. |
| `fraud_type` is present for every fraud case and absent for every non-fraud case. | Exclude it because it is prediction-time target leakage. |
| Account, IP, and device identifiers have high cardinality. | Exclude `sender_account`, `receiver_account`, `ip_address`, and `device_hash` from the initial modeling table. |
| `time_since_last_transaction` is missing in 17.93% of rows and only among non-fraud cases. | Preserve the missing values. Fit imputation on training data and test a missingness flag separately. |
| Amount is strongly right-skewed. | Keep raw `amount`; test `amount_log1p` only after recording the baseline. |
| Timestamps contain potentially useful temporal context. | Keep `event_timestamp`; test hour, weekday, and month as later feature experiments. |

## Pipeline order

The `main()` function performs:

1. path construction and a five-row input preview;
2. full raw-data loading with explicit data types;
3. schema validation and quality profiling;
4. duplicate and negative-amount checks;
5. ISO-8601 timestamp parsing and validation;
6. minimal cleaned-column selection;
7. quality-report generation;
8. atomic gold-table writing;
9. output verification.

## Cleaned table columns

The output contains 13 columns:

```text
transaction_id
event_timestamp
transaction_type
merchant_category
location
device_used
payment_channel
amount
time_since_last_transaction
spending_deviation_score
velocity_score
geo_anomaly_score
is_fraud
```

`transaction_id` is retained for lineage and auditability but must not be passed to the model as a
predictor.

## Deliberately postponed

The cleaning pipeline does not perform:

- missing-value imputation;
- a missingness-indicator feature;
- `amount_log1p`;
- hour, weekday, or month feature creation;
- categorical encoding;
- scaling;
- resampling or SMOTE;
- feature selection;
- model training;
- threshold tuning.

Imputation, encoding, and scaling must be fitted using training data only. Candidate engineered features
must be introduced through named experiments after an original-feature baseline has been recorded.

## Expected result for the current source

The initial EDA found no rows that require removal. Therefore, the aligned pipeline is expected to
preserve all 5,000,000 transactions. If the source data changes, the report must explain every difference
between the input and output row counts.
