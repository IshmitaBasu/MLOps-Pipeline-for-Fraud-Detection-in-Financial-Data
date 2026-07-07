# 01 - Data pipeline preprocessing

This note documents `01_data_pipeline_preprocessing.py`.

The script prepares the public financial fraud dataset for the later machine learning pipeline. It does not train a model and it does not make decisions that should depend on a train-test split. Its job is more specific: keep the raw input traceable, check that the dataset has the expected structure, apply basic cleaning, create simple row-level features, and save the result for the next step.

The main reason for keeping this preprocessing step separate is leakage control. Anything that learns from the distribution of the full dataset, such as scaling, learned imputation, one-hot encoding, SMOTE, feature selection, or threshold tuning, is left for the ML pipeline after the data has been split.

## Input file

The script expects the raw CSV file:

```text
financial_fraud_detection_dataset.csv
```

By default, it looks for this file in the `Code snippets/` folder. If the file is not found there, `build_paths()` falls back to the current working directory. This makes the script easier to run both from the project folder and from the folder that directly contains the CSV.

Run the script with:

```bash
python "Code snippets/01_data_pipeline_preprocessing.py"
```

All output folders are created automatically.

## Pipeline structure

The pipeline is coordinated by `main()`. The flow is:

```python
paths = build_paths()
keep_raw_copy(paths)
read_sample(paths.raw_copy)

df = load_dataset(paths.raw_copy)
df = standardise_columns(df)
validate_schema(df)
profile_dataset(df)

df_clean, cleaning_stats = apply_safe_cleaning(df)
df_clean, cleaning_stats = parse_event_timestamp(df_clean, cleaning_stats)

df_features = create_leakage_safe_features(df_clean)
df_features = select_final_columns(df_features)

processed_file, feature_store_file = save_processed_datasets(df_features, paths)
schema_file = save_feature_schema(df_features, paths)
metadata_file = save_preprocessing_metadata(df_features, paths, cleaning_stats)
quality_report_file = save_quality_report(df_features, paths, cleaning_stats)

verify_outputs(df_features, [processed_file, feature_store_file, schema_file, metadata_file, quality_report_file])
```

The functions are kept small on purpose. Each one has a clear role, which makes the script easier to inspect, rerun, and adapt if the dataset changes.

## Project folders

`PipelinePaths` stores all relevant paths in one typed object. `build_paths()` then creates the folder structure used by the script:

```text
data/raw/
data/interim/
data/processed/
feature_repo/data/
feature_repo/metadata/
reports/
```

The raw copy is stored in `data/raw/`. The processed feature table is saved twice: once in `data/processed/` and once in `feature_repo/data/`. The second copy is useful because later notebooks or scripts can treat it as the stable handoff from preprocessing.

`data/interim/` is created even though this version of the script does not write intermediate files. Keeping the folder in the structure leaves room for later preprocessing outputs without changing the project layout.

## Raw data handling

`keep_raw_copy()` copies the original CSV into `data/raw/`, but only if the copy does not already exist. The original input file is not changed.

This matters for reproducibility. If a later result needs to be checked, the processed feature table can be traced back to the exact raw file that was used at the start of the pipeline.

`read_sample()` reads the first five rows and prints the column names. This is a quick sanity check before loading the full file. `load_dataset()` then reads the full CSV with the data types defined in `DTYPE_MAP`.

Fixed data types are used because columns such as IDs, account numbers, IP addresses, and categorical fields should not be accidentally converted into numeric values. The target column is first read as boolean and then converted to `0` and `1` by `standardise_columns()`.

## Schema and profiling checks

`validate_schema()` checks that the input data has exactly the expected columns. The script stops if a column is missing or if an unexpected column appears.

The expected input columns are:

```text
transaction_id
timestamp
sender_account
receiver_account
amount
transaction_type
merchant_category
location
device_used
is_fraud
fraud_type
time_since_last_transaction
spending_deviation_score
velocity_score
geo_anomaly_score
payment_channel
ip_address
device_hash
```

This strict check is useful because a silent schema change could affect the later model pipeline. It is better for the script to fail early than to produce a feature file with different meaning than expected.

`profile_dataset()` prints and prepares basic data-quality summaries:

- row and column counts
- fraud and non-fraud counts
- fraud rate
- missing-value summary
- exact duplicate count
- target distribution
- fraud rate by transaction type
- numeric summary for selected numeric fields

These checks are descriptive. They help inspect the dataset, but they are not used to learn preprocessing parameters from the full dataset.

## Cleaning decisions

`apply_safe_cleaning()` performs two conservative cleaning steps:

- exact duplicate rows are removed
- rows with negative `amount` values are removed

Exact duplicates do not add new information. Negative transaction amounts are treated as invalid for this dataset, so they are removed before feature creation.

Other unusual numeric values are left in the data. In fraud detection, large or rare values may be meaningful, so the script does not remove outliers just because they look unusual.

`parse_event_timestamp()` converts the raw `timestamp` column into `event_timestamp` with `pd.to_datetime(..., errors="coerce")`. Rows with invalid timestamps are removed.

This is necessary because the timestamp is used later for time-based features and can also support time-aware splitting in the ML pipeline. If the timestamp cannot be parsed, the row is not reliable for those later steps.

The cleaning statistics are carried forward so they can be written to the metadata file and the data-quality report.

## Feature creation

`create_leakage_safe_features()` creates only deterministic features from values already present in the same row:

- `amount_log1p`
- `transaction_hour`
- `transaction_day_of_week`
- `transaction_month`
- `time_since_last_transaction_missing_flag`

`amount_log1p` is added because transaction amounts are often skewed. The log transform makes the scale less extreme while keeping the original `amount` column available for later cost-based evaluation.

The timestamp features capture simple timing patterns, such as hour of day and day of week. These are direct transformations of `event_timestamp`, so they do not use the target column or information from other rows.

Missing values in `time_since_last_transaction` are handled in a simple way. The script creates a missing-value flag and then fills the original field with `-1`. This keeps the information that the value was missing, while avoiding learned imputation before the train-test split.

## Final feature table

`select_final_columns()` keeps the columns that are handed to the next pipeline stage:

```text
transaction_id
event_timestamp
transaction_type
merchant_category
location
device_used
payment_channel
amount
amount_log1p
time_since_last_transaction
time_since_last_transaction_missing_flag
spending_deviation_score
velocity_score
geo_anomaly_score
transaction_hour
transaction_day_of_week
transaction_month
is_fraud
```

The following original columns are excluded from the processed feature table:

- `sender_account`
- `receiver_account`
- `ip_address`
- `device_hash`
- `fraud_type`

The account, IP, and device fields are direct identifiers. They are removed from the modelling table to avoid building a prototype that relies on raw identifiers instead of general transaction patterns.

`fraud_type` is excluded for a different reason. It describes the type of fraud after a transaction has already been labelled or investigated. That information would not be available as a normal input at prediction time, so keeping it would introduce label leakage.

`transaction_id` is kept because it is useful as an entity and lineage key. It allows rows in the processed table to be linked back to the original transaction without using the other identifier fields as model inputs.

## Saved outputs

`save_processed_datasets()` saves the feature table as Parquet when Parquet support is available:

```text
data/processed/fraud_features_v1.parquet
feature_repo/data/fraud_features_v1.parquet
```

If Parquet support is missing, the script saves CSV files instead:

```text
data/processed/fraud_features_v1.csv
feature_repo/data/fraud_features_v1.csv
```

The fallback keeps the pipeline usable on a basic Python setup. The script also prints a note that `pyarrow` can be installed later if Parquet output is preferred.

Three additional files are written:

```text
feature_repo/metadata/feature_schema_v1.json
feature_repo/metadata/feature_metadata_v1.json
reports/data_quality_report_v1.md
```

`feature_schema_v1.json` records the feature version, creation time, column names, data types, missing-value counts, target column, event timestamp column, and entity column.

`feature_metadata_v1.json` records the source file, row counts before and after cleaning, removed duplicate rows, removed negative amount rows, removed invalid timestamp rows, fraud count, fraud rate, excluded columns, and the main preprocessing notes.

`data_quality_report_v1.md` is a short human-readable report. It records the most important cleaning decisions and states which operations were intentionally postponed to the ML pipeline.

## What is left for the ML pipeline

The preprocessing script deliberately does not perform:

- train-validation-test splitting
- scaling or normalization
- one-hot encoding
- learned imputation
- SMOTE or other oversampling
- feature selection using the target
- model training
- threshold tuning

These steps depend on the training data distribution or on model behaviour. They should therefore be fitted only after the data is split. Keeping them out of this script helps make the later evaluation more trustworthy.

## Final verification

`verify_outputs()` prints the processed dataset shape, the total number of remaining missing values, the fraud rate, and whether each expected output file exists.

This is not a replacement for full testing, but it gives a clear end-of-run check that the preprocessing step completed and produced the expected handoff files.
