# 01 - Data pipeline preprocessing

This note documents `01_data_pipeline_preprocessing.py`.

The script prepares the public financial fraud dataset for the later machine learning pipeline. It does not train a model and it does not make decisions that should depend on a train-test split. Its job is more specific: check that the dataset has the expected structure, apply basic cleaning, create simple row-level features, and save two easy-to-find files in the active base directory selected by `build_paths()`.

The main reason for keeping this preprocessing step separate is leakage control. Anything that learns from the distribution of the full dataset, such as scaling, learned imputation, one-hot encoding, SMOTE, feature selection, or threshold tuning, is left for the ML pipeline after the data has been split.

## Input file

The script expects the raw CSV file:

```text
financial_fraud_detection_dataset.csv
```

By default, it looks for this file in the hard-coded `DEFAULT_BASE_DIR`, which currently points to the `Code snippets/` folder in this thesis workspace. If the file is not found there, `build_paths()` falls back to the current working directory. This makes the script easier to run both from the configured project folder and from a folder that directly contains the CSV.

Run the script with:

```bash
python "Code snippets/01_data_pipeline_preprocessing.py"
```

The script writes its outputs into whichever base directory `build_paths()` is using. In the normal project setup, that is `Code snippets/`. If the configured CSV is not found and the current working directory is used instead, the outputs are written there. The script does not create a separate raw folder, processed folder, feature repository folder, metadata folder, or reports folder.

## Pipeline structure

The pipeline is coordinated by `main()`. The flow is:

```python
raw_file, output_file, report_file = build_paths()
read_sample(raw_file)

df = load_dataset(raw_file)
df = standardise_columns(df)
validate_schema(df)
quality_profile = profile_dataset(df)

df_clean, cleaning_stats = apply_safe_cleaning(df)
df_clean, cleaning_stats = parse_event_timestamp(df_clean, cleaning_stats)

df_features = create_leakage_safe_features(df_clean)
df_features = select_final_columns(df_features)

quality_report_file = save_quality_report(
    df_features,
    raw_file,
    report_file,
    cleaning_stats,
    quality_profile,
)
gold_table_file = save_gold_table(df_features, output_file)
verify_outputs(df_features, [gold_table_file, quality_report_file])
```

The functions are kept small on purpose. Each one has a clear role, which makes the script easier to inspect, rerun, and adapt if the dataset changes.

## Path handling

`build_paths()` defines three paths:

```text
financial_fraud_detection_dataset.csv
gold_financial_fraud_detection_table.csv
data_quality_report_v1.md
```

All three live in the active base directory. In the normal setup this is the hard-coded `Code snippets/` path. If the CSV is not found there, the active base directory becomes the current working directory.

The script does not dynamically infer the folder that contains the Python file. It uses `DEFAULT_BASE_DIR` first, then `Path.cwd()` as a fallback.

No raw copy is created. The original input CSV is read directly and is not changed.

## Raw data loading

`read_sample()` reads the first five rows and prints the column names. This is a quick sanity check before loading the full file. `load_dataset()` then reads the full CSV with the data types defined in `DTYPE_MAP`.

Fixed data types are used because columns such as IDs, account numbers, IP addresses, and categorical fields should not be accidentally converted into numeric values. The target column is first read as boolean and then converted to `0` and `1` by `standardise_columns()`.

## Schema and profiling checks

`standardise_columns()` first converts `is_fraud` from boolean values to integer `0` and `1` values. After that, `validate_schema()` checks that the input data has exactly the expected columns. The script stops if a column is missing or if an unexpected column appears.

Because `standardise_columns()` runs before `validate_schema()`, the `is_fraud` column must already exist for the target conversion to succeed. If `is_fraud` is missing, the script fails during this target-conversion step before the formal schema validation message is reached.

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

The cleaning statistics are carried forward so they can be written to `data_quality_report_v1.md`.

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

## Final table columns

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

The following original columns are excluded from the final table:

- `sender_account`
- `receiver_account`
- `ip_address`
- `device_hash`
- `fraud_type`

The account, IP, and device fields are direct identifiers. They are removed from the modelling table to avoid building a prototype that relies on raw identifiers instead of general transaction patterns.

`fraud_type` is excluded for a different reason. It describes the type of fraud after a transaction has already been labelled or investigated. That information would not be available as a normal input at prediction time, so keeping it would introduce label leakage.

`transaction_id` is kept because it is useful as an entity and lineage key. It allows rows in the final table to be linked back to the original transaction without using the other identifier fields as model inputs.

## Gold table description

The main output of the preprocessing script is:

```text
gold_financial_fraud_detection_table.csv
```

This file is the **gold table** for the next stages of the thesis pipeline. In this project, "gold table" means the cleaned and documented handoff dataset that later notebooks can use for EDA and model development. It is not yet a fully model-ready matrix, because split-dependent transformations are deliberately postponed.

The current gold table contains:

- 4,277,262 transaction rows
- 18 columns
- 147,881 fraud cases
- a fraud rate of approximately 3.4574%
- no remaining missing values after the preprocessing decisions in this script

The table combines several kinds of fields:

- **Lineage field:** `transaction_id`, which keeps each row traceable to the original transaction.
- **Time field:** `event_timestamp`, which supports temporal inspection and possible time-aware splitting.
- **Descriptive transaction attributes:** `transaction_type`, `merchant_category`, `location`, `device_used`, `payment_channel`, and `amount`.
- **Leakage-safe derived features:** `amount_log1p`, `transaction_hour`, `transaction_day_of_week`, `transaction_month`, and `time_since_last_transaction_missing_flag`.
- **Existing behavioral/anomaly scores:** `time_since_last_transaction`, `spending_deviation_score`, `velocity_score`, and `geo_anomaly_score`.
- **Target field:** `is_fraud`, encoded as `0` for non-fraud and `1` for fraud.

The gold table is suitable for the EDA notebook because it is clean enough to inspect directly and still preserves the important transaction context. It is also suitable as the input to the later baseline-model notebook, provided that train-validation-test splitting is performed before any learned preprocessing.

The gold table is intentionally not over-processed. Categorical variables are not one-hot encoded, numeric variables are not scaled, class imbalance is not corrected, and feature selection is not performed. This protects the later evaluation from data leakage because those operations must be fitted only on training data.

## Saved outputs

The current version saves only these files in the active base directory, normally `Code snippets/`:

```text
gold_financial_fraud_detection_table.csv
data_quality_report_v1.md
```

`gold_financial_fraud_detection_table.csv` is the final preprocessed table for the next ML pipeline step.

`data_quality_report_v1.md` is the human-readable data quality report. It records the most important profiling results, cleaning decisions, and leakage-control notes.

If an older file still exists at `Code snippets/reports/data_quality_report_v1.md`, that is a leftover from the previous version of the pipeline. The current script does not create or update that folder.

## Data quality report

`save_quality_report()` writes `data_quality_report_v1.md` to the `report_file` path returned by `build_paths()`. In the normal project setup, that path is directly inside `Code snippets/`.

The report includes:

- source file path
- rows before and after cleaning
- number of columns after preprocessing
- fraud count and fraud rate
- missing values before cleaning
- target distribution before cleaning
- cleaning decisions
- transaction type summary
- numeric summary
- leakage-control decision

The report is saved before the large gold CSV is replaced. This means the quality report can still be created even if Windows blocks overwriting the existing gold CSV because the file is open somewhere.

## Gold table save behavior

`save_gold_table()` writes the full preprocessed table to a temporary file first:

```text
gold_financial_fraud_detection_table_temporary.csv
```

After the temporary file is written successfully, the script replaces:

```text
gold_financial_fraud_detection_table.csv
```

This avoids leaving a half-written final table if the CSV write fails partway through.

If Windows blocks the replacement, close the gold CSV if it is open in Excel, VS Code, or another program, then rerun the script. The completed temporary table can be used to confirm that the preprocessing itself finished.

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

## Script execution guard

The file can be run as a normal Python script through the `if __name__ == "__main__":` block.

Because the file is also organized with notebook-style cells, the final cell includes `REQUIRED_PIPELINE_NAMES`. Before calling `main()`, it checks that all earlier function definitions are available in `globals()`. If the final cell is run before the earlier cells in an IDE, the script prints a message listing the missing definitions instead of raising a confusing `NameError`.
