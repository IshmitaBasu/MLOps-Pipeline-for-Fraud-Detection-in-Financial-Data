# 02 – Preparing the transaction data for analysis and modeling

The preprocessing script, <code>02_data_pipeline_preprocessing.py</code>, turns the observations from the initial EDA into a reproducible dataset. Its responsibility is deliberately narrow: it validates the raw file, applies cleaning rules that are safe before a train–test split, and creates the data contracts used by the later notebooks and model scripts.

It does not impute missing values, encode categories, scale numerical fields, resample the target, or engineer model features. Those operations learn something from the data and therefore belong inside the modeling pipeline, where they can be fitted on training rows only.

The script is also a parameterized command-line entry point and exposes a reusable <code>prepare_dataframes</code> function. Its default invocation preserves the canonical <code>v1</code> Feast workflow. The independent automation layer reuses the same validation and cleaning functions when it appends new batches to the local data-pipeline database, so manual and automated preparation follow one set of rules.

## From the raw file to the cleaned table

The script begins by locating <code>financial_fraud_detection_dataset.csv</code>. It reads a five-row preview first so that the input can be inspected quickly, then loads the complete file using an explicit type map. Column names are standardized and checked against the expected schema. If a required field is missing, the script stops rather than producing an incomplete downstream artifact.

Before changing anything, the script profiles the source. It records missing-value counts, duplicate rows, duplicate transaction identifiers, target balance, timestamp validity, and numeric ranges. The cleaning functions then remove only conditions that are unambiguously invalid: exact duplicate rows, negative transaction amounts, and timestamps that cannot be parsed as ISO-8601 values. The current source contains none of these problems, so a normal run preserves all 5,000,000 transactions.

The final cleaned table contains the ten original candidate predictors together with the transaction identifier, event time, and fraud label:

~~~text
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
~~~

The identifier remains available for lineage and joins, while the timestamp supports chronological evaluation. Neither is passed to the baseline model as a predictor.

## How the EDA decisions are applied

The following table shows how the exploratory findings become preprocessing rules.

| Evidence from the raw-data EDA | Decision in the script |
| --- | --- |
| Timestamps use valid ISO-8601 strings with mixed fractional-second precision. | Parse them into a single <code>event_timestamp</code> column and reject only genuinely invalid values. |
| No duplicate rows or duplicate transaction IDs were found. | Keep validation checks so that a changed source cannot introduce duplicates unnoticed. |
| No negative transaction amounts were found. | Validate the domain rule while preserving all current rows. |
| <code>fraud_type</code> reveals whether the transaction is fraudulent. | Remove it because it would give the model information unavailable at prediction time. |
| Account, IP, and device identifiers have very high cardinality. | Exclude <code>sender_account</code>, <code>receiver_account</code>, <code>ip_address</code>, and <code>device_hash</code> from the baseline modeling table. |
| <code>time_since_last_transaction</code> is missing for 17.93% of rows, only among non-fraud cases. | Preserve the missing values so that imputation and a possible missingness indicator can be evaluated safely after splitting. |
| Transaction amount is strongly right-skewed. | Keep the original amount for the benchmark and test <code>amount_log1p</code> later as a named experiment. |
| Timestamps contain potentially useful temporal information. | Preserve event time for splitting; test hour, weekday, and month features separately later. |

This conservative design gives the later experiments a clear reference point. If transformations were introduced here, it would be difficult to tell whether a performance change came from a new model or from a changed feature set.

## Why the feature and label handoffs are separate

Alongside the cleaned CSV, the script creates two Parquet files for Feast. The feature file contains <code>transaction_id</code>, <code>event_timestamp</code>, and the ten original predictors. The label file contains the same join fields plus <code>is_fraud</code>.

The shared identifier tells Feast which transaction a label belongs to, and the shared timestamp supports point-in-time historical retrieval. Keeping <code>is_fraud</code> in a separate file prevents the target from being registered as a feature and makes the leakage boundary explicit.

The handoff columns therefore follow a simple rule:

~~~text
feature handoff = entity key + event time + prediction-time predictors
label handoff   = entity key + event time + supervised target
~~~

The script also writes a JSON schema and metadata record. These files describe the entity and feature service, preserve row and fraud counts, record the cleaning decisions, and store SHA-256 hashes for the source code and generated artifacts. Later modeling runs validate this information before training so that the experiment cannot silently use a different handoff.

## Files written by the script

A successful run creates:

~~~text
gold_financial_fraud_detection_table.csv
data_quality_report_v1.md
feature_repo/data/fraud_features_v1.parquet
feature_repo/data/fraud_labels_v1.parquet
feature_repo/metadata/feature_schema_v1.json
feature_repo/metadata/feature_metadata_v1.json
~~~

The CSV, Parquet, and JSON outputs are written atomically: each result is completed in a temporary file before it replaces the final path. This reduces the risk of leaving a partially written artifact after an interrupted run. The final verification step checks that every expected file exists and that the saved row counts agree with the cleaned table.

The parameterized command can still create a separate named artifact set when an explicit standalone snapshot is needed. Automated arrivals do not create Feast files, however: they append to <code>data/fraud_pipeline.db</code>. Consequently, an arriving batch cannot silently replace the artifacts used by the existing baseline model.

## Independent automated trigger

The companion script <code>automation/run_data_pipeline.py</code> scans <code>data/inbox</code> for stable CSV files. It calculates a content hash, skips previously successful content, stores the original columns in <code>raw_transactions</code>, applies this script's cleaning functions to the new batch, and stores the result in <code>clean_transactions</code>. The <code>ingestion_batches</code> table replaces separate manifest and lifecycle folders.

Successful and duplicate CSV inputs are removed after the database records them. Invalid inputs stay in the inbox with a <code>.failed</code> suffix. The automation does not call the model-comparison script, MLflow training functions, model registration, or deployment. This makes data arrival an ingestion event rather than an automatic retraining event. Full operational and first-run instructions are documented in <code>automated_data_pipeline.md</code>.

Run one inbox scan with:

~~~powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\automation\run_data_pipeline.py"
~~~

For the complete first-time sequence—including copying the sample batch, using the immediate stability override, checking outputs, running tests, and optionally installing the scheduled task—see <code>automated_data_pipeline.md</code>.

## Registering the Feast definitions

After preprocessing, the version-controlled Feast definitions can be applied from the <code>Code snippets</code> directory:

~~~powershell
Push-Location ".\Code snippets"
& "..\masters_thesis\Scripts\feast.exe" apply
Pop-Location
~~~

The definitions register the transaction entity, the <code>fraud_transaction_features</code> feature view, and the versioned <code>fraud_model_features_v1</code> feature service. This is a local offline-store prototype rather than a production online feature platform. The large Parquet files and generated registry remain local artifacts, while the schema, metadata, and Python definitions provide the reproducible contract.

