# How the local Feast handoff connects data preparation to modeling

The local Feast repository provides a clear boundary between the preprocessing script and the model experiments. Its role is not to clean data or train models. Instead, it defines which fields constitute a versioned feature set and reconstructs the training table from features and labels at the correct event time.

The automated batch trigger strengthens this boundary. It appends new batches to separate raw and clean SQLite tables, but it neither updates the canonical Feast registry nor invokes training. Selecting database batches for a future model snapshot remains a separate decision.

The workflow is:

~~~text
raw transaction CSV
    → validated and minimally cleaned data
    → separate versioned feature and label files
    → Feast historical retrieval
    → chronological split
    → train-fitted preprocessing and model training
    → MLflow tracking
~~~

## Artifacts created during preprocessing

Running <code>02_data_pipeline_preprocessing.py</code> writes four files used by the feature-store layer:

~~~text
feature_repo/data/fraud_features_v1.parquet
feature_repo/data/fraud_labels_v1.parquet
feature_repo/metadata/feature_schema_v1.json
feature_repo/metadata/feature_metadata_v1.json
~~~

The feature table contains the transaction key, event timestamp, and ten original prediction-time variables. The label table contains the same key and timestamp together with <code>is_fraud</code>. This separation is deliberate: Feast registers the predictors, while the target remains outside the feature view.

The schema file describes the columns and their Feast-compatible types. The metadata file records the feature version, row and fraud counts, cleaning statistics, artifact hashes, preprocessing-source hash, and known prototype limitations. These records turn the handoff into an explicit data contract rather than an undocumented pair of files.

Automated arrivals are recorded in <code>data/fraud_pipeline.db</code> rather than written into the Feast repository. The existing <code>v1</code> files and registry therefore remain stable until an explicit snapshot and promotion step is introduced. This preserves the reproducibility of the completed baseline while allowing the data pipeline to continue ingesting new records.

## What the Feast definitions register

The file <code>fraud_feature_definitions.py</code> declares the objects that Feast adds to its registry.

The <code>transaction</code> entity uses <code>transaction_id</code> as its join key. For this static research dataset, each row represents a unique transaction, so the key is suitable for reconstructing the offline training table. It is not intended as a claim that a production platform should use transactions instead of reusable customer or account entities.

A file source points to <code>fraud_features_v1.parquet</code> and identifies <code>event_timestamp</code> as event time. The <code>fraud_transaction_features</code> feature view then declares the ten predictor fields and their types. Finally, <code>fraud_model_features_v1</code> groups those fields into the versioned feature service requested by the modeling scripts.

This repository is an offline prototype. It does not configure an online store or materialize features for real-time serving.

## How the training table is reconstructed

The helper in <code>fraud_feature_store.py</code> first checks that the feature table, label table, metadata, schema, registry, definition file, and Feast configuration all exist. It compares the code’s expected version and feature service with the metadata and, unless hashing is disabled for a smoke run, recomputes the artifact hashes.

The label Parquet file is then loaded with three columns: <code>transaction_id</code>, <code>event_timestamp</code>, and <code>is_fraud</code>. These rows are passed to Feast as the entity dataframe. Feast retrieves the registered features for the corresponding transaction and event time, after which the helper verifies the returned schema, row count, unique identifiers, missing labels, and total number of fraud cases.

The final result has a predictable order:

~~~text
transaction_id
event_timestamp
ten model features
is_fraud
~~~

The model scripts receive this validated table together with dataset metadata and lineage values that can be logged to MLflow.

## Leakage boundary

Feast stores record-level values that are available at prediction time. It does not store learned preprocessing results. Median imputation, standardization, categorical encoding, class weighting or resampling, target-based feature selection, and threshold selection all remain inside the model experiment after the chronological split.

This division avoids a common feature-store mistake: moving transformations fitted on the full dataset ahead of model evaluation merely because a feature platform has been introduced.

## Running the handoff

From the thesis workspace, generate the artifacts and update the Feast registry:

~~~powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\02_data_pipeline_preprocessing.py"

Push-Location ".\Code snippets"
& "..\masters_thesis\Scripts\feast.exe" apply
Pop-Location
~~~

The baseline comparison script can then use Feast without additional options. Passing <code>--data-source csv</code> deliberately bypasses this interface for historical reproducibility. Passing <code>--skip-data-hash</code> avoids recomputing large-file hashes during a quick smoke test, but it does not remove the hashes already stored in the handoff metadata.

To process newly arrived batches independently, place them in <code>data/inbox</code> and schedule <code>automation/run_data_pipeline.py</code>. The <code>ingestion_batches</code> database table records whether the data stage succeeded and enforces that the ML pipeline was not triggered. See <code>automated_data_pipeline.md</code> for ingestion, duplicate, failure, status, and scheduling details.
