# Automated incremental data pipeline

The automated data pipeline uses one inbox and one local SQLite database. Its purpose is to accept newly arrived CSV batches, retain their raw records, create cleaned records, and document what happened. It stops after data preparation and never starts model training.

## Simple architecture

~~~text
new CSV in data/inbox
    -> wait until the file is stable
    -> validate the expected schema
    -> check whether the same file content was already ingested
    -> append original rows to raw_transactions
    -> clean the new batch with the existing preprocessing rules
    -> append valid rows to clean_transactions
    -> record the outcome in ingestion_batches
    -> remove the successfully ingested CSV
    -> stop
~~~

Only these persistent data locations are required:

~~~text
data/
|-- inbox/
`-- fraud_pipeline.db
~~~

The database is created automatically on the first scan. It is a generated local artifact and is not committed to Git.

## Database tables

| Table | Purpose |
| --- | --- |
| <code>raw_transactions</code> | Stores the original source columns together with a batch ID and source-row number. |
| <code>clean_transactions</code> | Stores the minimally cleaned columns used after the data-preparation boundary. |
| <code>ingestion_batches</code> | Stores the source filename, SHA-256 checksum, status, timestamps, row counts, errors, and retraining flag. |

Every successful CSV gets a unique <code>batch_id</code>. The pipeline processes only the rows in that incoming file and appends them to the existing database tables; it does not rerun preparation over all historical rows.

The clean table requires globally unique <code>transaction_id</code> values. If a later batch contains a transaction already stored in the database, the entire new batch is rejected rather than partially appended. This prevents accidental double-counting.

## Duplicate and failure behavior

Before ingestion, the pipeline calculates the CSV's SHA-256 checksum. If the same content was ingested successfully before, it records <code>skipped_duplicate</code>, does not append the rows again, and removes the duplicate input file.

If validation or database insertion fails, no partial rows from that batch are committed. The database records a <code>failed</code> outcome and the input remains in the inbox with a <code>.failed</code> suffix, for example <code>invalid.csv.failed</code>. Because it no longer ends in <code>.csv</code>, later scheduled scans ignore it until a person reviews or replaces it.

SQLite transactions make the raw and clean append atomic: both tables receive the batch, or neither table does.

## First-time run

Open PowerShell and run these commands one by one:

~~~powershell
Set-Location "D:\Germany\Documents\Magdeburg\Semester Documents\Sem 5\Thesis"

Copy-Item ".\Code snippets\sample_financial_fraud_detection_dataset.csv" `
  ".\Code snippets\data\inbox\sample_batch.csv"

.\masters_thesis\Scripts\python.exe `
  ".\Code snippets\automation\run_data_pipeline.py" `
  --stability-seconds 0
~~~

The last command should print a line beginning with <code>success:</code>. The sample CSV is then removed from the inbox because its raw contents and cleaned result are stored in the database.

Check the database status with:

~~~powershell
.\masters_thesis\Scripts\python.exe `
  ".\Code snippets\automation\run_data_pipeline.py" `
  --status
~~~

The status output shows the database path, recorded batch count, total raw rows, total clean rows, and recent outcomes. Each recent batch displays <code>ml_triggered=False</code>.

Run the integration tests with:

~~~powershell
.\masters_thesis\Scripts\python.exe -m unittest discover `
  -s ".\Code snippets\tests" -v
~~~

The zero-second stability override is for the controlled first demonstration. Normal unattended scans retain the safer 60-second window so a file still being copied is not ingested.

## Normal manual scan

For later batches, place a new CSV in <code>Code snippets/data/inbox</code>, wait at least 60 seconds, and run:

~~~powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\automation\run_data_pipeline.py"
~~~

If the inbox has no eligible CSV, the command prints <code>No stable, unprocessed CSV batches were found.</code>

## Optional Windows scheduling

The registration script creates a Windows Task Scheduler entry that performs the same scan every 15 minutes:

~~~powershell
& ".\Code snippets\automation\register_windows_task.ps1"
~~~

It accepts optional <code>-TaskName</code> and <code>-IntervalMinutes</code> parameters. Registration changes the operating-system schedule and therefore remains an explicit setup action.

Each scheduled execution checks the inbox once. With no stable CSV it immediately exits. With one or more new CSV files, it ingests each new batch and exits. The Windows task is configured to avoid overlapping instances.

## Separation from the ML pipeline

The ingestion code has no dependency on <code>04_baseline_model_comparison_with_mlflow.py</code>, <code>fraud_modeling_utils.py</code>, or MLflow. The <code>ingestion_batches</code> table enforces <code>ml_pipeline_triggered = 0</code> for every record.

The existing Feast <code>v1</code> artifacts remain frozen for reproducibility and are not overwritten by new arrivals. A future, separately controlled promotion or snapshot step can select database batches for a new ML experiment. That decision may consider labelled-data volume, drift, measured performance, cost, a periodic review, and human approval; data arrival alone is not a retraining decision.
