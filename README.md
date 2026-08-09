# MLOps Pipeline for Financial Fraud Detection

> Master's Thesis Project | Otto von Guericke University Magdeburg

## About the Project

This folder contains the current data-analysis and preparation work for my Master's thesis on building an end-to-end MLOps pipeline for financial fraud detection.

The project focuses on two things at the same time: detecting fraudulent transactions and showing how the machine-learning workflow can be made reproducible, traceable, and maintainable. Data preparation, the versioned local Feast handoff, and the full-data Feast-backed baseline model comparison are complete. The current milestone is to finalize and review the baseline results before starting advanced model development.

## Current Workflow

The workflow follows this process-model order:

1. Raw-data exploratory analysis - complete
2. Data cleaning and minimal preparation - complete
3. Versioned feature/label handoff and Feast registration - complete
4. Advanced post-cleaning EDA - complete
5. Controlled original-feature baseline model comparison with experiment tracking - complete
6. Advanced model development, including feature engineering, resampling, tuning, and additional boosting libraries - planned after the baseline milestone
7. Model acceptance and MLflow champion registration - planned
8. FastAPI serving and versioned prediction logging - planned
9. Batch monitoring, rule-based flags, and Streamlit visualization - planned

This order is intentional. The initial EDA happens before cleaning so that the cleaning decisions are based on evidence from the raw data. The baseline stage compares fixed non-skill, linear, and nonlinear configurations using only the original predictors. Feature engineering, resampling, systematic tuning, and additional boosting libraries are reserved for separate tracked experiments in the advanced stage.

## Completed Work

- Created an initial raw-data EDA notebook using the untouched source CSV.
- Confirmed that the raw dataset contains 5,000,000 rows and 18 columns.
- Confirmed that all timestamps are valid when parsed with an ISO-8601-aware parser.
- Identified `fraud_type` as target-derived leakage and excluded it from model inputs.
- Identified strong class imbalance: 179,553 fraud cases, about 3.59% of the data.
- Found that `time_since_last_transaction` missingness occurs only among non-fraud rows.
- Updated the preprocessing pipeline to create a minimally cleaned gold table.
- Added versioned Parquet feature and label handoffs, explicit feature definitions, schema/lineage metadata, and a local Feast registry.
- Preserved missing values for later train-only imputation instead of filling them globally.
- Avoided pre-baseline engineered features such as log amount, temporal fields, and missingness flags.
- Created an advanced post-cleaning EDA notebook with inline charts only.
- Updated the Markdown reports so they match the current notebooks and preprocessing logic.
- Added one leakage-aware MLflow baseline comparison covering Dummy, Logistic SGD, Random Forest, and Histogram Gradient Boosting on the original cleaned feature set.
- Kept Decision Tree and sampled k-NN as optional sensitivity or feasibility checks.
- Added dataset hashing, complete scalar parameter logging, runtime metrics, evaluation figures, model explanations, environment metadata, source snapshots, and model signatures.
- Generated and registered the full five-million-row Feast feature handoff.
- Validated the feature/label handoff with a successful Feast round-trip integration test.
- Completed the full-data Feast-backed baseline comparison and selected Random Forest using validation PR-AUC.
- Frozen the canonical original-feature protocol, lineage, run IDs, metrics, and limitations in `Markdown files/original_feature_benchmark_v1.md`.
- Created the review-ready baseline results document with full-data metrics, interpretation, DSPM alignment, and MLflow image placeholders in `Markdown files/baseline_model_development_results_v1.md`.
- Updated the three-page UML architecture file with component, activity, and local deployment views for the final target prototype.

## Current Data Artifacts

Raw input:

```text
financial_fraud_detection_dataset.csv
```

Cleaned output:

```text
gold_financial_fraud_detection_table.csv
```

Canonical feature-store handoff:

```text
feature_repo/data/fraud_features_v1.parquet
feature_repo/data/fraud_labels_v1.parquet
feature_repo/metadata/feature_schema_v1.json
feature_repo/metadata/feature_metadata_v1.json
```

The current gold table contains:

- 5,000,000 transactions
- 13 columns
- 179,553 fraudulent transactions
- fraud rate: approximately 3.59%
- no engineered model features
- missing `time_since_last_transaction` values preserved

The gold table keeps only the original usable predictors, `transaction_id` for lineage, `event_timestamp` for time-aware splitting and analysis, and `is_fraud` as the target.

The Feast handoff separates predictors from labels. Feast registers only the ten original predictors; `is_fraud` remains in the label table and is carried into historical retrieval as the supervised-learning target. Imputation, scaling, encoding, class-weight-based imbalance handling, and threshold tuning remain inside the ML pipeline.

## Repository Structure

```text
Code snippets/
|-- .gitignore                                  # Excludes generated data, tracking, and cache artifacts
|-- 01_initial_raw_data_eda.ipynb              # Initial EDA on the untouched raw CSV
|-- initial_raw_data_eda_report_v1.md          # Detailed raw EDA findings
|-- 02_data_pipeline_preprocessing.py          # Cleaning and minimal preparation pipeline
|-- data_quality_report_v1.md                  # Generated quality report from preprocessing
|-- feature_store.yaml                         # Local Feast project configuration
|-- fraud_feature_definitions.py               # Feast entity, feature view, and feature service
|-- fraud_feature_store.py                     # Historical retrieval, validation, and lineage
|-- feature_repo/
|   |-- data/                                  # Generated Parquet handoff and Feast registry database
|   `-- metadata/                              # Feature definitions, schema, and lineage metadata
|-- 03_exploratory_data_analysis.ipynb         # Advanced EDA on the cleaned gold table
|-- eda_report_v1.md                           # Detailed advanced EDA findings
|-- 04_baseline_model_comparison_with_mlflow.py  # Fixed original-feature baseline comparison with MLflow
|-- fraud_modeling_utils.py                    # Shared splitting, evaluation, plotting, and tracking logic
|-- Markdown files/                            # Documentation for pipeline, EDA, Feast, and modeling code
|-- tests/
|   `-- test_feature_store_handoff.py          # Feast handoff round-trip integration test
|-- Architecture_Diagram.drawio                # Component, activity, and deployment diagrams
|-- sample_financial_fraud_detection_dataset.csv
|-- README.md
```

Large raw and processed CSV files are local working artifacts and should not be committed to version control.

## Why There Are Two EDA Stages

The first EDA notebook looks at the raw dataset before any cleaning. Its purpose is to understand the data, spot quality issues, identify leakage risks, and decide what the preparation pipeline should do.

The advanced EDA notebook runs after the minimally cleaned gold table is created. Its purpose is different: it checks whether the cleaned artifact makes sense and studies deeper relationships before modeling. The advanced EDA creates temporary analysis variables inside the notebook, but it does not save those variables as model features.

This keeps the workflow clean: EDA can suggest feature ideas, but the value of those ideas must be tested later through tracked modeling experiments.

## Key Findings So Far

- The dataset is highly imbalanced, so accuracy alone will not be a useful evaluation metric.
- `fraud_type` is not a valid model input because it describes the known fraud outcome.
- Transaction amount is right-skewed, so `amount_log1p` is a candidate feature experiment after the baseline.
- Timestamp patterns may be useful, so hour, weekday, and month features should be tested later.
- Missingness in `time_since_last_transaction` is suspiciously related to the target and should be handled carefully as a separate sensitivity experiment.
- Numeric variables have weak direct linear correlation with the target; the baseline comparison therefore includes fixed nonlinear configurations alongside the linear reference.

## Running the Current Work

From the thesis workspace, run the preprocessing script with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\02_data_pipeline_preprocessing.py"
```

This writes the gold CSV and the versioned Parquet feature/label handoff. Register the definitions after the handoff files exist:

```powershell
Push-Location ".\Code snippets"
& "..\masters_thesis\Scripts\feast.exe" apply
Pop-Location
```

Then open the notebooks in order:

```text
01_initial_raw_data_eda.ipynb
03_exploratory_data_analysis.ipynb
```

The EDA notebooks display tables and charts inline. They do not save separate PNG, SVG, or chart-output files.

Run a quick MLflow smoke baseline comparison with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\04_baseline_model_comparison_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

Run the full original-feature baseline comparison with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\04_baseline_model_comparison_with_mlflow.py"
```

Run an optional Decision Tree or sampled k-NN check only if you need a sensitivity analysis:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\04_baseline_model_comparison_with_mlflow.py" --models decision_tree_depth_10 knn_neighbors_31 --sample-rows 50000 --skip-data-hash
```

The baseline comparison uses Feast by default. The old direct-CSV path remains available only as an explicit fallback by adding `--data-source csv`.

Run the Feast handoff integration test with:

```powershell
.\masters_thesis\Scripts\python.exe -m unittest discover -s ".\Code snippets\tests" -v
```

Open the local MLflow interface on Windows with one worker:

```powershell
.\masters_thesis\Scripts\mlflow.exe ui --workers 1 --backend-store-uri "sqlite:///D:/Germany/Documents/Magdeburg/Semester Documents/Sem 5/Thesis/Code snippets/mlflow_tracking.db"
```

## Next Steps

- Add the specified MLflow images to the baseline comparison document and complete its final human review.
- Begin controlled feature experimentation after the baseline milestone has been reviewed.
- Define model-acceptance criteria and transaction-value cost assumptions.
- Register the approved preprocessing/model pipeline and threshold in MLflow with a champion alias.
- Implement FastAPI prediction serving and persist versioned prediction logs.
- Build batch data-quality, drift, prediction, performance, and expected-cost monitoring.
- Add rule-based investigation/retraining recommendations without fully automated retraining.
- Build the Streamlit monitoring dashboard.
- Package the MLflow, FastAPI, and Streamlit services for the planned local Docker deployment.
- Add serving, monitoring, cost, registration, and end-to-end integration tests.

## Status

Data understanding, minimal preparation, the full Feast handoff, Feast historical retrieval, direct-CSV experiments, and the canonical full-data Feast-backed baseline comparison are complete. The current task is to finish the baseline results document and create a clean Git checkpoint. Random Forest is the strongest fixed original-feature baseline configuration, but it has not been formally accepted or registered as a champion model. Advanced feature engineering, resampling, systematic tuning, additional boosting libraries, serving, prediction logging, monitoring, the Streamlit dashboard, and Docker deployment remain outside the current milestone.

The three UML diagrams in `Architecture_Diagram.drawio` describe the final target thesis prototype. They include both implemented components and the remaining serving and monitoring components listed above.

## Author

**Ishmita Basu**

Master's Thesis  
Otto von Guericke University Magdeburg
