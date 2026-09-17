# MLOps Pipeline for Financial Fraud Detection

> Master's Thesis Project | Otto von Guericke University Magdeburg

## About the Project

This folder contains the current data-analysis and preparation work for my Master's thesis on building an end-to-end MLOps pipeline for financial fraud detection.

The project focuses on two things at the same time: detecting fraudulent transactions and showing how the machine-learning workflow can be made reproducible, traceable, and maintainable. Data preparation, the versioned local Feast handoff, automated batch-data triggering, the full-data Feast-backed baseline comparison, controlled feature engineering, and validation-only model optimisation are complete. The current milestone is to review the selected research candidate and operating-threshold assumptions before final evaluation, registration, serving, and monitoring.

## Current Workflow

The workflow follows this process-model order:

1. Raw-data exploratory analysis - complete
2. Data cleaning and minimal preparation - complete
3. Versioned feature/label handoff and Feast registration - complete
4. Independently triggered incremental batch ingestion into raw and clean database tables - complete
5. Advanced post-cleaning EDA - complete
6. Controlled original-feature baseline model comparison with experiment tracking - complete
7. Controlled feature-engineering comparison with a fixed Random Forest - complete
8. Advanced model development, including resampling, tuning, and an additional boosting library - complete
9. Candidate review, final evaluation, and MLflow model registration - pending supervisor guidance
10. FastAPI serving and versioned prediction logging - planned
11. Batch monitoring, rule-based flags, and Streamlit visualization - planned

This order is intentional. The initial EDA happens before cleaning so that the cleaning decisions are based on evidence from the raw data. The baseline stage compares fixed non-skill, linear, and nonlinear configurations using only the original predictors. The feature experiment then keeps the selected Random Forest fixed and changes only the input features. Model-family screening, imbalance handling, and tuning were subsequently kept as separate controlled experiments.

## Completed Work

- Created an initial raw-data EDA notebook using the untouched source CSV.
- Confirmed that the raw dataset contains 5,000,000 rows and 18 columns.
- Confirmed that all timestamps are valid when parsed with an ISO-8601-aware parser.
- Identified `fraud_type` as target-derived leakage and excluded it from model inputs.
- Identified strong class imbalance: 179,553 fraud cases, about 3.59% of the data.
- Found that `time_since_last_transaction` missingness occurs only among non-fraud rows.
- Updated the preprocessing pipeline to create a minimally cleaned gold table.
- Added versioned Parquet feature and label handoffs, explicit feature definitions, schema/lineage metadata, and a local Feast registry.
- Added an independently schedulable inbox trigger that appends stable CSV batches to local raw and clean SQLite tables without starting the ML pipeline.
- Added checksum-based duplicate prevention and a database ingestion log containing success, duplicate, and failure outcomes.
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
- Completed the full-data controlled feature comparison while keeping the Random Forest fixed.
- Found that `temporal_v2` improved validation Average Precision by 0.87% but performed 0.38% worse than the original benchmark on the test period, so it was not accepted as a canonical replacement for `original_v1`.
- Restored the feature experiment, added deterministic calculation tests, and separated its plan, implementation guide, and completed results into clearly named documents.
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
|-- automation/
|   |-- run_data_pipeline.py                   # One-shot inbox scanner for scheduled execution
|   `-- register_windows_task.ps1              # Optional local Task Scheduler registration
|-- data/
|   |-- inbox/                                 # New CSV batches arrive here
|   `-- fraud_pipeline.db                      # Generated raw, clean, and ingestion-log tables
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
|-- 05_feature_engineering_with_mlflow.py      # Controlled feature-group comparison with fixed Random Forest
|-- fraud_modeling_utils.py                    # Shared splitting, evaluation, plotting, and tracking logic
|-- model_optimization_utils.py                # Model, imbalance, threshold, and cost helpers
|-- 06_model_optimization_and_operational_evaluation_with_mlflow.py  # Validation-only optimisation runner
|-- requirements.txt                          # Pinned Python dependencies, including LightGBM
|-- Markdown files/
|   |-- 05_feature_engineering_experiment_plan.md       # Pre-run question, controls, and decision rules
|   |-- 05_feature_engineering_implementation_guide.md  # Script behavior and execution instructions
|   |-- 05_feature_engineering_results.md               # Full-data findings and current feature decision
|   |-- 06_model_optimization_and_operational_evaluation_plan.md  # Models, tuning search, and acceptance plan
|   |-- 06_model_optimization_and_operational_evaluation_implementation_guide.md  # Modes and commands
|   `-- 06_model_optimization_and_operational_evaluation_results.md  # Commands, results, and interpretation
|-- tests/
|   |-- test_feature_store_handoff.py          # Feast handoff round-trip integration test
|   |-- test_automated_data_pipeline.py        # Trigger, duplicate, failure, and separation tests
|   |-- test_feature_engineering.py            # Feature-calculation and validation safeguards
|   `-- test_model_optimization_and_operational_evaluation.py  # Tuning and test-isolation safeguards
|-- Architecture_Diagram.drawio                # Component, activity, and deployment diagrams
|-- sample_financial_fraud_detection_dataset.csv
|-- README.md
```

## Documentation Map

| Question | Document to read |
| --- | --- |
| What was planned for the feature experiment, and why? | `Markdown files/05_feature_engineering_experiment_plan.md` |
| How does the feature script work, and how do I run it? | `Markdown files/05_feature_engineering_implementation_guide.md` |
| What values were obtained, and which features should be kept? | `Markdown files/05_feature_engineering_results.md` |
| What is the current plan for model optimisation, imbalance handling, thresholds, and acceptance? | `Markdown files/06_model_optimization_and_operational_evaluation_plan.md` |
| How do the stage-06 screening and tuning modes work, and what command should be run? | `Markdown files/06_model_optimization_and_operational_evaluation_implementation_guide.md` |
| What has been run in stage 06, what values were obtained, and what do they mean? | `Markdown files/06_model_optimization_and_operational_evaluation_results.md` |
| What is the frozen original-feature benchmark? | `Markdown files/original_feature_benchmark_v1.md` |
| What are the detailed baseline model results? | `Markdown files/baseline_model_development_results_v1.md` |
| How do preprocessing, Feast, and automated ingestion work? | `Markdown files/02_data_pipeline_preprocessing.md`, `feature_store_handoff.md`, and `automated_data_pipeline.md` |

Large raw and processed CSV files are local working artifacts and should not be committed to version control.

## Why There Are Two EDA Stages

The first EDA notebook looks at the raw dataset before any cleaning. Its purpose is to understand the data, spot quality issues, identify leakage risks, and decide what the preparation pipeline should do.

The advanced EDA notebook runs after the minimally cleaned gold table is created. Its purpose is different: it checks whether the cleaned artifact makes sense and studies deeper relationships before modeling. The advanced EDA creates temporary analysis variables inside the notebook, but it does not save those variables as model features.

This keeps the workflow clean: EDA can suggest feature ideas, but the value of those ideas must be tested later through tracked modeling experiments.

## Key Findings So Far

- The dataset is highly imbalanced, so accuracy alone will not be a useful evaluation metric.
- `fraud_type` is not a valid model input because it describes the known fraud outcome.
- Adding `amount_log1p` reduced validation Average Precision by 0.20% relative to the original-feature control.
- Temporal features produced the best validation result, improving Average Precision by 0.87%, but their test Average Precision was 0.38% below the original benchmark.
- The missingness indicator changed validation Average Precision by only 0.05% and remains a dataset-bias sensitivity result.
- No engineered group provided convincing test evidence for replacing `original_v1`.
- Numeric variables have weak direct linear correlation with the target; the baseline comparison therefore includes fixed nonlinear configurations alongside the linear reference.
- LightGBM led the 50,000-row smoke comparison, but that advantage disappeared on the full validation period.
- Random Forest remains the full-data metric winner at 0.0439598 validation Average Precision. Histogram Gradient Boosting is only 0.27% lower and trained about 15.7 times faster.
- At a 10% validation alert capacity, Random Forest captured about 12.30% of fraud cases. Its maximum-F1 threshold still flagged about 82% of transactions.
- The full-data imbalance comparison selected 5:1 training-only undersampling at 0.0442268 validation Average Precision, a 0.61% improvement over class weighting. The improvement remains below the provisional 5% target.
- The nine-configuration tuning smoke test completed successfully. `rf_combined_flexible` had the highest smoke Average Precision, but no configuration was selected from smoke data.

## Running the Current Work

From the thesis workspace, run the preprocessing script with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\02_data_pipeline_preprocessing.py"
```

This unchanged command writes the canonical `v1` gold CSV and Parquet feature/label handoff used by the current ML pipeline. Register the definitions after the handoff files exist:

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

Run a quick technical smoke check of the feature experiment with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\05_feature_engineering_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

Reproduce the full five-million-row feature experiment only when a new full-data run is required:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\05_feature_engineering_with_mlflow.py"
```

Run a validation-only technical smoke check of the model-optimisation controls with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

After installing the pinned LightGBM dependency, add `lightgbm` with `--models`. The runner records validation threshold and cost tables but does not evaluate the held-out test split.

Run the fixed nine-configuration Random Forest tuning smoke test with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" --mode tuning --sample-rows 50000 --skip-data-hash
```

This tuning run always uses 5:1 training-only undersampling. Smoke scores confirm implementation only and must not be used as thesis findings.

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
- Review the full-data tuning decision with the supervisor and confirm the acceptance rule, alert-capacity assumption, and cost-sensitivity assumptions before any final test evaluation.
- If approved, freeze `rf_leaf_50`, 5:1 training-only undersampling, and one validation-derived operating threshold for a single held-out test evaluation.
- Register the evaluated preprocessing/model pipeline and threshold in MLflow with a `candidate` alias; assign `champion` only after an explicit acceptance decision.
- Implement FastAPI prediction serving and persist versioned prediction logs.
- Build batch data-quality, drift, prediction, performance, and expected-cost monitoring.
- Add rule-based investigation/retraining recommendations without fully automated retraining.
- Build the Streamlit monitoring dashboard.
- Package the MLflow, FastAPI, and Streamlit services for the planned local Docker deployment.
- Add serving, monitoring, cost, registration, and end-to-end integration tests.
- Define the human or rule-based promotion decision that selects an automated batch for a future ML experiment; do not retrain solely because data arrived.

## Status

Data understanding, minimal preparation, incremental raw/clean batch ingestion, the full Feast handoff, Feast historical retrieval, the canonical baseline comparison, the controlled feature experiment, the stage-06 model-family comparison, imbalance-strategy comparison, and full-data Random Forest tuning are complete. Data arrival triggers only validation and database ingestion; it does not trigger training or modify the frozen Feast `v1` baseline. `temporal_v2` won the validation-only feature comparison, but its test result did not confirm the improvement, so `original_v1` remains the stable reference. LightGBM did not improve the full-data model-family result. The tuning search selected Random Forest with 5:1 undersampling and a minimum leaf size of 50, but its validation Average Precision improvement was only 0.15% over the tuning reference and it did not improve the fixed-capacity results. It is therefore a research candidate, not an accepted or registered champion. The acceptance rule and operating threshold must be reviewed before any final held-out test evaluation. Serving, prediction logging, monitoring, the Streamlit dashboard, and Docker deployment remain to be completed.

The three UML diagrams in `Architecture_Diagram.drawio` describe the final target thesis prototype. They include both implemented components and the remaining serving and monitoring components listed above.

## Author

**Ishmita Basu**

Master's Thesis  
Otto von Guericke University Magdeburg
