# MLOps Pipeline for Financial Fraud Detection

> Master's Thesis Project | Otto von Guericke University Magdeburg

## About the Project

This repository contains the implementation and evidence for a Master's thesis on building an end-to-end MLOps pipeline for financial fraud detection.

The project focuses on detecting fraudulent transactions and making the machine-learning workflow reproducible, traceable, and maintainable. Data preparation, automated batch ingestion, Feast contracts, model comparisons, imbalance experiments, tuning, voting diagnostics, and final test evaluation are complete. The evaluated LightGBM uses the frozen validation maximum-F1 threshold, with separate review-workload scenarios. Test results confirm weak predictive performance. The fitted pipeline is used by a technically verified local research API with isolated frozen history and prediction logs. Its unchanged artifact is also registered as a research-only MLflow version; registry-backed API loading and production promotion have not occurred. A separate feasibility audit has accepted IBM HI-Small as a possible second dataset for a small AML baseline, without replacing the completed fraud pipeline.

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
9. Predictive-quality revision and behavioural diagnostics - complete
10. Controlled sender-location history feature experiment - complete
11. Versioned sender-location Feast contract and local online consistency - complete
12. Controlled oversampling comparison - complete; class weighting retained
13. Additional model-family comparison - complete; LightGBM selected
14. Limited LightGBM tuning - complete; reference configuration retained
15. Stage 13 validation threshold and voting-ensemble comparison - complete; LightGBM reference retained
16. Final evaluation rules - frozen; validation maximum-F1 threshold plus three workload scenarios
17. Stage 14 final test evaluation - complete; exact evaluated pipeline saved
18. Local FastAPI research serving and versioned prediction logging - implemented and technically verified; registry integration pending
19. Stage 16 research-model registration - complete; `financial_fraud_lightgbm_research`, version 1; no production promotion
20. Stage 17 training-only predictive-signal diagnostics - full training analysis completed; dataset-definition questions remain open
21. Stage 18 batch monitoring and human-gated investigation rules - implemented and smoke-checked; canonical reference pending
22. Stage 19 training-only multivariate-signal diagnostic - complete; no clear incremental interaction signal
23. Stage 20 IBM AML dataset feasibility audit - complete; retained as a second-dataset candidate
24. Stage 21 IBM AML small baseline - full validation complete; limited signal, test remains locked
25. Streamlit monitoring visualization - planned

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
- Frozen the canonical original-feature protocol, lineage, run IDs, metrics, and limitations in `docs/experiments/original_feature_benchmark_v1.md`.
- Created the review-ready baseline results document with full-data metrics, interpretation, DSPM alignment, and MLflow image placeholders in `docs/experiments/baseline_model_development_results_v1.md`.
- Completed the full-data controlled feature comparison while keeping the Random Forest fixed.
- Found that `temporal_v2` improved validation Average Precision by 0.87% but performed 0.38% worse than the original benchmark on the test period, so it was not accepted as a canonical replacement for `original_v1`.
- Restored the feature experiment, added deterministic calculation tests, and separated its plan, implementation guide, and completed results into clearly named documents.
- Updated the three-page UML architecture file with component, activity, and local deployment views for the final target prototype.
- Implemented the first Stage 07 diagnostic with a frozen stratified random split, training-only behavioural checks, past-only history calculations, MLflow artifacts, and automated leakage safeguards.
- Completed the full Stage 07 diagnostic and selected sender-location history for one controlled validation experiment.
- Implemented the Stage 08 control-versus-candidate runner with three point-in-time history features, fixed-workload fraud-value evaluation, and test-isolation safeguards.
- Completed the full Stage 08 comparison. `sender_location_history_v1` improved validation Average Precision by 5.69% and improved fraud-count and fraud-value recall at equal alert volumes, so it is retained for later validation experiments.
- Completed the additive `v2` Feast behavioural contract on all five million rows, with sender and sender-location entities, point-in-time state snapshots, a derived new-location flag, local SQLite materialization, and matching offline/online values for 50 checked entity rows.
- Added a training-only multivariate-signal diagnostic with an internal development/audit split, three fixed feature groups, four predeclared interactions, a shuffled-label control, stability slices, MLflow evidence and automated leakage checks.
- Completed the full Stage 19 diagnostic on 3.5 million frozen training rows. Real-label models exceeded the shuffled control, but the four explicit interactions added only 0.000108 AP over the supplied main fields and did not improve on the simpler core feature set.
- Completed the aggregate-only IBM HI-Small feasibility audit on 5,078,345 rows. All technical conditions passed, 5,177 laundering labels were present, and the dataset was retained as a second-dataset candidate rather than a replacement for the current fraud evidence.
- Implemented the fixed Stage 21 validation-only LightGBM baseline with primary-period filtering, deterministic stratified splitting, test isolation, fixed workload and stability evidence, currency-safe amount reporting and a non-progressing smoke run.
- Completed the Stage 21 full validation run. Average Precision was 0.006142, or 6.90 times the 0.0890% prevalence, but top-1% recall was only 9.00%; the fixed outcome is `discuss_limited_signal`, and the IBM test partition remains untouched.

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
|-- README.md                         # Project status, results and run commands
|-- requirements.txt                  # Python dependencies
|-- pyproject.toml                    # Formatting and test configuration
|-- stages/                           # Chronological runnable scripts, 02-21
|-- notebooks/                        # Stage 01 and Stage 03 exploratory notebooks
|-- fraud_detection/                  # Reusable implementation package
|   |-- common/                       # Paths, I/O, shared modelling and training data
|   |-- features/                     # History features and Feast helpers
|   |-- modeling/                     # Optimisation, resampling, model families and voting
|   |-- evaluation/                   # Validation and final-test rules
|   |-- diagnostics/                  # Predictive-signal investigations
|   |-- serving/                      # Research API and registered-model wrapper
|   |-- monitoring/                   # Monitoring and reference-profile logic
|   `-- datasets/                     # IBM AML feasibility and baseline logic
|-- docs/
|   |-- index.md                      # Documentation starting point
|   |-- experiments/                  # Plans, guides, results and interpretation
|   |-- images/                       # Committed evidence figures
|   |-- architecture/                 # Draw.io architecture source
|   `-- reports/                      # Notebook-generated reports
|-- tests/                            # Automated leakage, contract and behavior checks
|-- automation/                       # Independent batch-ingestion trigger
|-- feature_repo/                     # Feast data and metadata area
|-- tools/                            # Documentation maintenance utilities
|-- feature_store.yaml                # Feast repository configuration
`-- fraud_feature_definitions.py      # Feast discovery entry point
```

Raw CSV files, the MLflow database, MLflow artifacts, serving state, monitoring runs and candidate datasets are local generated artifacts. They remain in their existing locations and are excluded from Git.

The quickest documentation entry point is [`docs/index.md`](docs/index.md).

## Documentation Map

| Question | Document to read |
| --- | --- |
| What was planned for the feature experiment, and why? | `docs/experiments/05_feature_engineering_experiment_plan.md` |
| How does the feature script work, and how is it run? | `docs/experiments/05_feature_engineering_implementation_guide.md` |
| What values were obtained, and which features should be kept? | `docs/experiments/05_feature_engineering_results.md` |
| What is the current plan for model optimisation, imbalance handling, thresholds, and acceptance? | `docs/experiments/06_model_optimization_and_operational_evaluation_plan.md` |
| How do the stage-06 screening and tuning modes work, and what command should be run? | `docs/experiments/06_model_optimization_and_operational_evaluation_implementation_guide.md` |
| What has been run in stage 06, what values were obtained, and what do they mean? | `docs/experiments/06_model_optimization_and_operational_evaluation_results.md` |
| What is planned after the review of the weak predictive results? | `docs/experiments/07_predictive_quality_improvement_plan.md` |
| How does the first Stage 07 diagnostic work, and how should it be run? | `docs/experiments/07_predictive_quality_improvement_implementation_guide.md` |
| Which Stage 07 runs have been completed, what values were obtained, and what follows? | `docs/experiments/07_predictive_quality_improvement_results.md` |
| What is the controlled sender-location feature question and decision rule? | `docs/experiments/08_sender_location_history_feature_experiment_plan.md` |
| How is the Stage 08 feature comparison implemented and run? | `docs/experiments/08_sender_location_history_feature_experiment_implementation_guide.md` |
| What Stage 08 values were obtained and did the feature group progress? | `docs/experiments/08_sender_location_history_feature_experiment_results.md` |
| How is the retained history represented and served consistently through Feast? | `docs/experiments/09_sender_location_feature_store_implementation_guide.md` |
| What Stage 09 materialization and consistency results were obtained? | `docs/experiments/09_sender_location_feature_store_results.md` |
| What imbalance strategies will Stage 10 compare, and how will one be selected? | `docs/experiments/10_oversampling_experiment_plan.md` |
| How is the Stage 10 smoke workflow implemented and run? | `docs/experiments/10_oversampling_experiment_implementation_guide.md` |
| What Stage 10 runs have completed and what were their results? | `docs/experiments/10_oversampling_experiment_results.md` |
| Why does Stage 11 repeat earlier models, which additional families will it compare, and how can one progress? | `docs/experiments/11_model_family_comparison_plan.md` |
| How is the Stage 11 comparison controlled and executed? | `docs/experiments/11_model_family_comparison_implementation_guide.md` |
| What Stage 11 checks and model runs have completed? | `docs/experiments/11_model_family_comparison_results.md` |
| What LightGBM settings will Stage 12 test, and how can one be retained? | `docs/experiments/12_lightgbm_tuning_plan.md` |
| How is the limited LightGBM tuning workflow run safely? | `docs/experiments/12_lightgbm_tuning_implementation_guide.md` |
| What Stage 12 checks and tuning runs have completed? | `docs/experiments/12_lightgbm_tuning_results.md` |
| Why compare voting classifiers and which candidates are fixed? | `docs/experiments/13_voting_classifier_comparison_plan.md` |
| How are voting and threshold diagnostics run, and where are their artifacts? | `docs/experiments/13_voting_classifier_comparison_implementation_guide.md` |
| What Stage 13 checks and model comparisons have completed? | `docs/experiments/13_voting_classifier_comparison_results.md` |
| Which model and rules are fixed for final test evaluation? | `docs/experiments/14_final_test_evaluation_plan.md` |
| How does final evaluation prevent test tuning and repeated scoring? | `docs/experiments/14_final_test_evaluation_implementation_guide.md` |
| What are the final test metrics and their limitations? | `docs/experiments/14_final_test_evaluation_results.md` |
| What must the API receive, what does Feast supply, and how will serving parity be verified? | `docs/experiments/15_model_serving_feature_contract.md` |
| How is the local research prediction API started and its private state managed? | `docs/experiments/15_research_prediction_api_implementation_guide.md` |
| Which serving checks have passed, and what is still unimplemented? | `docs/experiments/15_research_prediction_api_results.md` |
| How is the evaluated model registered, and what happens when a future model changes? | [Stage 16 registration and future-version procedure](docs/experiments/16_research_model_registration.md) |
| How does the training-only predictive-signal investigation work? | [Stage 17 plan and run commands](docs/experiments/17_predictive_signal_diagnostics_plan.md) |
| Which Stage 17 checks and diagnostic runs have completed? | [Stage 17 results](docs/experiments/17_predictive_signal_diagnostics_results.md) |
| Where did the dataset come from, and which field definitions remain uncertain? | [Dataset source and field audit](docs/experiments/17_dataset_source_and_field_audit.md) |
| What does Stage 18 monitor, and which rules lead to investigation? | [Stage 18 monitoring plan](docs/experiments/18_batch_monitoring_plan.md) |
| How are reference profiles, CSV batches and the API ledger monitored? | [Stage 18 implementation guide](docs/experiments/18_batch_monitoring_implementation_guide.md) |
| Which Stage 18 checks have completed? | [Stage 18 results](docs/experiments/18_batch_monitoring_results.md) |
| How does the training-only multivariate-pattern check work, and how is it run? | [Stage 19 multivariate-signal diagnostic](docs/experiments/19_multivariate_signal_diagnostics.md) |
| Why was IBM HI-Small examined, and what did the feasibility audit conclude? | [Stage 20 IBM AML dataset feasibility audit](docs/experiments/20_ibm_aml_dataset_feasibility_audit.md) |
| What was fixed for the IBM AML baseline, and what did validation conclude? | [Stage 21 IBM AML baseline plan and results](docs/experiments/21_ibm_aml_baseline_plan.md) |
| Where are precision, recall, F1, accuracy, balanced accuracy, and confusion counts for the revised models? | [Stage 11 classification comparison](docs/experiments/11_model_family_comparison_results.md#classification-metrics-at-each-models-maximum-f1-rule) |
| What do the retained LightGBM's complete metrics and different review workloads mean? | [Stage 12 consolidated validation report](docs/experiments/12_lightgbm_tuning_results.md#complete-validation-metrics-for-the-retained-lightgbm) |
| What is the frozen original-feature benchmark? | `docs/experiments/original_feature_benchmark_v1.md` |
| What are the detailed baseline model results? | `docs/experiments/baseline_model_development_results_v1.md` |
| How do preprocessing, Feast, and automated ingestion work? | `docs/experiments/02_data_pipeline_preprocessing.md`, `docs/experiments/feature_store_handoff.md`, and `docs/experiments/automated_data_pipeline.md` |

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
- The full Stage 07 training-only diagnostic found that sender-location history was the only examined entity to pass the predefined feasibility gate: 20.844% history coverage and a 15.638% relative difference between new and returning fraud rates. This is feature-engineering evidence, not validation performance; the test split remained closed.
- The full controlled Stage 08 comparison increased validation Average Precision from 0.043949 to 0.046448, a 5.69% relative improvement. At a 5% alert volume, the history candidate captured 1,790 fraud cases instead of 1,669 and increased fraud-value recall from 2.90% to 6.80%. The candidate proceeds to later validation experiments; it is not a final or test-approved model.
- The full Stage 12 tuning comparison retained the Stage 11 LightGBM reference at 0.046721 validation Average Precision. Two alternatives produced increases below 0.2% but reduced both fraud-count and fraud-value recall at the fixed 5% workload, so they did not pass the progression rule. The test split remained closed.
- The full Stage 13 voting comparison also retained LightGBM. Equal-weight ensembles increased AP by less than 0.2% but found fewer fraud cases and less fraudulent value at the 5% workload. The maximum-F1 LightGBM threshold, 0.522602, flags about 70% of validation transactions; Stage 14 subsequently froze it as the primary research benchmark.
- Stage 14 applied the frozen score threshold once to the revised test set. Test AP is 0.046860 and F1 is 0.086240. Recall is 88.3303%, but precision is only 4.5333% and 69.9707% of transactions are flagged. At a 5% batch review workload, fraud-count recall is 6.4828%. The weak validation trade-off persists in final testing.

## Running the Current Work

From the thesis workspace, run the preprocessing script with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\02_data_pipeline_preprocessing.py"
```

This unchanged command writes the canonical `v1` gold CSV and Parquet feature/label handoff used by the current ML pipeline. Register the definitions after the handoff files exist:

```powershell
Push-Location ".\Code snippets"
& "..\masters_thesis\Scripts\feast.exe" apply
Pop-Location
```

Then open the notebooks in order:

```text
notebooks/01_initial_raw_data_eda.ipynb
notebooks/03_exploratory_data_analysis.ipynb
```

The EDA notebooks display tables and charts inline. They do not save separate PNG, SVG, or chart-output files.

Run a quick MLflow smoke baseline comparison with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\04_baseline_model_comparison_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

Run the full original-feature baseline comparison with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\04_baseline_model_comparison_with_mlflow.py"
```

Run an optional Decision Tree or sampled k-NN check only if you need a sensitivity analysis:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\04_baseline_model_comparison_with_mlflow.py" --models decision_tree_depth_10 knn_neighbors_31 --sample-rows 50000 --skip-data-hash
```

The baseline comparison uses Feast by default. The old direct-CSV path remains available only as an explicit fallback by adding `--data-source csv`.

Run a quick technical smoke check of the feature experiment with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\05_feature_engineering_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

Reproduce the full five-million-row feature experiment only when a new full-data run is required:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\05_feature_engineering_with_mlflow.py"
```

Run a validation-only technical smoke check of the model-optimisation controls with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\06_model_optimization_and_operational_evaluation_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

After installing the pinned LightGBM dependency, add `lightgbm` with `--models`. The runner records validation threshold and cost tables but does not evaluate the held-out test split.

Run the fixed nine-configuration Random Forest tuning smoke test with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\06_model_optimization_and_operational_evaluation_with_mlflow.py" --mode tuning --sample-rows 50000 --skip-data-hash
```

This tuning run always uses 5:1 training-only undersampling. Smoke scores confirm implementation only and must not be used as thesis findings.

Run the first Stage 07 training-only feature-feasibility smoke check with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\07_predictive_quality_improvement_with_mlflow.py" --mode diagnostics --sample-rows 50000 --skip-data-hash
```

This run checks whether the raw account, receiver, device, IP, and location keys contain enough repeated behaviour to justify historical feature engineering. It creates the revised random split but does not inspect validation or test features. Record the output in `docs/experiments/07_predictive_quality_improvement_results.md`; smoke values are technical checks only.

Run the controlled Stage 08 sender-location feature smoke comparison with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\08_sender_location_history_feature_experiment_with_mlflow.py" --mode comparison --sample-rows 50000 --skip-data-hash
```

This fits the same Random Forest with `original_v1` and with the three added history features. It compares validation ranking, equal-workload fraud capture, and fraud-value capture without materialising or evaluating the test partition. Smoke scores are technical checks only.

Run the isolated Stage 09 Feast smoke test with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\09_sender_location_feature_store.py" --mode smoke
```

This uses a temporary Feast registry and SQLite online store. It verifies
historical/online consistency without changing the canonical feature artifacts.

Run the Stage 12 limited LightGBM tuning smoke check with:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\12_lightgbm_tuning_with_mlflow.py" --mode smoke
```

This reuses the completed Stage 11 decision, retained features, class weighting, and frozen split. It compares only the predeclared LightGBM configurations on validation data; smoke results cannot select a configuration and the test partition remains closed.

Run the Stage 13 voting and threshold smoke check first:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\13_voting_classifier_comparison_with_mlflow.py" --mode smoke
```

After checking its output, use `--mode full` for the full validation comparison. It fits LightGBM alone, an equal-weight LightGBM–CatBoost soft vote, and an equal-weight LightGBM–CatBoost–Random Forest soft vote. Each candidate records complete metrics and threshold trade-offs. Smoke mode cannot select a model; neither mode evaluates the test split or freezes a final operating rule.

For the separate validation-only hard-voting follow-up, use the same runner:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\13_voting_classifier_comparison_with_mlflow.py" --voting hard --mode smoke
```

After checking the smoke output, change `--mode smoke` to `--mode full`. Results go to `financial-fraud-stage-13-hard-voting-followup`. This reports native class decisions and matched soft policies without changing the frozen model, API threshold, or completed test evaluation. See the Stage 13 plan and implementation guide for the tie rule and comparison limits.

The hard-voting full-data follow-up completed on 8 October 2026 (summary run `c7aadb13580746b4a58018aa8e506deb`). It did not resolve the low-precision/high-alert-volume limitation, and no model was promoted. Full metrics, matched soft policies, and confusion-matrix figures are recorded in the Stage 13 results document.

Run the automated test suite with:

```powershell
Push-Location ".\Code snippets"
try {
    & "..\masters_thesis\Scripts\python.exe" -m pytest tests -q
} finally {
    Pop-Location
}
```

Open the local MLflow interface on Windows with one worker:

```powershell
.\masters_thesis\Scripts\mlflow.exe ui --workers 1 --backend-store-uri "sqlite:///D:/Germany/Documents/Magdeburg/Semester Documents/Sem 5/Thesis/Code snippets/mlflow_tracking.db"
```

## Training-only predictive-signal investigation

The [dataset source and field audit](docs/experiments/17_dataset_source_and_field_audit.md) records the confirmed Aryan208 Kaggle origin, also linked by an official DuckDB example. Published negative elapsed-time values were found, but field formulas, generator details and fraud-label generation remain unverified. No values were automatically corrected and no model was changed.

The training-only data investigation is implemented in `stages/17_training_predictive_signal_diagnostics_with_mlflow.py`. See the [Stage 17 plan and commands](docs/experiments/17_predictive_signal_diagnostics_plan.md) and [results log](docs/experiments/17_predictive_signal_diagnostics_results.md). It extends earlier EDA with missingness, practical association sizes, bounded interactions and training-subset stability, without fitting models or changing the registry/API.

The smoke run (`e495312ff9444136ba84af9b4de43bfa`) and full training run (`8912b6cc67ca47fb8b5f44d66be6ddd8`) completed on 9 October 2026. The experiment is `financial-fraud-stage-17-training-predictive-signal`. Smoke outputs are technical checks, not thesis evidence. The full run found no fraud labels among 627,581 training rows with missing time-since-last values, weak marginal associations for most original features, and modest behavioural associations. Negative elapsed-time values and unverified dataset/label definitions need clarification. Full values, limitations and four evidence figures are recorded in the results log; the frozen model and API are unchanged.

## Batch monitoring

Stage 18 implements aggregate [batch monitoring rules](docs/experiments/18_batch_monitoring_plan.md) for data quality, input and prediction drift, delayed-label performance, fraud-value recall and assumption-based cost. The [implementation guide](docs/experiments/18_batch_monitoring_implementation_guide.md) explains reference preparation, CSV monitoring and read-only API-ledger monitoring. The [results log](docs/experiments/18_batch_monitoring_results.md) records the synthetic monitoring smoke run `fb6bf3f939f645ad89ea2bc161aacf74` and reproducible training-history smoke runs `4ef6554919034959a1aa69b8ae08a73f` and `e71177954aa04b9585d19a50b3df4f08`.

The smoke run proves that a known shift produces a human investigation recommendation while automatic retraining remains disabled. It is not real drift or model-performance evidence. A canonical scored reference is still required, and the consumed test partition cannot be used for that purpose.

## Training-only multivariate-signal investigation

Stage 19 is implemented in `stages/19_training_multivariate_signal_diagnostics_with_mlflow.py` and explained in the [single Stage 19 document](docs/experiments/19_multivariate_signal_diagnostics.md). It uses only frozen training rows, makes a reproducible internal development/audit split, and compares prevalence, the strongest Stage 17 marginal pattern, fixed multivariate feature groups, explicit interactions and a shuffled-label control. Aggregate stability checks use four time slices and two ID-hash halves. The full run `8b3b5e25606c45d987cd6ef4dce8a657` found real but weak multivariate signal and no clear incremental interaction signal. It did not change the registered model, frozen threshold, API or completed test evidence.

## IBM AML second-dataset feasibility

Stage 20 is described in the [IBM AML dataset feasibility audit](docs/experiments/20_ibm_aml_dataset_feasibility_audit.md). The read-only, chunked audit verified the local IBM HI-Small file and inspected all 5,078,345 rows without exporting account identifiers or training a model. The corrected full run `5bdc8b842b5e4ac5a88925e5935ef6ea` found 5,177 laundering labels, clean structure and timestamps, and high past-only sender and receiver history coverage. All fixed technical conditions passed.

The outcome is `second_dataset_candidate`, not dataset replacement. The target is synthetic money laundering rather than general transaction fraud, the raw rows require chronological sorting, and the unusually enriched late tail is excluded using the boundary fixed before modelling. The [Stage 21 baseline protocol and results](docs/experiments/21_ibm_aml_baseline_plan.md) record the primary-period population, stratified split, one untuned LightGBM, validation metrics and stopping rule. The successful full run `94ccbdf8542f41f88b8b9fbb6d694db2` achieved 6.90 times prevalence Average Precision and stable subgroup lifts, but top-1% recall was only 9.00%. Its outcome is `discuss_limited_signal`; the held-out IBM test remains locked. The existing model, registry, API, Feast contract and completed fraud-test evidence are unchanged.

## Research model registry

Model registration is complete and described in [16_research_model_registration.md](docs/experiments/16_research_model_registration.md). Version `1` is pinned by `models:/financial_fraud_lightgbm_research/1`. Its cell-based runner registers the unchanged evaluated artifact without retraining, test scoring, production promotion, or an automatic API switch. Open **Models** in the local MLflow UI to inspect it.

The [future-version procedure](docs/experiments/16_research_model_registration.md#registering-a-future-model-version) explains how a new candidate would be evaluated, packaged, registered, checked for feature/runtime compatibility, deliberately deployed, and rolled back. It explicitly identifies mechanisms that remain unimplemented; rerunning the current script only verifies the frozen version-1 package.

Read-only verification from the thesis root:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\16_register_research_model.py" --mode verify
```

## Documentation figures

The result documents for stages 07–14 include training-only feature screening, validation comparisons, confusion matrices, threshold trade-offs, and final test review scenarios where saved evidence supports them. Stage 09 is explained through the architecture diagram, and Stage 15 includes the API verification screenshots. Historical reconstructed confusion counts are labelled explicitly; figures do not imply new experiments or statistical significance.

The images are committed under `docs/images/experiment_figures/`. Their source artifact paths and SHA-256 hashes are listed in [figure_sources.json](docs/images/experiment_figures/figure_sources.json). Saved local MLflow artifacts are required only to regenerate them, not to read the documentation.

To regenerate the figures from the thesis root:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\tools/generate_documentation_figures.py"
```

This script reads recorded JSON and CSV results only. It does not train models, score transactions, modify MLflow evidence, or repeat the final test evaluation. ROC and precision-recall curves are not fabricated from aggregate metrics.

## Next Steps

- Share the completed Stage 21 validation evidence and decide whether the limited IBM signal justifies one further controlled experiment; keep its test partition locked meanwhile.
- Preserve the completed Stage 19 aggregate evidence and use it when explaining why the dataset supports only weak fraud discrimination.
- Add the specified MLflow images to the baseline comparison document and complete its final human review.
- Communicate the completed validation, voting, and final test findings, with the model's limitations.
- Extend the implemented future-online adapter's parity evidence for historical training/validation replay if that separate path is needed; never replay the consumed final test for selection.
- Preserve the registered research model, frozen benchmark, and original test evidence; do not retune against the completed test results.
- Integrate explicit research-model registration with the implemented local FastAPI serving and versioned prediction ledger.
- Freeze a provenance-controlled scored monitoring reference without using the consumed test partition.
- Run Stage 18 on sufficiently large future API batches and join delayed labels through a separately controlled process.
- Build the Streamlit monitoring dashboard.
- Package the MLflow, FastAPI, and Streamlit services for the planned local Docker deployment.
- Add serving, monitoring, cost, registration, and end-to-end integration tests.
- Define the human or rule-based promotion decision that selects an automated batch for a future ML experiment; do not retrain solely because data arrived.

## Status

Modelling through Stage 14 final evaluation is complete. The final test was scored once at validation-frozen rules; it confirmed weak fraud discrimination and an excessive review burden at the primary benchmark. The exact fitted LightGBM pipeline and protocol are preserved in MLflow run `37b41018f28646598344fbcfe4ea4db6`. The revised test partition is now evaluated and cannot be reused for further tuning. The unchanged artifact is registered as research model `financial_fraud_lightgbm_research`, version 1, with production approval false. Data arrival still triggers only validation and database ingestion, rather than training or modification of the frozen Feast baseline. Stage 15 implements local future-online FastAPI serving using an isolated frozen history and persistent prediction ledger. Stage 18 implements batch monitoring and human-gated investigation rules; its canonical reference is pending. Stage 19 is complete and confirms real but weak multivariate signal without a useful increment from the four tested interactions. Stage 20 accepts IBM HI-Small only as a technically feasible second-dataset candidate. Stage 21 full validation found stable but operationally limited signal, so its test partition remains locked pending a scope decision. Registry-backed API loading, the dashboard, and container deployment remain to be implemented.

Stage 13 voting and threshold diagnostics are also complete. Neither ensemble passed the equal-workload progression rule, so LightGBM remains the development model. The full results and threshold trade-offs are documented in `docs/experiments/13_voting_classifier_comparison_results.md`.

The three UML diagrams in `docs/architecture/Architecture_Diagram.drawio` describe the target thesis prototype and distinguish implemented workflows, local research serving, research registration, and the Stage 18 monitoring engine from pending registry-to-API integration, a canonical monitoring reference, dashboard, and Docker components. The activity diagram shows validation-only selection, a frozen model and rule, and the completed one-time final test without a feedback path into tuning. The evaluated LightGBM pipeline is a research artifact, not a production-approved champion. Both modelling histories and Feast v2 counts use the frozen training population. The implemented online adapter verifies frozen latest-state counts and rejects historical timestamps; a historical replay endpoint remains unimplemented. Its design and checks are recorded in the three Stage 15 documents.

## Author

**Ishmita Basu**

Master's Thesis  
Otto von Guericke University Magdeburg
