# MLOps Pipeline for Financial Fraud Detection

> Master's Thesis Project | Otto von Guericke University Magdeburg

## About the Project

This folder contains the current data-analysis and preparation work for my Master's thesis on building an
end-to-end MLOps pipeline for financial fraud detection.

The project focuses on two things at the same time: detecting fraudulent transactions and showing how the
machine-learning workflow can be made reproducible, traceable, and maintainable. The current stage is the
data-understanding and data-preparation phase before baseline modeling.

## Current Workflow

The workflow now follows a clear process-model order:

1. Raw-data exploratory analysis
2. Data cleaning and minimal preparation
3. Advanced post-cleaning EDA
4. Baseline modeling with experiment tracking
5. Controlled feature-engineering experiments

This order is intentional. The initial EDA happens before cleaning so that the cleaning decisions are based
on evidence from the raw data. Feature engineering is not added directly to the preprocessing table. Instead,
the project will first train an original-feature baseline, then test engineered features as separate tracked
experiments.

## Completed Work

- Created an initial raw-data EDA notebook using the untouched source CSV.
- Confirmed that the raw dataset contains 5,000,000 rows and 18 columns.
- Confirmed that all timestamps are valid when parsed with an ISO-8601-aware parser.
- Identified `fraud_type` as target-derived leakage and excluded it from model inputs.
- Identified strong class imbalance: 179,553 fraud cases, about 3.59% of the data.
- Found that `time_since_last_transaction` missingness occurs only among non-fraud rows.
- Updated the preprocessing pipeline to create a minimally cleaned gold table.
- Preserved missing values for later train-only imputation instead of filling them globally.
- Avoided pre-baseline engineered features such as log amount, temporal fields, and missingness flags.
- Created an advanced post-cleaning EDA notebook with inline charts only.
- Updated the Markdown reports so they match the current notebooks and preprocessing logic.

## Current Data Artifacts

Raw input:

```text
financial_fraud_detection_dataset.csv
```

Cleaned output:

```text
gold_financial_fraud_detection_table.csv
```

The current gold table contains:

- 5,000,000 transactions
- 13 columns
- 179,553 fraudulent transactions
- fraud rate: approximately 3.59%
- no engineered model features
- missing `time_since_last_transaction` values preserved

The gold table keeps only the original usable predictors, `transaction_id` for lineage,
`event_timestamp` for time-aware splitting and analysis, and `is_fraud` as the target.

## Repository Structure

```text
Code snippets/
|-- 01_initial_raw_data_eda.ipynb              # Initial EDA on the untouched raw CSV
|-- 01_initial_raw_data_eda.md                 # Short documentation for the raw EDA notebook
|-- initial_raw_data_eda_report_v1.md          # Detailed raw EDA findings
|-- 02_data_pipeline_preprocessing.py          # Cleaning and minimal preparation pipeline
|-- 02_data_pipeline_preprocessing.md          # Preprocessing documentation
|-- data_quality_report_v1.md                  # Generated quality report from preprocessing
|-- 03_exploratory_data_analysis.ipynb         # Advanced EDA on the cleaned gold table
|-- 03_exploratory_data_analysis.md            # Short documentation for the advanced EDA notebook
|-- eda_report_v1.md                           # Detailed advanced EDA findings
|-- sample_financial_fraud_detection_dataset.csv
|-- README.md
```

Large raw and processed CSV files are local working artifacts and should not be committed to version
control.

## Why There Are Two EDA Stages

The first EDA notebook looks at the raw dataset before any cleaning. Its purpose is to understand the data,
spot quality issues, identify leakage risks, and decide what the preparation pipeline should do.

The advanced EDA notebook runs after the minimally cleaned gold table is created. Its purpose is different:
it checks whether the cleaned artifact makes sense and studies deeper relationships before modeling. The
advanced EDA creates temporary analysis variables inside the notebook, but it does not save those variables
as model features.

This keeps the workflow clean: EDA can suggest feature ideas, but the value of those ideas must be tested
later through tracked modeling experiments.

## Key Findings So Far

- The dataset is highly imbalanced, so accuracy alone will not be a useful evaluation metric.
- `fraud_type` is not a valid model input because it describes the known fraud outcome.
- Transaction amount is right-skewed, so `amount_log1p` is a candidate feature experiment after the baseline.
- Timestamp patterns may be useful, so hour, weekday, and month features should be tested later.
- Missingness in `time_since_last_transaction` is suspiciously related to the target and should be handled
  carefully as a separate sensitivity experiment.
- Numeric variables have weak direct linear correlation with the target, so nonlinear models or interactions
  may be worth comparing later.

## Running the Current Work

From the thesis workspace, run the preprocessing script with:

```bash
python "Code snippets/02_data_pipeline_preprocessing.py"
```

Then open the notebooks in order:

```text
01_initial_raw_data_eda.ipynb
03_exploratory_data_analysis.ipynb
```

The EDA notebooks display tables and charts inline. They do not save separate PNG, SVG, or chart-output
files.

## Next Steps

- Define the train-validation-test split, preferably with a time-aware strategy.
- Add experiment tracking, most likely with MLflow.
- Train an original-feature baseline model first.
- Track baseline metrics such as PR-AUC, ROC-AUC, precision, recall, F1-score, and confusion matrices.
- Test engineered features only after the baseline has been recorded.
- Compare feature-engineering experiments fairly using the same split and evaluation metrics.
- Continue later MLOps stages: model versioning, deployment, monitoring, and retraining.

## Status

Data understanding and minimal data preparation are aligned. The next major stage is baseline modeling with
experiment tracking.

## Author

**Ishmita Basu**

Master's Thesis  
Otto von Guericke University Magdeburg
