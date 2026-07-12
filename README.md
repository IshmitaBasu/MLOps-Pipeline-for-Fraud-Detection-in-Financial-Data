# MLOps Pipeline for Financial Fraud Detection

> Master's Thesis Project | Otto von Guericke University Magdeburg

## About the Project

This repository contains the implementation of my Master's thesis on building an end-to-end MLOps pipeline for financial fraud detection.

Financial fraud detection has become increasingly challenging due to the growing volume of digital transactions and the constantly evolving nature of fraudulent activities. While many studies focus on improving machine learning models, this project explores how those models can be developed within a reproducible and maintainable MLOps workflow.

The goal is to build a prototype that detects fraudulent transactions while demonstrating experiment tracking, model versioning, deployment, and monitoring throughout the machine learning lifecycle.

---

## Current Progress

The data preparation and exploratory analysis phase is complete. The project is now moving into leakage-aware model development.

### Completed

- Thesis proposal and project planning
- Repository setup and initial pipeline architecture
- Raw dataset loading, schema validation, and data-quality profiling
- Removal of exact duplicates, invalid negative amounts, and invalid timestamps
- Missing-value handling for `time_since_last_transaction` using a missing-value flag and a deterministic sentinel value
- Removal of direct identifiers and the label-derived `fraud_type` field from the modeling table
- Leakage-safe row-level feature preparation, including amount and timestamp features
- Creation of the 18-column preprocessed gold table
- Exploratory Data Analysis (EDA) covering class balance, numeric, categorical, temporal, and transaction-amount patterns
- Documentation of the preprocessing pipeline and EDA notebook

### Currently Working On

- Literature review
- Designing the train-validation-test strategy, including consideration of a time-aware split
- Preparing the baseline machine learning pipeline
- Selecting imbalance-aware evaluation metrics and baseline models
- Reviewing the strong relationship between the missing-value flag and the target before modeling

### Planned Next Steps

- Fit categorical encoding, scaling, and any learned imputation on training data only
- Train and compare baseline fraud-detection models
- Address class imbalance through class weighting and/or resampling within the training pipeline
- Evaluate models using precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion matrices
- Tune the decision threshold using validation data
- Add experiment tracking with MLflow
- Add data and model versioning with DVC
- Deploy the selected model through FastAPI and Docker
- Implement monitoring and a retraining workflow

---

## Project Goals

This project aims to:

- Build a reproducible fraud-detection pipeline
- Compare multiple machine learning models
- Handle highly imbalanced financial data
- Apply MLOps practices throughout the ML lifecycle
- Improve experiment reproducibility and traceability

---

## Current Data Pipeline

The preprocessing script validates the expected schema, profiles data quality, applies conservative cleaning, and creates deterministic row-level features. Transformations that learn from the data distribution are deliberately postponed until after the train-validation-test split to prevent data leakage.

The resulting gold table contains:

- 4,277,262 transactions
- 18 columns
- 147,881 fraudulent transactions
- A fraud rate of approximately 3.46%
- No remaining missing values after the documented preprocessing decisions

The completed EDA confirms that the target is imbalanced and identifies a notable relationship between `time_since_last_transaction_missing_flag` and the fraud label. This signal will be reviewed carefully during model development.

---

## Tech Stack

Currently used:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Planned additions:

- MLflow
- DVC
- FastAPI
- Docker

---

## Repository Structure

```text
.
|-- 01_data_pipeline_preprocessing.py     # Data validation, cleaning, and feature preparation
|-- 01_data_pipeline_preprocessing.md     # Preprocessing documentation
|-- 02_exploratory_data_analysis.ipynb    # Executed exploratory analysis
|-- 02_exploratory_data_analysis.md       # EDA documentation and findings
|-- Architetcure Diagram.drawio           # Initial pipeline architecture diagram
|-- sample_financial_fraud_detection_dataset.csv
|-- README.md
```

Large raw and processed datasets are not stored in this repository. The sample CSV is included only to show the dataset structure.

---

## Dataset and Key Findings

The project uses a public synthetic financial fraud detection dataset. The preprocessing pipeline produces a documented gold table for EDA and later model development.

Current findings include:

- Fraud accounts for approximately 3.46% of the preprocessed transactions, confirming class imbalance.
- Transaction amounts are strongly skewed, motivating the leakage-safe `amount_log1p` feature.
- Fraud rates vary slightly across transaction types, merchant categories, devices, hours, and amount bands.
- The missing-value flag for `time_since_last_transaction` differs substantially between fraud and non-fraud rows and may reflect a data-generation pattern.
- Encoding, scaling, resampling, feature selection, model training, and threshold tuning remain split-dependent steps and have not yet been applied.

---

## Running the Current Work

Run the preprocessing script from the thesis workspace:

```bash
python "Gitpush/01_data_pipeline_preprocessing.py"
```

Then open and run `02_exploratory_data_analysis.ipynb` after confirming that `gold_financial_fraud_detection_table.csv` is available in the configured data directory.

---

## Status

Work in progress — data preprocessing and exploratory analysis are complete; model development and MLOps integration are the next major stages.

---

## Author

**Ishmita Basu**

Master's Thesis  
Otto von Guericke University Magdeburg
