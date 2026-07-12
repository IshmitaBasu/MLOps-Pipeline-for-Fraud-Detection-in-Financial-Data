# 02 - Exploratory Data Analysis

This note explains `02_exploratory_data_analysis.ipynb`.

The notebook documents the exploratory data analysis step for the financial fraud detection pipeline. It is aligned with the MLOps-DSPM phase **Data Collection, Exploration, and Preparation**, especially **DP06 Data exploration**. The purpose is to understand the preprocessed dataset before model development, identify important data characteristics, and document observations that guide later modeling decisions.

The notebook is intentionally descriptive. It does not train a model, save figures, export analysis tables, perform feature selection, apply scaling, encode categorical variables, oversample fraud cases, or tune thresholds. Those steps belong to the later machine learning pipeline and must be fitted only after a train-validation-test split to avoid data leakage.

## Why this EDA is needed

Exploratory data analysis is needed because fraud detection datasets often have strong class imbalance, skewed transaction amounts, time-dependent patterns, and categorical differences across transaction contexts. Before building a model, the dataset must be understood at a descriptive level so that later design choices are justified rather than arbitrary.

In this project, EDA serves four roles:

- It checks whether the preprocessed gold table has the expected structure.
- It documents class imbalance and therefore justifies imbalance-aware evaluation metrics.
- It describes numeric, categorical, temporal, and amount-based patterns.
- It highlights potential data-quality or dataset-generation signals that should be considered before modeling.

## Input

The notebook reads:

```text
gold_financial_fraud_detection_table.csv
```

This file is created by `01_data_pipeline_preprocessing.py`. The EDA notebook does not go back to the raw dataset because the aim is to explore the cleaned, leakage-aware table that will be handed to the modeling pipeline.

## Output

The notebook itself is the output artifact. The cell outputs are visible directly below the code cells in the `.ipynb` file.

The notebook also contains inline `matplotlib` charts. These charts are displayed directly below the relevant code cells. They are not saved as separate diagram or image files.

No additional CSV summaries or image files are saved by the notebook.

## Cell-by-cell explanation

### Cell 1 - Import libraries

This cell imports `Path`, `pandas`, `matplotlib.pyplot`, and `display`.

`pandas` is used for tabular analysis. `Path` keeps file paths readable and reproducible. `matplotlib.pyplot` creates the inline charts shown later in the notebook. `display` makes data frames appear clearly below notebook cells. This keeps the EDA easy to inspect in VS Code or Jupyter.

### Cell 2 - Define paths and analysis columns

This cell defines the location of the gold table and lists the categorical, numeric, timestamp, and target columns.

The column lists are written explicitly so the analysis is auditable. Anyone reading the notebook can see exactly which variables are treated as categorical variables, which are treated as numeric variables, and which column is used as the fraud label.

### Cell 3 - Load the preprocessed gold table

This cell loads the preprocessed table and displays its shape, date range, and first rows.

This confirms that the EDA is based on the correct handoff artifact from preprocessing. The current dataset contains **4,277,262 transactions** and covers the period from **2023-01-01** to **2024-01-01**.

### Cell 4 - Validate the expected schema

This cell checks missing and extra columns, then displays the data type, missing-value count, missing percentage, and unique-value count for each column.

This is important because schema drift can silently break later modeling. For example, if a required feature disappears or changes type, model results may become invalid or difficult to interpret.

### Cell 5 - Create a high-level dataset overview

This cell summarizes the whole table using row count, column count, date range, fraud count, fraud rate, duplicate transaction IDs, missing values, and memory usage.

The overview gives the first compact description of the data. It also documents that the current preprocessed table has no remaining missing values and no duplicate transaction IDs.

### Cell 6 - Inspect the target distribution

This cell compares fraudulent and non-fraudulent transactions.

The current fraud rate is **3.4574%**, which means the dataset is clearly imbalanced. This justifies using evaluation metrics such as precision, recall, F1-score, PR-AUC, and confusion matrices later. Accuracy alone would be misleading because a model could achieve high accuracy by mostly predicting the majority class.

### Cell 7 - Summarize numeric features

This cell calculates descriptive statistics for the numeric features, including percentiles.

This helps understand feature ranges and distribution shapes. For example, transaction amount is skewed, which supports keeping `amount_log1p` as a transformed amount feature for later modeling. The cell is descriptive only and does not fit any scaler or transformation.

### Cell 8 - Compare numeric features by target class

This cell compares numeric feature statistics separately for non-fraud and fraud transactions.

The purpose is to identify visible differences between the two target classes. This is not feature selection. It simply documents whether fraud and non-fraud transactions show different descriptive patterns.

One important observation is that `time_since_last_transaction_missing_flag` behaves differently across target classes. Fraud cases have a mean of **0.0000**, while non-fraud cases have a mean of **0.2161**. This should be treated carefully as a possible data-quality or dataset-generation signal before modeling.

### Cell 9 - Analyze categorical features

This cell calculates row counts, fraud counts, average amounts, median amounts, row percentages, and fraud rates for categorical variables:

- `transaction_type`
- `merchant_category`
- `location`
- `device_used`
- `payment_channel`

This helps show whether some transaction contexts have slightly higher fraud rates than others. The differences are descriptive and should not be converted into target-based encodings at this stage, because that would risk leakage if fitted before splitting the data.

### Cell 10 - Analyze time-based patterns

This cell summarizes fraud rates by transaction hour, weekday, and month.

Time-based exploration is useful in fraud detection because behavior may vary by time of day, weekday, or month. The notebook currently finds the highest hourly fraud rate around **12:00**, but the difference is small and should be validated later during model training.

### Cell 11 - Analyze amount bands

This cell groups transaction amounts into interpretable ranges and calculates fraud rates for each band.

The reason for binning is interpretability. Instead of only looking at raw amount statistics, amount bands help show whether fraud is concentrated in low-value, medium-value, or high-value transactions. The current highest fraud rate appears in the **250-500** amount band.

### Cell 12 - Check simple numeric correlations with the target

This cell calculates Pearson correlations between numeric features and the binary target.

The result is only a descriptive check. It should not be used as final feature selection because feature selection must be performed inside the training workflow after splitting the data.

The strongest linear numeric association in the current data is `time_since_last_transaction_missing_flag`, with a correlation of approximately **-0.097160**.

### Cell 13 - Summarize the main EDA findings

This cell converts the most important results into short text findings.

These findings are useful for the thesis because they connect the technical analysis to interpretation. They also provide the rationale for later decisions, such as using imbalance-aware metrics and investigating suspicious missingness patterns before model development.


### Inline matplotlib chart cells

The notebook also includes an inline matplotlib chart section. These cells create charts directly inside the notebook output area without saving files to disk.

The chart cells use `matplotlib.pyplot` because it is a standard and thesis-friendly plotting library for Python EDA. The notebook calls `plt.show()` to display figures inline and deliberately avoids `plt.savefig()`, so no PNG files are written to disk.

#### Chart Cell 1 - Class distribution

The class distribution chart visualizes the imbalance between non-fraud and fraud transactions. This supports the later decision to use imbalance-aware evaluation metrics.

#### Chart Cell 2 - Fraud rate by transaction type

The transaction-type fraud-rate chart compares fraud rates across transaction types. This helps inspect whether certain transaction contexts show slightly higher fraud concentration.

#### Chart Cell 3 - Fraud rate by hour

The hourly fraud-rate chart shows how fraud rates vary by hour of day. This is relevant because time can represent behavioral context in transaction data.

#### Chart Cell 4 - Fraud rate by amount band

The amount-band fraud-rate chart groups transaction amounts into interpretable ranges. This makes amount-related patterns easier to discuss than raw numeric summaries alone.

#### Chart Cell 5 - Numeric correlation with the fraud target

The numeric-correlation chart visualizes simple Pearson correlations between numeric fields and the fraud target. This is descriptive only and does not replace proper feature selection inside the later train-only modeling workflow.

## Main findings from the current run

- The preprocessed table contains **4,277,262 transactions**.
- The dataset contains **147,881 fraud cases**.
- Fraud represents **3.4574%** of rows, confirming class imbalance.
- The highest supported fraud rate by transaction type is for `transfer`.
- The highest supported fraud rate by merchant category is for `entertainment`.
- The highest supported fraud rate by device is for `atm`.
- The highest fraud rate by hour appears around `12:00`.
- The highest fraud rate by amount band appears in `250-500`.
- `time_since_last_transaction_missing_flag` requires special attention because its distribution differs strongly between fraud and non-fraud rows.

## Leakage-control justification

This EDA is safe before modeling because it only produces aggregate descriptions and visual inspection through notebook outputs. It does not save learned preprocessing parameters or modify the modeling table.

The following steps are deliberately left for the later ML pipeline:

- train-validation-test splitting
- learned imputation
- scaling or normalization
- one-hot encoding or target encoding
- SMOTE or other resampling
- feature selection
- model training
- threshold tuning

This separation keeps the workflow reproducible and supports a leakage-aware MLOps pipeline.
