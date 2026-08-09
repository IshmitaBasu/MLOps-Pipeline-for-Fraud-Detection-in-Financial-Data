# 01 – Initial exploration of the raw transaction data

The project begins with <code>01_initial_raw_data_eda.ipynb</code>. At this point the transaction file is treated as an untouched source: the notebook reads <code>financial_fraud_detection_dataset.csv</code>, examines what is present, and records the evidence needed to design the cleaning pipeline. It does not modify the CSV or make irreversible modeling decisions.

## How the notebook works

The notebook first defines the expected role of each field and then loads the complete dataset with explicit data types. This makes schema problems visible early instead of allowing pandas to infer different types from one run to another. After loading, it reports the number of rows and columns, the timestamp range, memory use, duplicate counts, and the uniqueness of <code>transaction_id</code>.

The next part examines data quality. Missing values are counted both for the full dataset and separately for fraudulent and legitimate transactions. Looking at missingness by target class matters here because <code>time_since_last_transaction</code> is not missing randomly: in this dataset its missing values occur only among non-fraud records. That pattern may later be useful as a feature, but it may also reflect how the synthetic data was generated. The notebook therefore records the observation without silently converting it into a model input.

## Understanding the target and predictors

Because fraud accounts for only about 3.59% of the transactions, the notebook reports both class counts and class proportions. This imbalance explains why accuracy alone would be misleading during model evaluation: a model could classify nearly everything as legitimate and still appear accurate.

Numeric variables are studied through descriptive statistics and distribution plots. Transaction amount is strongly right-skewed, while the remaining scores operate on different numerical scales. These findings suggest possible later experiments, such as a logarithmic amount transformation, but the notebook does not assume that a visually appealing transformation will improve out-of-time prediction.

Categorical variables are reviewed through frequencies and fraud rates. Temporal summaries show how transaction volume and fraud incidence vary across the available period. Identifier cardinality is also inspected so that account numbers, IP addresses, and device hashes are not mistaken for ordinary categorical predictors.

One particularly important check concerns <code>fraud_type</code>. The field is populated for fraud cases and absent for legitimate cases, which means it effectively reveals the target. The notebook identifies it as prediction-time leakage and recommends excluding it from the modeling data.

## What the notebook produces

All tables and Matplotlib figures remain inside the executed notebook. No cleaned dataset, chart file, or separate report is written, and the raw CSV is never changed. The final section brings the findings together and links each observation to a proposed action for the preprocessing stage.

Those actions are hypotheses and data-handling decisions, not evidence that an engineered feature is valuable. Feature usefulness is tested later against a tracked original-feature baseline. The immediate next step is <code>02_data_pipeline_preprocessing.py</code>, which converts the findings from this notebook into a reproducible, leakage-aware data handoff.

