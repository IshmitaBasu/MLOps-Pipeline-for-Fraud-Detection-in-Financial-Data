# 01 - Initial raw-data exploratory analysis

This note documents `01_initial_raw_data_eda.ipynb`.

The notebook is the first analytical step after data collection and description. It reads the untouched
`financial_fraud_detection_dataset.csv` and performs exploratory analysis before any cleaning or feature
engineering decisions are applied.

All tables, findings, and Matplotlib charts are displayed directly inside the executed notebook. The
notebook does not save PNG, SVG, CSV, or separate EDA-report files. It also does not modify the raw CSV.

The analysis covers:

- dataset shape, schema, and date range;
- missing values and missingness by target class;
- duplicate rows and transaction IDs;
- amount and numeric distributions;
- class imbalance;
- categorical and temporal patterns;
- identifier cardinality;
- possible target leakage from `fraud_type`;
- observations that should guide the later cleaning and feature-experiment stages.

The final notebook table connects raw-data observations with proposed next actions. These are documented
decisions to assess, not proof that every engineered feature will improve a model. Feature value will be
tested later against a tracked baseline.
