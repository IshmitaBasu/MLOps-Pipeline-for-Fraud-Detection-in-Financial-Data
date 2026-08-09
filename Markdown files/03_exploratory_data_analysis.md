# 03 – Exploring the cleaned transaction data

After the raw-data review and preprocessing stages, <code>03_exploratory_data_analysis.ipynb</code> examines the cleaned table in greater depth. The first notebook established what exists in the source; this notebook asks whether the retained variables contain patterns that should influence the modeling design.

The notebook reads <code>gold_financial_fraud_detection_table.csv</code>. It does not train a model or modify the gold table. Temporary columns are created in memory for grouping and plotting, but no engineered feature is written back to disk.

## Checking the handoff before analysis

The notebook begins by loading the thirteen-column gold table and reconciling it with the preprocessing report. It verifies the row count, timestamp range, class distribution, missing values, and retained columns. This step is important because the later charts should describe the actual modeling handoff rather than an earlier copy of the raw data.

The gold table still contains missing values in <code>time_since_last_transaction</code>. This is intentional: filling them before the chronological split would allow information from future rows to influence the imputation value used for older rows.

## Analyses performed in the notebook

The first detailed analysis compares transactions with and without a recorded time since the previous transaction. It shows both their share of the population and their fraud rate. The purpose is not merely to confirm that values are missing, but to reveal that missingness is associated with the target in this dataset. Because that relationship could reflect synthetic-data bias, a missingness indicator is reserved for a later sensitivity experiment rather than adopted automatically.

Transaction amounts are then compared by target class using empirical cumulative distribution functions. An ECDF shows the whole distribution without depending on arbitrary histogram bins, while the logarithmic x-axis keeps the strongly skewed amount range readable. A second amount analysis divides observations into twenty equal-frequency groups and measures fraud lift relative to the overall fraud rate. Together, these views test whether different parts of the amount distribution concentrate fraud cases.

For categorical variables, the notebook calculates fraud rates for transaction type, merchant category, location, device, and payment channel. The charts include 95% Wilson confidence intervals so that small visual differences are shown together with their uncertainty rather than being treated as established effects.

Temporal behavior is examined in two ways. An hour-by-weekday heatmap looks for interactions that would disappear in separate hourly and weekday summaries. A weekly view then places fraud rate, a four-week rolling mean, and transaction volume on the same timeline. This makes changes over time and incomplete boundary periods visible and supports the decision to evaluate models chronologically.

The remaining analyses focus on relationships among numerical variables. A Pearson correlation matrix provides a descriptive view of linear association, but it is not used to select features from the full dataset. Finally, a decile-by-decile heatmap combines spending-deviation and geographic-anomaly scores. This view asks whether combinations of anomaly levels reveal structure that is weak or invisible when either score is considered alone.

## Temporary analytical fields

To produce these summaries, the notebook derives hour, weekday, week, amount quantiles, score deciles, and a missingness-status field. They exist only for analysis inside the notebook. Their presence in a chart does not make them approved model inputs.

This distinction matters because the next experiments need a stable original-feature benchmark. Log amount, calendar fields, and the missingness indicator are plausible candidates, but each should be introduced through a named MLflow experiment so that its effect can be measured independently.

## How this stage informs modeling

The analysis motivates three main modeling choices. First, the low fraud prevalence makes precision–recall metrics more informative than accuracy alone. Second, temporal variation supports a train–validation–test split ordered by <code>event_timestamp</code>. Third, weak individual correlations alongside possible threshold and interaction patterns justify comparing a simple linear reference with nonlinear tree-based models.

These are reasons to test particular approaches, not claims that the EDA has already proved them superior. The next stage, <code>04_baseline_model_comparison_with_mlflow.py</code>, compares non-skill, linear, and fixed nonlinear baseline configurations using the untouched set of original predictors.
