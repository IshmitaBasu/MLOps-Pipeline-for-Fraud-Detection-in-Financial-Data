# 04 – Comparing baseline fraud models with MLflow

The baseline comparison in <code>04_baseline_model_comparison_with_mlflow.py</code> establishes the original-feature benchmark before advanced model development begins. It compares a non-skill reference, a linear baseline, and fixed nonlinear baseline configurations while keeping the feature set, chronological split, metrics, and tracking rules unchanged. Only the model family and its documented preprocessing differ.

This controlled design is important. If a new model and new features were introduced together, an improvement could not be attributed confidently to either one. Random Forest and Histogram Gradient Boosting are baseline comparison models here: they use fixed configurations and the original cleaned predictors. The later advanced stage is reserved for engineered features, resampling experiments, systematic hyperparameter tuning, and additional libraries such as XGBoost, LightGBM, or CatBoost.

## Models included in the baseline comparison

A default run evaluates four configurations:

| Configuration | Role in the experiment |
| --- | --- |
| <code>dummy_prior</code> | Non-skill reference based only on the training-set class prior |
| <code>logistic_sgd</code> | Scalable, balanced linear reference |
| <code>random_forest_depth_12</code> | Bagged tree ensemble for nonlinear thresholds and interactions |
| <code>hist_gradient_boosting_leaves_31</code> | Sequential tree ensemble with compact categorical preprocessing |

A depth-limited Decision Tree is available as an optional interpretable nonlinear check. A 31-neighbour k-NN model is also available, but only as a sampled feasibility experiment.

The choice of models follows from the EDA rather than from an assumption that a particular algorithm must win. Weak individual correlations justify retaining a linear reference, while the amount-lift and anomaly-interaction analyses suggest that thresholds and variable combinations may matter. Tree ensembles can learn those relationships without manually defining every interaction.

Random Forest uses 100 trees, a maximum depth of 12, a minimum leaf size of 100, square-root feature sampling, and balanced subsample weights. Histogram Gradient Boosting uses 150 iterations, 31 leaves, a learning rate of 0.1, a minimum leaf size of 100, balanced class weights, and no early stopping. These are fixed baseline configurations, not the outcome of test-set tuning.

## Data and evaluation design

The script loads the versioned Feast v1 feature set. The separate label table becomes the entity dataframe for historical retrieval, and the returned table is validated against the handoff metadata. A direct CSV fallback is available only when explicitly requested.

This ML entry point is intentionally independent from <code>automation/run_data_pipeline.py</code>. A new CSV arrival is appended to the raw and clean data-pipeline database tables, but it does not start this script or modify the canonical <code>v1</code> registry. Retraining therefore requires a separate manual or future rule-based snapshot decision and does not consume resources merely because data arrived.

An optional stratified sample can be taken for a smoke run. Whether sampled or complete, transactions are sorted by event time and divided into the same 70% training, 15% validation, and 15% test partitions. Every default model therefore sees equivalent historical periods.

Each model is fitted on the training partition. Its fraud scores on validation data are used both to calculate validation PR-AUC and to select the F1-maximizing decision threshold. The selected threshold is then applied unchanged to the chronological test partition.

The comparison summary chooses the preferred full-data baseline configuration by validation PR-AUC only. Test metrics are recorded as out-of-time evidence and are never used by the selection code. If a run contains only sampled feasibility configurations, the summary labels that reduced scope explicitly.

## Preprocessing for different model families

The linear model and optional k-NN pipeline median-impute and standardize numerical fields, then most-frequent-impute and one-hot encode categorical fields. Scaling is necessary for these models because their coefficients or distances depend on the numerical magnitude of each input.

Decision Tree and Random Forest use the same imputation and one-hot encoding but leave numerical variables unscaled. Tree splits depend on ordering rather than measurement units, so standardization would add work without changing their decisions.

Histogram Gradient Boosting uses a compact dense table. Numerical fields are median-imputed, while categories are most-frequent-imputed and ordinal encoded. A categorical mask tells the estimator which of the resulting columns represent categories. This avoids constructing a dense one-hot representation for five million transactions.

All preprocessing remains inside the fitted pipelines. No transformation is learned before the chronological split.

## Why k-NN is kept separate

A neighbour model stores its training observations and searches them during inference. That is expensive on a five-million-row dataset. The mixture of continuous variables and one-hot categories also makes distance harder to interpret: two transactions can appear close because of encoding choices rather than meaningful fraud similarity.

For that reason, selecting <code>knn_neighbors_31</code> triggers separate stratified limits for training, validation, and test rows. Its MLflow run is tagged <code>sampled_feasibility</code> and must not be compared as if it were a full-data result.

## What happens during a run

For each requested configuration, the script builds the appropriate scikit-learn pipeline and passes it to the shared training and evaluation function. That function fits the pipeline, calculates validation and test metrics, creates the diagnostic figures, and logs the fitted model with its signature and input example.

The script then creates a separate MLflow summary run. This summary stores every model’s run ID, model family, rationale, validation and test metrics, selected threshold, and comparison scope. It also records the baseline configuration chosen by validation PR-AUC and makes the selection rule explicit.

Each model run retains the data hashes, Feast lineage, feature list, split dates and class balance, complete scalar parameters, random seed, timing information, environment details, and source snapshots. These artifacts make it possible to reconstruct not only which model performed best, but also exactly which data contract and implementation produced the result.

## Running the comparison

Generate the handoff and apply the Feast definitions first:

~~~powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\02_data_pipeline_preprocessing.py"
Push-Location ".\Code snippets"
& "..\masters_thesis\Scripts\feast.exe" apply
Pop-Location
~~~

A smoke comparison can then be run with:

~~~powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\04_baseline_model_comparison_with_mlflow.py" --sample-rows 50000 --skip-data-hash
~~~

Omitting the sampling arguments runs the four core configurations on the complete dataset:

~~~powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\04_baseline_model_comparison_with_mlflow.py"
~~~

Optional checks are selected by name:

~~~powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\04_baseline_model_comparison_with_mlflow.py" --models decision_tree_depth_10

.\masters_thesis\Scripts\python.exe ".\Code snippets\04_baseline_model_comparison_with_mlflow.py" --models knn_neighbors_31 --knn-training-rows 50000 --knn-evaluation-rows 25000
~~~

The default Feast-backed experiment is <code>financial-fraud-feast-original-feature-baseline-comparison</code>. Log amount, timestamp-derived fields, resampling strategies, systematic tuning, and additional boosting libraries remain outside this comparison. They belong in the later advanced-modeling stage, which will reuse the selected baseline configuration and the same chronological evaluation protocol.
