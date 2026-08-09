# Shared utilities for the fraud-model experiments

The module <code>fraud_modeling_utils.py</code> contains the experiment rules used by the baseline model-comparison script and intended for reuse by later advanced experiments. Centralising these rules prevents small implementation differences—such as a changed split, metric definition, or logging convention—from making two model runs appear comparable when they are not.

The file uses <code># %%</code> markers so that its sections can be inspected interactively in a compatible editor, but it remains an ordinary importable Python module.

## Loading a consistent modeling table

The utility layer supports two data interfaces. The default path asks <code>fraud_feature_store.py</code> to reconstruct the versioned training table through Feast. The fallback path reads the cleaned gold CSV and checks that all required lineage, feature, and target columns are present.

In either case, timestamps are parsed into a consistent datetime type and the target is converted to a compact integer representation. Optional smoke samples are stratified so that the rare fraud class is still represented. The sample is then sorted by event time before splitting, ensuring that even a quick run follows the same chronological logic as the full experiment.

The <code>time_aware_split</code> function assigns the oldest rows to training, the next period to validation, and the newest rows to testing. It also validates the requested fractions and refuses configurations that would leave one of the three partitions empty.

## Training and threshold selection

The main experiment scripts build their own scikit-learn pipelines and pass them to <code>fit_evaluate_and_log</code>. This shared function separates model fitting, score prediction, threshold selection, evaluation, and tracking.

After fitting on the training partition, the function obtains continuous fraud scores from <code>predict_proba</code> or <code>decision_function</code>. It searches the validation precision–recall curve for the threshold that gives the highest F1 score. That threshold is fixed before the test partition is evaluated.

This distinction is important: PR-AUC measures ranking quality across possible thresholds, while precision, recall, specificity, F1, and confusion-matrix counts describe one chosen operating point. Selecting the operating point on validation data prevents the test results from influencing the decision.

## Metrics and diagnostic artifacts

The shared metric function calculates Average Precision, logged as <code>pr_auc</code>, together with ROC-AUC, accuracy, balanced accuracy, precision, recall, specificity, F1, and the four confusion-matrix counts.

For validation and test data, the module creates precision–recall curves, ROC curves, threshold trade-off plots, and confusion matrices. Very long curves are downsampled for plotting without changing the metrics themselves. When the estimator exposes coefficients or feature importances, a separate explanation artifact is also written.

## Reproducibility through MLflow

Before training, <code>configure_mlflow</code> creates or connects to the local SQLite tracking database and artifact directory. Each run receives the dataset metadata, feature list, chronological split summaries, pipeline parameters, random seed, timings, metrics, and tags supplied by the calling script.

The function also records the Python and operating-system versions, important package versions, source-code snapshots, and—when requested—the fitted scikit-learn model with an input example and inferred signature. Feast-backed runs add the feature version, feature service, entity key, handoff hashes, registry hash, definition hash, row count, and fraud rate.

The returned <code>RunResult</code> object gives the calling script the run ID, selected threshold, validation PR-AUC, validation metrics, and test metrics. Training and inference timings remain available in the corresponding MLflow run. The baseline comparison uses the returned values to build its separate summary run.

## Relationship to the modeling scripts

The responsibilities are intentionally divided:

- <code>04_baseline_model_comparison_with_mlflow.py</code> defines the non-skill, linear, and fixed nonlinear baseline configurations together with model-specific preprocessing.
- Later advanced-modeling scripts should add engineered features, resampling, tuning, or additional estimator libraries without changing the shared evaluation contract.
- <code>fraud_modeling_utils.py</code> makes sure both scripts load, split, evaluate, and log data in the same way.

As a result, changing a model configuration does not silently change the evaluation protocol around it. Any later feature-engineering script should reuse this utility layer so that its results remain comparable with the frozen original-feature benchmark.
