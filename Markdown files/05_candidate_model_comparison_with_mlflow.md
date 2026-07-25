# 05 - Candidate-model comparison with MLflow

This note documents `05_candidate_model_comparison_with_mlflow.py`.

## Current structure

The file is organized as a cell-wise Python script using `# %%` markers. The cells separate imports, model
candidate configuration, preprocessing factories, pipeline construction, experiment metadata helpers, and the
main comparison runner. The script still works normally from the command line.

## Purpose

This stage compares a compact set of EDA-justified candidate models without changing the original cleaned feature set. Keeping
features and chronological splits fixed means that observed performance differences can be attributed to
the model choice or its hyperparameters rather than simultaneous feature engineering.

The default comparison includes:

- prior-only `DummyClassifier` as the non-skill reference;
- logistic `SGDClassifier` as the scalable linear reference;
- Random Forest;
- histogram-based gradient boosting.

A constrained Decision Tree and k-NN are available as optional checks. k-NN is restricted to explicitly
logged training and evaluation samples because it stores training observations and performs expensive
neighbor searches during inference.

## Leakage-aware evaluation protocol

Each default model has one documented configuration. The model is fitted on the training split, its decision
threshold is selected on validation data, and its final metrics are calculated on the chronological test
split. The summary run selects the preferred full-data candidate by validation PR-AUC and records the test
metrics as final out-of-time evidence.

## Model-specific preprocessing

Linear and distance-based models receive median-imputed and standardized numerical fields,
plus most-frequent-imputed and one-hot-encoded categorical fields. Decision Tree and Random Forest use the
same imputers and sparse one-hot categories but do not standardize numeric fields. Histogram gradient
boosting uses a compact ten-column representation with ordinal-encoded categorical values, avoiding a dense
five-million-row one-hot matrix.

All imputers, scalers, and encoders are inside scikit-learn pipelines and are fitted only on training data.

## Tracked evidence

Every run records:

- dataset path, size, modification time, and SHA-256 hash;
- source and experiment row counts;
- feature lists and excluded columns;
- chronological split sizes, fraud rates, and date boundaries;
- complete scalar preprocessing and model parameters;
- random seed, model family, configuration name, model rationale, and comparison scope;
- validation/test PR-AUC, ROC-AUC, accuracy, balanced accuracy, precision, recall, specificity, F1, and
  confusion-matrix counts;
- validation-selected decision threshold;
- training and inference time;
- precision-recall, ROC, confusion-matrix, and threshold-trade-off figures;
- coefficients or feature importances when supported;
- Python, operating-system, and package-version metadata;
- source-code snapshots, model input example, input/output signature, and each fitted candidate model.

Accuracy is logged for completeness but is not used for model selection because fraud represents only about
3.59% of observations. PR-AUC is the primary selection metric.

## Commands

Run a small end-to-end smoke comparison:

```bash
python "Code snippets/05_candidate_model_comparison_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

Run the core four-model comparison on the full dataset:

```bash
python "Code snippets/05_candidate_model_comparison_with_mlflow.py"
```

Run the optional Decision Tree check:

```bash
python "Code snippets/05_candidate_model_comparison_with_mlflow.py" --models decision_tree_depth_10
```

Run the sampled k-NN feasibility experiment:

```bash
python "Code snippets/05_candidate_model_comparison_with_mlflow.py" --models knn_neighbors_31 --knn-training-rows 50000 --knn-evaluation-rows 25000
```

k-NN runs are tagged `sampled_feasibility` and must not be presented as directly comparable full-data
results.

## Experiment boundary

This file compares candidate models using original cleaned predictors only. Log amount, timestamp-derived
features, and the missingness indicator remain later named feature experiments. A model change and a
feature-set change should not be introduced in the same run when the purpose is causal attribution of an
improvement.
