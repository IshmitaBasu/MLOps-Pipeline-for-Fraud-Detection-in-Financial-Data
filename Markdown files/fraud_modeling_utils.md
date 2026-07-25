# Fraud modeling utilities

This note documents `fraud_modeling_utils.py`.

## Purpose

This file contains the shared helper code used by the tracked fraud-model experiments. It keeps data
loading, chronological splitting, MLflow setup, metric calculation, evaluation plots, reproducibility
metadata, and model logging in one place so the baseline and candidate-model scripts use the same experiment
rules.

## Current structure

The file is now organized as a cell-wise Python script using `# %%` markers. These markers make the file
easier to inspect and run section by section in VS Code, Spyder, or notebook-like IDE views while keeping it
valid as a normal `.py` module.

The main cells are:

- imports;
- project paths and feature schema;
- shared result container;
- data loading and validation helpers;
- MLflow setup and reproducibility metadata;
- scoring and metric helpers;
- evaluation artifact helpers;
- common experiment logging;
- fit, evaluate, and log one MLflow run;
- local summary writer.

## Important behavior

The utilities enforce a leakage-aware workflow. The gold table is loaded with only the required columns,
timestamps are parsed, optional smoke samples are stratified by target, and train/validation/test splits are
created in chronological order.

Model evaluation uses validation data for threshold tuning and logs test metrics only when a caller passes a
test split. The shared MLflow function records metrics, plots, source files, environment metadata, feature
metadata, split summaries, and fitted models where requested.

## Relationship to experiment scripts

`04_baseline_modeling_with_mlflow.py` uses these utilities to run the dummy and logistic baselines.

`05_candidate_model_comparison_with_mlflow.py` uses the same utilities so all candidate models share consistent
splits, metrics, metadata, and artifact logging.
