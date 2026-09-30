# Model-Family Comparison Implementation Guide

## Current state

The Stage 11 runner, experiment rules and automated safeguards are implemented.
The full Stage 10 comparison retained class weighting, which is now a verified
and fixed input to Stage 11. No Stage 11 model experiment has been run yet.

## Why this is separate from Stage 06

Stage 11 uses a separate implementation because its features, imbalance
treatment and operational evaluation differ from Stage 06. The complete
rationale and model-family choices are recorded once in
`11_model_family_comparison_plan.md`.

## What is already fixed

`model_family_comparison_utils.py` freezes:

- the Random Forest reference and the starting configurations for Histogram
  Gradient Boosting, LightGBM, XGBoost, CatBoost and a scalable linear SVM;
- a separate sampled KNN feasibility configuration;
- the accepted Stage 10 strategy names;
- the frozen split hash;
- deterministic KNN sampling; and
- the validation-only model progression rule.

This separation prevents the comparison rules from being adjusted after the
model scores are visible.

## Stage 10 handoff

The completed Stage 10 summary must contain full-data evidence, a valid
selection decision, the frozen split hash and its MLflow summary run ID. Smoke
evidence is rejected. Any field beginning with `test_` is also rejected.

In simple terms, Stage 11 first checks that it is receiving the real Stage 10
decision and not a convenient result copied from a small trial.

The frozen handoff is:

- summary run: `7d1e57bd9c0a49a59e0898bb3c7bad21`;
- strategy: `class_weight_reference`;
- split hash: `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242`;
- feasibility exclusions: none; and
- test evaluated: no.

## Full-data candidates

The candidate names, reasons and parameters are defined in the experiment plan
and frozen in `model_family_comparison_utils.py`. This guide does not repeat
that list. The runner must use those definitions directly rather than maintain
a second configuration.

## Selection logic

A full-data model can replace Random Forest only when it is feasible, has
higher validation Average Precision, does not reduce fraud-count recall or
fraud-value recall at the fixed 5% workload, and improves at least one of those
two recall measures.

Differences in Average Precision below `0.0001` are treated as a tie. The first
tie-breaker is fraud-value recall at 5%; the second is validation inference
time. Smoke results can never select a model.

## Automated checks

Run only the Stage 11 safeguard tests with:

```powershell
.\masters_thesis\Scripts\python.exe -m unittest ".\Code snippets\tests\test_model_family_comparison.py" -v
```

The checks confirm that:

- the candidate list and starting parameters are frozen;
- only a complete full-data Stage 10 decision is accepted;
- test evidence cannot enter model selection;
- the KNN sample is deterministic and stratified;
- sampled KNN cannot win the full-data comparison; and
- the operational progression and tie-breaking rules behave as documented.

## Run the smoke workflow

Run this first:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\11_model_family_comparison_with_mlflow.py" --mode smoke
```

This uses 35,000 training rows, 7,500 validation rows and an untouched 7,500-row
test allocation. It checks that all model libraries, pipelines, metrics and
MLflow artifacts work together. Its scores cannot select a model.

## Run the full comparison

Run this only after the smoke workflow passes:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\11_model_family_comparison_with_mlflow.py" --mode full
```

The full run fits all eligible model families on the complete training
partition. KNN receives only its frozen stratified feasibility sample. The
runner reads the Stage 10 summary directly from MLflow and stops if its run ID,
decision or split does not match.

## After execution

Record the smoke output first. After the full run, record every model result,
MLflow run ID and the progression decision in the Stage 11 results document.

The held-out test partition stays closed throughout this stage.
