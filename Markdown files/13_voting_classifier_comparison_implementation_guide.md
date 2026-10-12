# Stage 13: Voting Classifier Implementation Guide

## Files

- `13_voting_classifier_comparison_with_mlflow.py`: executable runner with `# %%` cells.
- `voting_classifier_utils.py`: fixed candidates, sklearn VotingClassifier construction, handoff validation, threshold tables, and progression decision.
- `validation_evaluation_utils.py`: shared metrics used by model comparisons and tuning.
- `tests/test_voting_classifier.py`: actual estimator compatibility, score averaging, threshold evidence, and selection safeguards.

The rationale and rules are in the plan. Completed evidence belongs in the results log.

## Behaviour

The runner downloads the full Stage 12 summary from run `cceb57941a914ba2b3e015573b3e34b1`, validates the retained LightGBM reference, and recreates the frozen development frames. It fits the reference and two actual sklearn `VotingClassifier` candidates, each with complete preprocessing pipelines for its members.

VotingClassifier clones and fits its members. Each ensemble therefore refits its constituent models; no fitted historical model is silently reused. Members fit sequentially (`n_jobs=1` at ensemble level) to limit simultaneous memory use. Each underlying model keeps the established threading settings. The three-member full run includes Random Forest and can take considerably longer than the LightGBM-only tuning candidates.

## Run from the thesis workspace

Smoke check:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\13_voting_classifier_comparison_with_mlflow.py" --mode smoke
```

After checking smoke output, run the full validation comparison:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\13_voting_classifier_comparison_with_mlflow.py" --mode full
```

The default data source is Feast. The optional `--data-source csv` path uses the same history-building controls. The full split must match the recorded Stage 12 hash. No new dependencies are required beyond the existing pinned environment.

## MLflow artifacts

Experiment: `financial-fraud-stage-13-voting-classifier-comparison`.

Every candidate saves:

- `candidate_definition.json`: members, parameters, equal weights, and calibration status;
- `voting_result.json`: complete validation scalar metrics, including the maximum-F1 threshold and confusion counts;
- `validation_thresholds.csv`: fixed-score and maximum-F1 diagnostic rows;
- `validation_confusion_matrices.csv`: confusion counts for each threshold rule;
- `validation_workloads.csv`: 1%, 5%, and 10% review-volume results and missed fraud value;
- split, data lineage, environment metadata, and source snapshots.

The summary run saves `voting_smoke_test_summary.json` or `voting_full_data_summary.json`, and `voting_comparison_results.csv`. Failed fits are recorded as failed candidate runs; they do not create a complete selection decision.

## Expected output

All three candidates print Average Precision, F1, and their run IDs. Smoke mode ends without model selection. Full mode prints either the retained LightGBM reference or a qualifying voting candidate. Both modes report that the test split was not evaluated and the operating rule is still unfrozen.

The next step after full execution is to record the model comparison and inspect threshold trade-offs before fixing the final decision rule.

## Hard-voting follow-up added on 8 October 2026

Use the existing cell-based runner; no duplicate experiment script or new dependency is required. Omitting `--voting` preserves the original soft workflow.

From the thesis root, check the hard workflow first:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\13_voting_classifier_comparison_with_mlflow.py" --voting hard --mode smoke
```

After checking smoke output, run the full validation-only follow-up:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\13_voting_classifier_comparison_with_mlflow.py" --voting hard --mode full
```

The default experiment is `financial-fraud-stage-13-hard-voting-followup`. The runner fits native LightGBM, a hard LightGBM–CatBoost vote, and a hard LightGBM–CatBoost–Random Forest vote. The two-member tie resolves to legitimate; the three-member decision requires two fraud votes. Member thresholds are not tuned. Full mode enforces the same five-million-row input and frozen split hash as the original workflow. It reconstructs development history using training events only and never builds or scores a test frame.

Candidate artifacts are `hard_voting_result.json`, `validation_classification_policies.csv`, and `paired_soft_validation_thresholds.csv`, plus source snapshots, versions, split and lineage records. The classification CSV includes native hard decisions, soft argmax decisions at 0.5 (a probability tie resolves to legitimate), and the explicitly labelled soft validation maximum-F1 rule from the same fitted members. Soft threshold diagnostics are not hard-voting scores.

The summary saves `hard_voting_smoke_test_summary.json` or `hard_voting_full_data_summary.json`, and `hard_soft_classification_comparison.csv`. Candidate MLflow metrics describe hard/native classifications only; paired soft metrics remain separately labelled in artifacts. Hard voting does not save fictitious ranking or top-k metrics.

Both modes are diagnostic only and mark model selection ineligible. They do not modify the completed final test receipt, registered-model state, or Stage 15 serving bundle. Full execution may take time, particularly for the three-member Random Forest ensemble. Record run IDs and the full-data findings in the results document after execution; do not report smoke rankings as thesis conclusions.
