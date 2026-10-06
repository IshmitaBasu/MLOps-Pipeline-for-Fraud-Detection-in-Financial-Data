# Stage 13: Voting Classifier Comparison Plan

## Question

Does averaging predictions from complementary models improve fraud ranking and capture compared with the retained LightGBM model?

Individual models may make different errors. Soft voting averages their scores and might reduce some of those errors. Similar mistakes or incompatible score scales can also make an ensemble worse, so the benefit must be measured.

## Fixed candidates

| Candidate | Members | Weighting |
| --- | --- | --- |
| `lgbm_reference` | Retained Stage 12 LightGBM | Single model |
| `soft_vote_lgbm_catboost` | LightGBM and CatBoost | Equal weights |
| `soft_vote_lgbm_catboost_rf` | LightGBM, CatBoost, and Random Forest | Equal weights |

LightGBM and CatBoost were strong candidates in Stage 11. Random Forest provides a different tree-ensemble method and acts as a check on whether adding a bagging model helps the two boosting models. The two-member candidate also shows whether adding Random Forest is worth the extra fitting and scoring cost.

Model settings come from the completed Stage 11 configurations and the retained Stage 12 LightGBM reference. There is no weight search or additional hyperparameter search.

## Example

If LightGBM, CatBoost, and Random Forest return scores of 0.2, 0.4, and 0.3, their equal-weight soft vote is 0.3. Transactions can then be ranked by this average score or flagged using a score threshold.

Soft voting uses `predict_proba` outputs. The class-weighted models are not calibrated here, so those outputs and their average are treated as ranking scores, not verified real-world fraud probabilities. Calibration, if later required, needs a separately designed training-only procedure. The scikit-learn [VotingClassifier documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.VotingClassifier.html) explains probability averaging and recommends calibrated classifiers for soft voting.

## Controls

The experiment retains the frozen stratified random split, behavioural feature set, class weighting, and random state 42. Each member's preprocessing is fitted only on training rows. Validation rows keep their natural class balance. Test rows are counted in the manifest but not materialised or evaluated.

Smoke mode uses 50,000 rows and cannot select a candidate. Full mode requires five million rows, the frozen split hash, and the full Stage 12 handoff.

## Decision rule

An ensemble may replace LightGBM only if validation Average Precision increases, neither fraud-count nor fraud-value recall at the fixed 5% review workload decreases, and at least one recall measure increases. This reuses the established Stage 12 progression rule. Near ties within 0.0001 AP are resolved by fraud-value recall, then inference time. Otherwise, retain the LightGBM reference.

The 5% workload is a comparison scenario, not a confirmed institutional capacity. Candidate selection does not establish that the absolute performance is acceptable.

## Threshold diagnostics

Each candidate is assessed at score thresholds 0.1 through 0.9 in steps of 0.1 and at its validation maximum-F1 threshold. The report includes precision, recall, F1, accuracy, balanced accuracy, specificity, confusion counts, review volume, captured fraud value, and missed fraud value. Separate 1%, 5%, and 10% workload tables assess ranking at equal review volumes.

Threshold changes affect classifications but not the Average Precision of an unchanged score ranking. Missed fraud value is an exposure measure, not an established institutional loss estimate. No review cost or recovery assumptions are invented.

These tables support later operating-rule selection. A final operating rule is not automatically selected from this diagnostic, and no test evaluation occurs in Stage 13.
