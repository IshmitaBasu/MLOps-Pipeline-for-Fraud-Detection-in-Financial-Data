# Model-Family Comparison Results

## Status

The smoke and full-data model-family comparisons are complete. LightGBM was
selected and subsequently retained during Stage 12 tuning. The revised test
partition remains unevaluated. Additional classification metrics were
documented on 5 October 2026.

## Automated test result

Twelve Stage 11 safeguard tests passed on 30 September 2026. These tests cover
the candidate configuration, Stage 10 handoff, deterministic KNN sampling,
test-isolation rule, construction of all pinned model pipelines and
model-selection logic.

The complete repository test suite also passed: 60 tests run, 60 passed.

This result verifies the experiment controls only. It is not predictive-quality
evidence.

The definitions behind these checks remain in the plan and implementation
guide rather than being repeated in this results log.

## Stage 10 handoff

- MLflow summary run ID: `7d1e57bd9c0a49a59e0898bb3c7bad21`
- selected imbalance strategy: `class_weight_reference`
- split hash: `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242`
- feasibility exclusions: none
- held-out test evaluated: no

Class weighting will therefore remain fixed across the full-data model-family
comparison. Each model API will use its documented equivalent training weight;
training rows will not be resampled.

## Smoke comparison

Status: passed.

| Model | Average Precision | ROC-AUC | 5% fraud-count recall | 5% fraud-value recall | Training time | Inference time | MLflow run ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random Forest reference | 0.045428 | 0.578554 | 5.9259% | 2.3391% | 4.77 s | 0.18 s | `de7eea5077e64ed5b10a6e2724359291` |
| Histogram Gradient Boosting | 0.046750 | 0.599672 | 5.9259% | 11.1077% | 2.32 s | 0.32 s | `7a55bdd161e040288c3a7fd3adb40553` |
| LightGBM | 0.044519 | 0.599768 | 6.2963% | 6.4766% | 1.68 s | 0.27 s | `ddfe4c997b6044219d818fffd9eb6ba4` |
| XGBoost | 0.047634 | 0.598155 | 4.8148% | 4.7737% | 5.75 s | 0.15 s | `ae0be817dcba43c48b73080660cafccd` |
| CatBoost | 0.047170 | 0.605223 | 7.4074% | 6.4200% | 3.55 s | 0.06 s | `f21ed82ac44a43bc8eb6b113c1f88967` |
| Linear SVM | 0.037491 | 0.504483 | 5.5556% | 13.2102% | 1.13 s | 0.04 s | `dcd77ecee22e43b099dfe283dae1a107` |
| KNN sampled feasibility | 0.038919 | 0.509179 | 5.5556% | 7.5523% | 0.39 s | 2.39 s | `4a78754e81a2482fad1360151db83225` |

| Summary item | Value |
| --- | --- |
| Summary run ID | `e55b1689ad414957b4217ecbdde293d5` |
| Training rows | 35,000 |
| Validation rows | 7,500 |
| Held-out test rows | 7,500 |
| Stage 10 treatment | `class_weight_reference` |
| Feasibility exclusions | None |
| Model selected | None; smoke evidence is ineligible |
| Test evaluated | No |

All model pipelines, dependencies, metrics and MLflow logging paths completed.
XGBoost produced the highest smoke Average Precision, while CatBoost captured
the most fraud cases at the fixed 5% workload among the full-data candidates.
Histogram Gradient Boosting and the linear SVM happened to capture a high share
of fraudulent transaction value, but the smoke validation partition contains
only 270 fraud rows, so individual high-value cases can move this metric
substantially.

The linear SVM and sampled KNN were close to random ranking by ROC-AUC and had
the two lowest Average Precision values. KNN also had the slowest validation
inference despite being evaluated on the small smoke population. These are
useful feasibility observations, not final findings.

The ranking differs by metric and cannot be used to select a model. The smoke
run establishes only that the complete full-data comparison is technically
ready.

## Full-data validation comparison

Status: complete.

| Model | Average Precision | ROC-AUC | 5% fraud-count recall | 5% fraud-value recall | Training time | Inference time | MLflow run ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random Forest reference | 0.046448 | 0.612248 | 6.6461% | 6.8003% | 1,158.1 s | 12.51 s | `b26b3edddd2d441da1dd6330d24ca128` |
| Histogram Gradient Boosting | 0.046328 | 0.612409 | 6.4196% | 6.2673% | 127.8 s | 15.87 s | `37edbb5c7a8147cf983506f054476344` |
| LightGBM | 0.046721 | 0.612986 | 6.7724% | 7.1674% | 131.9 s | 18.41 s | `4ad97f5df151448480277b8270130c9c` |
| XGBoost | 0.046602 | 0.611928 | 6.6535% | 7.1540% | 303.5 s | 9.73 s | `7fd601e882584d9cabe2bb02765941ea` |
| CatBoost | 0.046778 | 0.614162 | 6.7501% | 6.8604% | 121.1 s | 3.40 s | `09b6e9b05b714ddfad20c4a8783ccf03` |
| Linear SVM | 0.038410 | 0.525317 | 5.5842% | 8.2221% | 44.0 s | 2.74 s | `a2125e6ab48d4f13aa7e193233e1462f` |
| KNN sampled feasibility | 0.035237 | 0.496662 | 4.4543% | 4.2289% | 0.76 s | 16.85 s | `f5c34ce83d504aeb85437cf5fc75d5aa` |

| Summary item | Value |
| --- | --- |
| Summary run ID | `ba890331d5a149f0bb17ee1a7f63c589` |
| Training rows for full candidates | 3,500,000 |
| Validation rows for full candidates | 750,000 |
| KNN feasibility sample | 100,000 training and 25,000 validation rows |
| Held-out test rows | 750,000 |
| Feasibility exclusions | None |
| Selected model | `lightgbm` |
| Progression decision | `progress_selected_model_to_limited_tuning` |
| Test evaluated | No |

LightGBM, XGBoost and CatBoost all passed the progression rule against Random
Forest. Each had higher Average Precision, did not reduce fraud-count or
fraud-value recall at the fixed 5% workload, and improved at least one of those
recall measures.

CatBoost had the numerically highest Average Precision (`0.046778`), but its
lead over LightGBM was only `0.000056`, below the predeclared `0.0001` tie
tolerance. The tie was therefore resolved using 5% fraud-value recall, where
LightGBM captured `7.1674%` compared with CatBoost's `6.8604%`. LightGBM was
selected according to the frozen rule rather than chosen after inspecting the
results.

XGBoost also improved over Random Forest, but its Average Precision was below
LightGBM by about `0.000120`, which is outside the tie tolerance. Histogram
Gradient Boosting did not improve the reference ranking or operational recall.
The linear SVM captured more fraudulent value but had substantially weaker
ranking and fraud-count recall, so it did not progress.

The sampled KNN result was close to random ranking and required `16.85` seconds
to score only 25,000 validation rows. This supports its planned exclusion from
the full-data candidate set.

## Classification metrics at each model's maximum-F1 rule

The six full-data candidates below were evaluated on the same 750,000 validation rows: 26,933 fraudulent and 723,067 legitimate transactions. Each model uses the score threshold that maximised its own validation F1, rather than a common review volume. These values supplement the equal-workload metrics used for model selection.

| Model | Precision | Recall | F1 | Accuracy | Balanced accuracy | Alert share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest reference | 4.5476% | 85.5159% | 0.086359 | 35.0215% | 59.3283% | 67.5293% |
| Histogram Gradient Boosting | 4.5704% | 76.6198% | 0.086262 | 41.7093% | 58.5144% | 60.2025% |
| LightGBM | 4.5345% | 88.4157% | 0.086266 | 32.7395% | 59.5407% | 70.0196% |
| XGBoost | 4.5324% | 87.6843% | 0.086192 | 33.2329% | 59.4445% | 69.4736% |
| CatBoost | 4.5594% | 85.4602% | 0.086570 | 35.2369% | 59.4132% | 67.3099% |
| Linear SVM | 3.8545% | 60.9438% | 0.072505 | 44.0080% | 52.1605% | 56.7780% |

KNN is excluded from this table because it used a different validation population for feasibility assessment. Its sampled ranking metrics remain in the earlier table.

Confusion counts at those same maximum-F1 rules:

| Model | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| Random Forest reference | 239,629 | 483,438 | 3,901 | 23,032 |
| Histogram Gradient Boosting | 292,184 | 430,883 | 6,297 | 20,636 |
| LightGBM | 221,733 | 501,334 | 3,120 | 23,813 |
| XGBoost | 225,631 | 497,436 | 3,317 | 23,616 |
| CatBoost | 241,260 | 481,807 | 3,916 | 23,017 |
| Linear SVM | 313,646 | 409,421 | 10,519 | 16,414 |

Source: the full-data summary artifact `model_family_full_data_summary.json` in run `ba890331d5a149f0bb17ee1a7f63c589`, with class totals from the frozen split manifest. Precision, recall, and F1 were saved directly. Accuracy, balanced accuracy, alert share, and confusion counts were reconstructed from those full-precision metrics and class totals. TP = round(recall × fraud rows), predicted fraud = round(TP / precision), FP = predicted fraud − TP, FN = fraud rows − TP, and TN = legitimate rows − FP. These counts reproduce the saved metrics. The numerical maximum-F1 thresholds were not saved in the original runs; they cannot be recovered from the scalar metrics alone.

All models have low precision. Their high recalls at maximum F1 depend on reviewing more than half of all transactions. For comparison, classifying every transaction as legitimate would produce 96.4089% accuracy and zero fraud recall. Neither high accuracy from majority-class prediction nor high recall from excessive alerts establishes an effective fraud detector.

The retained LightGBM's full metric explanation, confusion matrix, and 1%, 5%, and 10% workload tables are consolidated in [Stage 12 results](12_lightgbm_tuning_results.md#complete-validation-metrics-for-the-retained-lightgbm).

## Inference

LightGBM was selected during Stage 11 and retained after the completed Stage 12
tuning comparison. The absolute predictive improvement over Random Forest is
small, and the additional metrics confirm that fraud discrimination remains
weak. Further threshold and voting-ensemble experiments are planned on
validation data before the model and operating rule are frozen for final test
evaluation.
