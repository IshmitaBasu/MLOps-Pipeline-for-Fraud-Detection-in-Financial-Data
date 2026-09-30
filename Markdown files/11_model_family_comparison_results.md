# Model-Family Comparison Results

## Status

The experiment design, fixed configurations and automated safeguards are ready.
The full Stage 10 handoff is recorded. No Stage 11 model has yet been trained,
so no model-family conclusion has been made.

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

## Inference

LightGBM is the selected Stage 11 model family and may proceed to a small,
predeclared validation-only tuning experiment. This selection is still not a
final deployment claim: the absolute predictive improvement over Random Forest
is small, and the held-out test partition remains unopened.
