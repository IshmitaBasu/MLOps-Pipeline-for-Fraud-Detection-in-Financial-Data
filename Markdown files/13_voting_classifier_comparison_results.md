# Stage 13: Voting Classifier Comparison Results

## Status

Both smoke and full-data comparisons are complete. The full-data decision retained `lgbm_reference`; neither voting ensemble passed the progression rule. During Stage 13 the operating rule remained unfrozen and the revised test partition was unevaluated. The subsequent frozen protocol and final test results are recorded in [Stage 14 results](14_final_test_evaluation_results.md).

## Starting evidence

- Stage 12 full-data summary: `cceb57941a914ba2b3e015573b3e34b1`.
- Retained configuration: `lgbm_reference`.
- Full validation reference AP: 0.046721.
- Class-weight treatment and behavioural features remain fixed.
- The revised held-out test partition remains unevaluated.

## Implementation verification

All 76 repository tests passed on 5 October 2026, including seven Stage 13 tests. The voting compatibility test fitted both actual sklearn ensembles on small synthetic training data and verified that their predictions equal the mean member probabilities. Other checks cover confusion counts and missed value at known thresholds, malformed inputs, the Stage 12 handoff, smoke ineligibility, and the progression rule.

The CLI imports successfully and displays its supported modes. These checks establish correct implementation; they do not establish predictive improvement. The smoke and full-data experiments subsequently completed, as recorded below.

## Smoke experiment

Recorded on 5 October 2026. The 50,000-row sample used 35,000 training rows, 7,500 validation rows, and 7,500 untouched test rows. The validation partition contained 270 fraudulent and 7,230 legitimate transactions. The source features and class weighting match the Stage 12 handoff.

| Candidate | Average Precision | ROC-AUC | Maximum F1 | Training time | Validation scoring time | MLflow run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lgbm_reference` | 0.044519 | 0.599768 | 0.088144 | 2.03 s | 0.30 s | `6593eff7df244bc4a0124fcd9d728552` |
| `soft_vote_lgbm_catboost` | 0.045562 | 0.603088 | 0.087614 | 5.12 s | 0.33 s | `ace051ab3bc14701b7ad5cb3f5bd4db9` |
| `soft_vote_lgbm_catboost_rf` | 0.045547 | 0.599096 | 0.087622 | 10.31 s | 0.44 s | `c408c40884fb4143aa72e8d49e18aa0f` |

Summary run: `fc6083646ea34c8997474c38f11f91fa`.

Source: `voting_smoke_test_summary.json` and the candidate threshold/workload artifacts. Unlike the historical Stage 11/12 runs, Stage 13 directly saved the thresholds and confusion counts; the following classification values do not require reconstruction.

### Maximum-F1 classification results

Each candidate uses its own validation-selected threshold. Different review volumes mean these rows should not be treated as an equal-workload comparison.

| Candidate | Threshold | Precision | Recall | Accuracy | Balanced accuracy | Alert share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.211692 | 4.6351% | 89.6296% | 33.2400% | 60.3819% | 69.6133% |
| LightGBM + CatBoost | 0.358724 | 4.6605% | 72.9630% | 45.2933% | 58.6115% | 56.3600% |
| LightGBM + CatBoost + Random Forest | 0.395878 | 4.6736% | 70.0000% | 47.5200% | 58.3402% | 53.9200% |

Confusion counts at these same thresholds, with fraud as the positive class:

| Candidate | TN | FP | FN | TP | Missed fraud value |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 2,251 | 4,979 | 28 | 242 | 11,472.61 |
| LightGBM + CatBoost | 3,200 | 4,030 | 73 | 197 | 29,665.62 |
| LightGBM + CatBoost + Random Forest | 3,375 | 3,855 | 81 | 189 | 32,302.08 |

Missed fraud value is the sum of transaction amounts for false negatives, in the dataset's amount units; it is not a measured institutional loss. All three maximum-F1 rules flag more than half the validation transactions.

### Equal-workload comparison at 5%

All candidates review 375 transactions in this scenario.

| Candidate | Fraud cases found | False alerts | Precision | Fraud-count recall | Fraud-value recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 17 | 358 | 4.5333% | 6.2963% | 6.4766% |
| LightGBM + CatBoost | 15 | 360 | 4.0000% | 5.5556% | 2.4394% |
| LightGBM + CatBoost + Random Forest | 18 | 357 | 4.8000% | 6.6667% | 2.6609% |

Both ensembles increased smoke AP by about 2.3% relative to LightGBM but slightly reduced maximum F1. The two-model ensemble found fewer fraud cases at the 5% workload; the three-model ensemble found one more case. Both captured substantially less fraudulent transaction value in this small sample. The measures therefore give different rankings, and higher AP alone does not establish an operational improvement.

The small validation partition contains only 270 fraud cases, so individual transactions can strongly affect value recall. These observations support checking the full comparison, not choosing a model from smoke data. The summary correctly records `decision_eligible: false`, no selected configuration, and `operating_rule_frozen: false`. The held-out test split was not materialised or evaluated.

## Full-data experiment

The full comparison used 3,500,000 training rows and 750,000 validation rows. Validation contained 26,933 fraudulent and 723,067 legitimate transactions. The frozen split hash matched the Stage 12 handoff. The remaining 750,000 test rows were not materialised or evaluated.

| Candidate | Average Precision | ROC-AUC | Maximum F1 | Training time | Validation scoring time | MLflow run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lgbm_reference` | 0.046721 | 0.612986 | 0.086266 | 125.82 s | 14.36 s | `cb06aff922d544d4a34651550b147e42` |
| `soft_vote_lgbm_catboost` | 0.046796 | 0.613904 | 0.086392 | 212.33 s | 17.14 s | `03d444f5219842e19c55f53a7cc62765` |
| `soft_vote_lgbm_catboost_rf` | 0.046794 | 0.613936 | 0.086392 | 1,373.25 s | 28.22 s | `744253b0ef414c35bafc12559003f918` |

Summary run: `5920c17849574c11b4854c3c5290bc41`. Source artifact: `voting_full_data_summary.json`, with threshold and workload CSVs in the three candidate runs.

### Classification at each candidate's maximum-F1 threshold

| Candidate | Threshold | Precision | Recall | Accuracy | Balanced accuracy | Alert share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.522602 | 4.5345% | 88.4157% | 32.7395% | 59.5407% | 70.0196% |
| LightGBM + CatBoost | 0.521220 | 4.5381% | 89.7226% | 31.8540% | 59.7106% | 70.9989% |
| LightGBM + CatBoost + Random Forest | 0.526465 | 4.5537% | 84.0419% | 36.1688% | 59.2137% | 66.2761% |

| Candidate | TN | FP | FN | TP | Missed fraud value |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 221,733 | 501,334 | 3,120 | 23,813 | 1,189,022.62 |
| LightGBM + CatBoost | 214,740 | 508,327 | 2,768 | 24,165 | 1,004,237.98 |
| LightGBM + CatBoost + Random Forest | 248,631 | 474,436 | 4,298 | 22,635 | 1,608,765.16 |

The two-member vote catches more fraud at its maximum-F1 threshold, but also generates 6,993 more false positives and reviews more transactions than LightGBM. The three-member vote reduces false positives but misses more fraud cases and more fraud value. F1 increases only slightly for both ensembles. Since each row uses a different alert volume, these improvements and reductions cannot replace the equal-workload comparison.

### Equal review workloads

| Review share | Candidate | Fraud cases found | False alerts | Precision | Fraud-count recall | Fraud-value recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1% | LightGBM | 375 | 7,125 | 5.0000% | 1.3923% | 1.4538% |
| 1% | LightGBM + CatBoost | 382 | 7,118 | 5.0933% | 1.4183% | 1.1804% |
| 1% | LightGBM + CatBoost + Random Forest | 389 | 7,111 | 5.1867% | 1.4443% | 1.1734% |
| 5% | LightGBM | 1,824 | 35,676 | 4.8640% | 6.7724% | 7.1674% |
| 5% | LightGBM + CatBoost | 1,807 | 35,693 | 4.8187% | 6.7092% | 6.8158% |
| 5% | LightGBM + CatBoost + Random Forest | 1,806 | 35,694 | 4.8160% | 6.7055% | 6.4372% |
| 10% | LightGBM | 3,628 | 71,372 | 4.8373% | 13.4705% | 14.2316% |
| 10% | LightGBM + CatBoost | 3,564 | 71,436 | 4.7520% | 13.2328% | 13.7244% |
| 10% | LightGBM + CatBoost + Random Forest | 3,570 | 71,430 | 4.7600% | 13.2551% | 13.4568% |

At the predeclared 5% comparison workload, both ensembles find fewer fraud cases and capture less fraud value than LightGBM. They also have worse count and value capture at 10%. At 1%, they find slightly more cases but still less fraudulent value. No tested ensemble provides a consistent operational improvement across these scenarios.

### Progression decision and inference

Selected candidate: `lgbm_reference`. Recommendation: `retain_lightgbm_reference`. Both ensemble progression checks are false; the operating rule remains unfrozen.

The two-member ensemble increases AP by approximately 0.159% relative to the reference, and the three-member ensemble by approximately 0.156%. These small numerical increases do not satisfy the rule because both fraud-count and fraud-value recall fall at the fixed 5% workload. There is no statistical uncertainty assessment establishing that the small AP changes are reliable improvements.

The three-member ensemble also takes approximately 10.9 times as long to train and 2.0 times as long to score validation rows as LightGBM alone. Its added complexity is not supported by the progression evidence. These measured runtimes refer to this local experiment, not a deployment latency benchmark.

The voting experiment is a completed, informative negative result: averaging these particular model scores did not improve the full comparison criteria. This does not establish that every possible ensemble would fail. It also does not establish that the retained LightGBM has satisfactory absolute fraud-detection performance.

## Threshold interpretation

Threshold CSVs and confusion-matrix CSVs were saved for all three smoke candidates. For example, LightGBM at score threshold 0.5 flagged 1,126 rows and reached 17.0370% recall with F1 0.065903. Its maximum-F1 threshold of approximately 0.211692 flagged 5,221 rows and reached 89.6296% recall with F1 0.088144. This illustrates the trade-off: finding more fraud requires many more reviews, while precision remains low.

These smoke thresholds are technical examples, not final decision rules. Changing a threshold does not change AP when the score ranking stays unchanged.

### Full-data LightGBM threshold report

The retained reference now has a directly recorded maximum-F1 threshold of `0.5226022799524157`. Its AP, precision, recall, F1, and confusion counts reproduce the Stage 12 reference results. This newly executed Stage 13 report supplies the numerical threshold missing from the original historical run; the original artifacts remain unchanged.

| Rule / threshold | Alerts | Precision | Recall | F1 | Accuracy | Missed fraud value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 615,321 | 4.3764% | 99.9851% | 0.083858 | 21.5472% | 469.09 |
| 0.2 | 614,953 | 4.3782% | 99.9666% | 0.083890 | 21.5949% | 1,148.47 |
| 0.3 | 611,034 | 4.3962% | 99.7364% | 0.084211 | 22.1009% | 27,717.64 |
| 0.4 | 603,619 | 4.4205% | 99.0718% | 0.084634 | 23.0419% | 103,698.76 |
| 0.5 | 561,522 | 4.4898% | 93.6064% | 0.085685 | 28.2623% | 638,980.97 |
| Validation maximum F1: 0.522602 | 525,147 | 4.5345% | 88.4157% | 0.086266 | 32.7395% | 1,189,022.62 |
| 0.6 | 3,231 | 5.3234% | 0.6386% | 0.011404 | 96.0240% | 9,627,404.59 |
| 0.7 | 0 | 0.0000% | 0.0000% | 0.000000 | 96.4089% | 9,683,820.29 |
| 0.8 | 0 | 0.0000% | 0.0000% | 0.000000 | 96.4089% | 9,683,820.29 |
| 0.9 | 0 | 0.0000% | 0.0000% | 0.000000 | 96.4089% | 9,683,820.29 |

At zero alerts, precision is undefined mathematically and is reported as zero by the metric function. High accuracy at thresholds 0.7–0.9 comes entirely from predicting every transaction as legitimate; it detects no fraud. At maximum F1, about 70% of transactions are flagged. Thus none of these facts demonstrates a practical detector.

The large change in alert volume between thresholds 0.5 and 0.6 shows why a coarse score grid cannot by itself establish a suitable operational threshold. If a restricted review volume is required, finer validation threshold analysis near this region, or a clearly documented batch ranking rule, is needed before freezing the operating policy. A top-5% batch rule is not interchangeable with a fixed score threshold.

Full threshold and confusion CSVs are available in each candidate run. All amounts above are in dataset units and describe fraud exposure rather than an assumed institutional loss.

## Next action

The subsequent Stage 14 evaluation retained these LightGBM settings, froze validation maximum F1 as the primary research benchmark, and reported the three predeclared batch workloads separately. Final test evaluation is complete. See the Stage 14 results for measured test performance, limitations, and the next prototype steps. No further Stage 13 model selection should use the completed test evidence.
