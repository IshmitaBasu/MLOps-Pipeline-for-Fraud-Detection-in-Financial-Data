# Stage 12: LightGBM Tuning Results

## Experiment status

The smoke and full-data comparisons are complete. The Stage 11 LightGBM reference was retained. The complete validation metrics and their interpretation were added on 5 October 2026; the held-out test partition remains unevaluated.

## Fixed inputs

- Selected model family: LightGBM
- Stage 11 summary run: `ba890331d5a149f0bb17ee1a7f63c589`
- Feature set: original features plus retained behavioural history features
- Imbalance treatment: class weighting
- Split: frozen stratified random split
- Test evaluation: prohibited during this stage

## Implementation checks

Automated checks cover the fixed candidate list, mandatory Stage 11 handoff, split identity, smoke-run ineligibility, operational selection rule, reference fallback, and rejection of test metrics.

The implementation check completed successfully on 1 October 2026:

- all 68 repository tests passed;
- the Python files compiled successfully;
- all installed packages passed the dependency check;
- the saved Stage 11 MLflow summary passed the Stage 12 handoff validator; and
- formatting and Git whitespace checks passed.

The validation calculation shared by Stages 11 and 12 was moved into `validation_evaluation_utils.py`. This removes a repeated fitting-and-metrics block without changing the Stage 11 experiment definition or its recorded results.

## Smoke result

The 50,000-row smoke workflow completed successfully on 1 October 2026. The frozen split contained 35,000 training rows, 7,500 validation rows, and 7,500 untouched test rows.

| Configuration | Validation Average Precision | MLflow run |
| --- | ---: | --- |
| `lgbm_reference` | 0.044519 | `8a73de0ec7104d6cad7ad74f2ba89218` |
| `lgbm_trees_500` | 0.045128 | `27ad6dd404da4cc78567b3417748c82f` |
| `lgbm_slow_500` | 0.047586 | `4e156dc861cc447da5faec98e0d94d94` |
| `lgbm_leaves_15` | 0.048021 | `35f8b78445674c288e51dc7bad8d272c` |
| `lgbm_leaves_63` | 0.051075 | `0fc972d5ef12498db23f63826649545b` |
| `lgbm_min_child_50` | 0.047446 | `32bb46a983da407b86c29b4da3772ba5` |
| `lgbm_min_child_250` | 0.046728 | `4b1846599c4942ad9842a2e32145a181` |
| `lgbm_regularized` | 0.048341 | `8f27360d07f2499386928430a15a6d61` |

Summary run: `3753d0f14b764bbfaa9f4de8eb629caf`

`lgbm_leaves_63` produced the highest smoke Average Precision. Its score was 0.006556 higher than the reference, equivalent to an approximate 14.73% relative increase. Several other adjustments also exceeded the reference score, which confirms that the tuning configurations are executing and producing different model behaviour.

No configuration was selected. The smoke sample is too small for a thesis conclusion, and the decision rule also requires the full-data fraud-count and fraud-value recall values. The held-out test split was not materialised or evaluated.

## Full-data result

The full-data comparison completed on 1 October 2026 using 3,500,000 training rows and 750,000 validation rows. The remaining 750,000 test rows were not materialised or evaluated.

| Configuration | Validation AP | Fraud-count recall at 5% | Fraud-value recall at 5% | MLflow run |
| --- | ---: | ---: | ---: | --- |
| `lgbm_reference` | 0.046721 | 0.067724 | 0.071674 | `7b84d7eb1ffa4063ade15b08e0027db0` |
| `lgbm_trees_500` | 0.046429 | 0.066981 | 0.070511 | `e6e8dc6480364b7fac4a3cee2807c254` |
| `lgbm_slow_500` | 0.046550 | 0.065979 | 0.065128 | `340e9ae9a187439aafb4ab894ae4f273` |
| `lgbm_leaves_15` | 0.046800 | 0.066795 | 0.064799 | `48fa11eaf60a4e4ab70d53503c891f2b` |
| `lgbm_leaves_63` | 0.046262 | 0.064939 | 0.069887 | `74f633847d6c4a468385b0422b161e0e` |
| `lgbm_min_child_50` | 0.046795 | 0.067649 | 0.070414 | `e8131332a93c4c918d232bcc15885f2f` |
| `lgbm_min_child_250` | 0.046648 | 0.066795 | 0.063449 | `8a1643e149764b32b887e31ee1e84eac` |
| `lgbm_regularized` | 0.046577 | 0.066981 | 0.065766 | `f15bdc778d3442aebf0bc343de700e24` |

Summary run: `cceb57941a914ba2b3e015573b3e34b1`

The smoke leader, `lgbm_leaves_63`, did not retain its advantage on the full dataset. This demonstrates why smoke values were treated only as technical evidence.

`lgbm_leaves_15` and `lgbm_min_child_50` produced slightly higher Average Precision than the reference, by approximately 0.17% and 0.16% respectively. Both reduced fraud-count recall and fraud-value recall at the fixed 5% workload, so neither passed the predeclared progression rule. Every other alternative had lower Average Precision and also failed the rule.

The retained configuration is therefore `lgbm_reference`: 300 estimators, learning rate 0.05, 31 leaves, minimum 100 child samples, 0.8 row and feature subsampling, and L2 regularisation of 1.0. This is the unchanged LightGBM configuration selected during the model-family comparison.

The conclusion is not that tuning failed technically. It showed that the tested parameter changes did not produce a safer overall improvement under equal review capacity. Retaining the reference avoids selecting a model from a very small Average Precision increase that would find fewer fraudulent transactions and less fraudulent transaction value.

Stage 12 is complete. Further validation experiments and final operating-rule selection precede the one-time held-out test evaluation.

## Complete validation metrics for the retained LightGBM

These results describe the 750,000-row validation partition, containing 26,933 fraudulent and 723,067 legitimate transactions. They are development results. No final test performance has been measured.

Source run: `7b84d7eb1ffa4063ade15b08e0027db0`. Ranking and maximum-F1 metrics come from `tuning_result.json`; workload results come from `validation_workloads.csv`. The split counts come from `split_manifest.json`.

### Ranking quality

| Metric | Validation value | Meaning |
| --- | ---: | --- |
| Average Precision (AP) | 0.046721 | Precision-recall ranking quality across score thresholds |
| ROC-AUC | 0.612986 | Ability to rank fraud above legitimate transactions |
| Fraud prevalence | 3.5911% | Reference level for an uninformative precision-recall ranking |

The reported AP is scikit-learn Average Precision, rather than a trapezoidal area under the precision-recall curve. AP exceeds the prevalence reference by about 30.1% relatively, but its absolute value is still low. This relative comparison does not establish that the model is operationally suitable.

### Classification at the maximum-F1 rule

The following metrics use the score threshold that maximised F1 on this validation partition. They do not use a fixed 5% review workload. Selecting and measuring a threshold on the same validation data makes these development values; a frozen rule still needs independent test evaluation.

| Metric | Validation value |
| --- | ---: |
| Precision | 4.5345% |
| Recall | 88.4157% |
| F1 | 0.086266 |
| Accuracy | 32.7395% |
| Balanced accuracy | 59.5407% |
| Transactions flagged | 525,147 of 750,000 (70.0196%) |

Confusion matrix: rows are actual labels; columns are predicted labels. Fraud is the positive class.

| Actual / predicted | Predicted legitimate | Predicted fraud |
| --- | ---: | ---: |
| Actual legitimate | 221,733 true negatives | 501,334 false positives |
| Actual fraud | 3,120 false negatives | 23,813 true positives |

Precision means that approximately 4.5 out of every 100 alerts correspond to actual fraud. Recall means that approximately 88 out of every 100 fraud cases are detected, but achieving that recall requires flagging about seven out of every ten transactions. The large false-positive burden explains the low F1 and low accuracy at this rule.

For context, predicting every transaction as legitimate would achieve 96.4089% accuracy while detecting zero fraud cases. Accuracy alone therefore cannot determine whether a fraud detector is useful. Balanced accuracy gives equal weight to recognising fraud and legitimate transactions.

Historical reporting limitation: accuracy, balanced accuracy, and the confusion counts above were reconstructed from the saved full-precision precision and recall values and the frozen class counts. Specifically, TP = round(recall × fraud rows), predicted fraud = round(TP / precision), FP = predicted fraud − TP, FN = fraud rows − TP, and TN = legitimate rows − FP. The reconstructed counts reproduce the saved precision, recall, and F1. The original run did not save its numerical maximum-F1 threshold, and it cannot be recovered from these scalar metrics alone. The model was not retrained to produce this documentation, and historical MLflow artifacts were not changed.

### Classification at fixed review workloads

These rules rank transactions by score and review the highest-scoring 1%, 5%, or 10%. They are batch ranking scenarios, not evidence of an agreed institutional review capacity or a fixed numerical API threshold.

| Review share | Alerts | TP | FP | FN | TN | Precision | Fraud recall | Fraud-value recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 7,500 | 375 | 7,125 | 26,558 | 715,942 | 5.0000% | 1.3923% | 1.4538% |
| 5% | 37,500 | 1,824 | 35,676 | 25,109 | 687,391 | 4.8640% | 6.7724% | 7.1674% |
| 10% | 75,000 | 3,628 | 71,372 | 23,305 | 651,695 | 4.8373% | 13.4705% | 14.2316% |

TP and FP in this table are saved workload counts; FN and TN are obtained by subtracting those counts from the frozen class totals. At a 5% workload, 1,824 fraud cases are found and 25,109 are missed. The model captures only 7.17% of fraudulent transaction value. Restricting the review workload reduces the false-alert volume but also sharply reduces fraud recall.

## Additional classification metrics across tuning configurations

Each row uses that configuration's own validation maximum-F1 rule, so the review volumes differ. This table supplements the fixed-workload comparison used for selection; it does not change the predeclared decision.

| Configuration | ROC-AUC | Precision | Recall | F1 | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_reference` | 0.612986 | 4.5345% | 88.4157% | 0.086266 | 32.7395% | 59.5407% |
| `lgbm_trees_500` | 0.611423 | 4.5289% | 85.4936% | 0.086021 | 34.7589% | 59.1814% |
| `lgbm_slow_500` | 0.612989 | 4.5466% | 86.6001% | 0.086396 | 34.2285% | 59.4389% |
| `lgbm_leaves_15` | 0.613158 | 4.5420% | 85.5493% | 0.086261 | 34.9153% | 59.2893% |
| `lgbm_leaves_63` | 0.610725 | 4.5413% | 82.7906% | 0.086103 | 36.8879% | 58.9843% |
| `lgbm_min_child_50` | 0.613043 | 4.5249% | 88.8947% | 0.086115 | 32.2453% | 59.5150% |
| `lgbm_min_child_250` | 0.613156 | 4.5383% | 86.2993% | 0.086232 | 34.3205% | 59.3419% |
| `lgbm_regularized` | 0.612543 | 4.5286% | 88.4194% | 0.086158 | 32.6443% | 59.4931% |

Accuracy and balanced accuracy in this table were reconstructed using the same method described above. The saved full-data summary is `cceb57941a914ba2b3e015573b3e34b1`.

## Interpretation and remaining work

The current model has weak fraud discrimination. High recall at the maximum-F1 rule comes with an excessive false-positive burden, while restricted review workloads miss most fraud cases. Retaining the reference during tuning establishes which tested configuration meets the comparison rule; it does not establish acceptable predictive performance.

The subsequent validation threshold and voting-ensemble comparison is now complete in [Stage 13 results](13_voting_classifier_comparison_results.md). Neither voting ensemble passed the progression rule, so the LightGBM reference remains selected. A threshold changes precision, recall, F1, and the confusion matrix, but does not change Average Precision when the score ranking stays unchanged.

The model settings are retained; the operating rule still needs to be fixed before a one-time test evaluation. Final test results will be documented separately.

## Reporting update on 5 October 2026

Future Stage 11 and Stage 12 runs now save the maximum-F1 numerical threshold, accuracy, balanced accuracy, specificity, TP, FP, FN, TN, and alert rate alongside the existing metrics. These fields are logged by the shared evaluation helper into MLflow metrics and the existing result JSON and summary tables. Existing historical runs retain their original artifacts.

Verification: 21 targeted reporting, handoff, and selection tests passed. The reporting test checks that the saved threshold, confusion counts, accuracy, balanced accuracy, and alert rate agree on a small known example. The reconstructed counts for all 14 full-data Stage 11 and Stage 12 results reproduce their saved precision, recall, and F1 within numerical tolerance. No new model experiment or test-set evaluation was run for this update.
