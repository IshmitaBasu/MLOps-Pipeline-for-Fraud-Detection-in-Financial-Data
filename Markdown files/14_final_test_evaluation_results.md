# Stage 14: Final Test Evaluation Results

## Status

The protocol was frozen on 6 October 2026 before final test scoring. The one-time canonical evaluation is complete. All rules were applied unchanged from validation evidence. The model has been exported but is not registered or approved for production use.

## Frozen reference

- Stage 13 summary: `5920c17849574c11b4854c3c5290bc41`.
- Validation reference: `cb06aff922d544d4a34651550b147e42`.
- LightGBM reference with behavioural features and class weighting.
- Primary threshold: `0.5226022799524157` from validation maximum F1.
- Secondary scenarios: top 1%, 5%, and 10% batch review workloads.

The plan defines the complete frozen protocol and methodological limitations. The final evidence and interpretation are recorded below.

## Implementation checks

All 81 repository tests passed on 6 October 2026. Five final-evaluation tests cover the frozen handoff and threshold, absence of test threshold optimisation, equality with the development history definition, independence from query labels and events, and exclusive receipt creation. Formatting, compilation, and whitespace checks passed.

## Technical smoke evaluation

Run: `9bac93f8070c46a885281dfb8089269b`. The sampled workflow fitted 35,000 training rows and evaluated 7,500 sample test rows containing 269 fraud cases. It applied the full-validation threshold unchanged. This is a technical execution check, not the canonical revised test evaluation or model-selection evidence.

| Metric | Smoke value |
| --- | ---: |
| Average Precision | 0.043008 |
| ROC-AUC | 0.587727 |
| Precision | 3.1567% |
| Recall | 10.4089% |
| F1 | 0.048443 |
| Accuracy | 85.3333% |
| Balanced accuracy | 49.2648% |
| Specificity | 88.1206% |
| TN / FP / FN / TP | 6,372 / 859 / 241 / 28 |
| Alerts | 887 (11.8267%) |
| Fraud value captured | 11,680.72 (13.1916%) |
| Missed fraud value | 76,865.64 |
| Training / scoring time | 1.89 s / 0.30 s |

| Review share | TP | FP | FN | TN | Precision | Recall | Fraud-value recall | Missed fraud value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 2 | 73 | 267 | 7,158 | 2.6667% | 0.7435% | 1.6022% | 87,127.66 |
| 5% | 11 | 364 | 258 | 6,867 | 2.9333% | 4.0892% | 5.3858% | 83,777.41 |
| 10% | 25 | 725 | 244 | 6,506 | 3.3333% | 9.2937% | 11.6289% | 78,249.42 |

The sampled model has a different score distribution from the full-data model, which explains why the unchanged full-validation threshold does not reproduce the smoke maximum-F1 behaviour. No threshold was adjusted from these results. The smoke run verified metric logging, workload tables, history construction, and pipeline artifact export.

## Canonical full-data evaluation

Completed on 6 October 2026. MLflow run: `37b41018f28646598344fbcfe4ea4db6`. Experiment: `financial-fraud-stage-14-final-test-evaluation`.

The source data hashes, recorded model parameters, key library versions, and frozen split hash matched Stage 13. The same LightGBM configuration was fitted on the original 3,500,000 training rows. Validation rows were not added to training. Test histories queried strictly earlier training transactions; test events and labels did not update history.

The evaluated test partition contains 750,000 transactions: 26,933 fraudulent and 723,067 legitimate. No test threshold search, feature selection, parameter adjustment, or ensemble selection occurred.

### Fixed-threshold metrics

Primary threshold: score greater than or equal to `0.5226022799524157`, chosen from validation maximum F1.

| Metric | Test value |
| --- | ---: |
| Average Precision | 0.046860 |
| ROC-AUC | 0.615593 |
| Precision | 4.5333% |
| Recall | 88.3303% |
| F1 | 0.086240 |
| Accuracy | 32.7823% |
| Balanced accuracy | 59.5217% |
| Specificity | 30.7132% |
| Alerts | 524,780 of 750,000 (69.9707%) |
| Fraud value captured | 8,512,261.04 (88.1124%) |
| Missed fraud value | 1,148,423.78 |
| Training time | 956.52 s |
| Test scoring time | 14.96 s |

Average Precision is calculated with scikit-learn's `average_precision_score`. It is not the trapezoidal area under the precision-recall curve. Timing measures refer to this local run and are not deployment latency guarantees.

### Test confusion matrix

Rows are actual labels; columns are predicted labels. Fraud is the positive class.

| Actual / predicted | Predicted legitimate | Predicted fraud |
| --- | ---: | ---: |
| Actual legitimate | 222,077 true negatives | 500,990 false positives |
| Actual fraud | 3,143 false negatives | 23,790 true positives |

The four counts sum to 750,000, and their class totals match the frozen manifest. These are directly measured counts from final test scoring, not reconstructed historical values.

### Validation versus test at the same fixed rule

| Metric | Stage 13 validation | Final test |
| --- | ---: | ---: |
| Average Precision | 0.046721 | 0.046860 |
| ROC-AUC | 0.612986 | 0.615593 |
| Precision | 4.5345% | 4.5333% |
| Recall | 88.4157% | 88.3303% |
| F1 | 0.086266 | 0.086240 |
| Accuracy | 32.7395% | 32.7823% |
| Balanced accuracy | 59.5407% | 59.5217% |
| Alert share | 70.0196% | 69.9707% |
| Fraud-value recall | 87.7216% | 88.1124% |

The metrics are numerically close. This supports the observation that the weak performance seen during validation persists on the revised held-out split, rather than disappearing or improving substantially at final evaluation. It does not establish statistical equivalence or real-world validity.

### Predeclared batch review scenarios

These are separate fixed-proportion ranking rules. They are not fixed numerical API thresholds and do not imply actual institutional review capacity. Equal-score ties preserve original numeric transaction-ID order.

| Review share | Alerts | TP | FP | FN | TN | Precision | Recall | Fraud-value recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 7,500 | 363 | 7,137 | 26,570 | 715,930 | 4.8400% | 1.3478% | 1.3691% |
| 5% | 37,500 | 1,746 | 35,754 | 25,187 | 687,313 | 4.6560% | 6.4828% | 7.0846% |
| 10% | 75,000 | 3,575 | 71,425 | 23,358 | 651,642 | 4.7667% | 13.2737% | 14.1728% |

| Review share | F1 | Accuracy | Balanced accuracy | Fraud value captured | Missed fraud value |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.021084 | 95.5057% | 50.1804% | 132,267.54 | 9,528,417.28 |
| 5% | 0.054196 | 91.8745% | 50.7690% | 684,425.67 | 8,976,259.15 |
| 10% | 0.070144 | 87.3623% | 51.6978% | 1,369,191.57 | 8,291,493.25 |

At a 5% workload, the model finds 1,746 fraud cases and misses 25,187. Fraud-count recall falls from 6.7724% in validation to 6.4828% in test, while fraud-value recall falls from 7.1674% to 7.0846%. The restricted workload sharply reduces review volume, but most fraud cases and value remain undetected.

All amount totals use dataset units and represent fraud exposure. No currency, recovery rate, review cost, or institutional loss model is assumed.

## Observation and thesis interpretation

The revised model demonstrates reproducible training and final evaluation with recorded data lineage, parameters, rules, metrics, and a preserved fitted pipeline. Its predictive performance remains weak.

At the primary research rule, high recall comes from flagging about 70% of transactions, and more than 95% of alerts are false positives. At restricted review workloads, recall is low. An all-legitimate classifier would achieve 96.4089% test accuracy while detecting no fraud, which illustrates why accuracy alone is misleading.

Behavioural feature engineering provided a validation improvement earlier in the workflow. Subsequent model comparison, tuning, and voting experiments yielded little additional operational benefit. Final test evaluation confirms that the retained model's measured limitations persist under the revised protocol. It does not explain the cause of the limited predictive signal, establish future temporal performance, or justify production deployment.

The revised random-split test population comes from the same dataset used in earlier temporal experiments. This evaluation is the final held-out result for the revised protocol, not external validation on newly collected transactions.

Do not adjust this candidate's threshold or parameters using these test results and then report the same split as an untouched test. Further modelling would require a new evaluation design. The current findings can now be used for the thesis's modelling discussion and the prototype's documented limitations.

## Saved evidence and one-time receipt

The final run contains the frozen protocol, complete metric JSON, test workload CSV, confusion-matrix JSON, lineage and library metadata, and source snapshots. The exact fitted pipeline is saved as `evaluated_model/evaluated_lightgbm_pipeline.joblib` (1,064,433 bytes).

The local ignored receipt `final_test_evaluation/e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242.json` is marked `complete` for run `37b41018f28646598344fbcfe4ea4db6`. The MLflow run is marked as a canonical test evaluation. These controls block another full scoring run. No model registration or promotion has occurred.

## Next step

Communicate the completed modelling findings and final evaluation, including the weak precision/recall trade-off. The exact evaluated pipeline is available for a subsequent documented prototype registration and serving workflow; registration should preserve its research status and limitations.
