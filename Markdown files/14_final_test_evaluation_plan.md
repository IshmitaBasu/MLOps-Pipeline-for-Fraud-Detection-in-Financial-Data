# Stage 14: Frozen Final Test Evaluation Plan

## Protocol fixed before test scoring

Recorded on 6 October 2026. The final development model is the retained Stage 13 LightGBM reference, based on summary run `5920c17849574c11b4854c3c5290bc41` and reference run `cb06aff922d544d4a34651550b147e42`.

| Item | Frozen decision |
| --- | --- |
| Model | LightGBM reference, unchanged Stage 11/12 parameters |
| Features | Original features plus retained behavioural features |
| Imbalance treatment | Class weighting from training class counts |
| Training population | Original 3,500,000-row training partition only |
| Primary rule | Score greater than or equal to `0.5226022799524157` |
| Threshold source | Maximum F1 from the completed Stage 13 validation run |
| Secondary scenarios | Review the highest-scoring 1%, 5%, and 10% of the test batch |
| History source | Strictly earlier training transactions only |
| Random state | 42 |
| Test population | Frozen 750,000-row revised test partition |
| Split hash | `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242` |

## Why these rules

Maximum F1 is the primary research benchmark because it is already defined by validation evidence and can be applied as a fixed score threshold. Its validation review burden is about 70%, so it is not an assertion of a practical banking policy.

The 1%, 5%, and 10% workloads illustrate the fraud-count and fraud-value trade-off at equal batch review volumes. They do not represent known institutional capacity. Batch ranking uses descending score with stable ties resolved in the original numeric transaction-ID order. The proportion is fixed beforehand; no fraud labels select which rows to review.

## Restrictions

Do not search for a maximum-F1 threshold on the test set. Do not change model parameters, features, ensemble membership, weights, or the evaluation rules after reading the results. The validation partition is not added to training; doing so would alter the fitted model and its score distribution relative to the frozen threshold.

The final feature path treats test events as queries. Neither test labels nor test events update the historical counts. Training history is the same definition used during validation. No feature selection is performed during final evaluation.

## Evidence to report

Report Average Precision, ROC-AUC, precision, recall, F1, accuracy, balanced accuracy, specificity, confusion counts, review volume, captured fraud value, and missed fraud value. The workload table reports its own classification and value measures. Amounts remain in the dataset's units; they are exposure measures rather than institutional loss estimates.

Compare the fixed-threshold test result to the corresponding Stage 13 validation result, not to a newly optimised test rule. Document weak absolute performance even if validation and test metrics are similar.

## Scope and limitations

This is the final evaluation for the revised random-split protocol established in Stage 7. Earlier baseline and feature experiments used a different temporal test partition from the same dataset. This revised test split has been kept out of subsequent development evaluation, but it is not a newly collected external dataset. The random split and training-only history experiment do not establish performance on a future production stream.

Test findings may motivate future research, but must not be reused to tune this same evaluated candidate and then presented as an untouched final test result.
