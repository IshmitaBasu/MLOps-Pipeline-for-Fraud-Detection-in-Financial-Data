# Sender-Location History Feature Experiment Results

## Status

The controlled experiment and its 50,000-row smoke comparison were completed
on 26 September 2026. The full-data validation comparison then passed the
predeclared progression rule. `sender_location_history_v1` is retained for
later modelling experiments, while the test partition remains closed.

## Experimental question

Does `sender_location_history_v1` improve the fixed Random Forest over
`original_v1` on validation data while also maintaining or improving fraud-count
and fraud-value capture at an equal 5% alert volume?

## Frozen controls

- Stratified random 70/15/15 split with random state 42
- Fixed depth-12, class-weighted Random Forest
- Identical preprocessing and original features
- Three candidate history features only
- Validation-only comparison
- Test split closed

## Smoke run

Status: completed successfully on 26 September 2026.

| Item | Value |
| --- | --- |
| Control run ID | `5026f7ad0ba14c8a962620ba75298001` |
| Candidate run ID | `619ccc12b945457895a79ccd6adc557a` |
| Summary run ID | `dd153ae0bd3948479979943c11ed694e` |
| Split hash | `0e65b641f0618a0e2c6a4a63dc85ad4cb71100360f87600abcf23af83b15bc34` |
| Training rows | 35,000, including 1,257 fraud cases |
| Validation rows | 7,500, including 270 fraud cases |
| Test rows held aside | 7,500, including 269 fraud cases |
| Test evaluated | No |
| Technical outcome | Passed; both model runs, workload artifacts, and the comparison summary completed |

| Measure | `original_v1` | `sender_location_history_v1` | Candidate change |
| --- | ---: | ---: | ---: |
| Validation Average Precision | 0.041070 | 0.040182 | -0.000888 (-2.16%) |
| ROC-AUC | 0.560565 | 0.547802 | -0.012763 |
| Precision at maximum F1 | 0.043606 | 0.043828 | +0.000222 |
| Recall at maximum F1 | 0.992593 | 0.988889 | -0.003704 |
| F1 | 0.083541 | 0.083936 | +0.000395 |
| Training time | 5.181 seconds | 5.378 seconds | +0.197 seconds |
| Validation inference time | 0.136 seconds | 0.131 seconds | -0.004 seconds |

At an equal 5% workload, both models generated 375 alerts:

| Measure at 5% alerts | `original_v1` | `sender_location_history_v1` |
| --- | ---: | ---: |
| Fraud cases captured | 14 | 19 |
| Legitimate alerts | 361 | 356 |
| Precision | 3.7333% | 5.0667% |
| Fraud-count recall | 5.1852% | 7.0370% |
| Fraudulent value captured | 1,771.51 | 2,319.95 |
| Fraud-value recall | 1.8570% | 2.4320% |

The smoke result is mixed. The history candidate ranked transactions less well
overall according to Average Precision and ROC-AUC, but captured five more fraud
cases and more fraudulent transaction value at the 5% review volume. Only 270
fraud cases were present in validation, and a 50,000-row sample contains much
less sender history than the complete dataset. These differences are therefore
technical observations, not evidence for retaining or rejecting the features.

The original summary artifact applied the numerical rule and displayed
`retain_original_v1_and_do_not_progress_this_feature_group` because Average
Precision did not improve. That recommendation is explicitly ineligible: this
was a smoke run. The implementation has been clarified so future smoke
summaries report `smoke_run_no_feature_decision` instead.

## Full-data comparison

Status: completed successfully on 26 September 2026.

### Stopped full-run attempt

The first full command stopped before feature construction and model fitting
because the reconstructed split hash was
`2bb40c34b1ac5ef5b5af33e23500460578434c8c8a4d2b8f02bdcb2acc43da31`
instead of the frozen Stage 07 hash
`e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242`.

The underlying transactions and labels had not changed. The difference occurred
because Feast retrieval did not preserve exactly the raw CSV row order, while
the stratified random assignment is position-dependent. The safety check worked
as intended: no comparison runs were created, no model was fitted, and the test
partition was not evaluated.

The raw source was verified to contain five million continuously ordered IDs
from `T100000` through `T5099999`. The runner now restores that numeric
transaction-ID order before recreating the split. It still requires the exact
frozen hash before continuing.

| Measure | `original_v1` | `sender_location_history_v1` | Change |
| --- | ---: | ---: | ---: |
| Validation Average Precision | 0.043949 | 0.046448 | +0.002500 (+5.69%) |
| ROC-AUC | 0.594090 | 0.612248 | +0.018158 |
| Precision at maximum F1 | 0.043791 | 0.045476 | +0.001685 |
| Recall at maximum F1 | 0.997512 | 0.855159 | -0.142353 |
| F1 | 0.083898 | 0.086359 | +0.002460 |
| Fraud-count recall at 5% alerts | 6.1969% | 6.6461% | +0.4493 percentage points |
| Fraud-value recall at 5% alerts | 2.9031% | 6.8003% | +3.8972 percentage points |
| Training time | 986.106 seconds | 1,096.001 seconds | +109.896 seconds |
| Validation inference time | 11.167 seconds | 11.939 seconds | +0.773 seconds |

| Item | Value |
| --- | --- |
| Control run ID | `4c70f9e3cb78480cb524114ef52c45b2` |
| Candidate run ID | `09fe5e7b6f0e450aa2962f8542ca1f7c` |
| Summary run ID | `72fcb6b0a5394bcea26ddd2b75a79c61` |
| Reproduced split hash | `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242` |
| Test evaluated | No |
| Progression decision | Retain `sender_location_history_v1` for later model experiments |

### Equal-workload validation results

| Alert share | Feature group | Alerts | Fraud cases | Precision | Fraud-count recall | Fraud-value recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1% | `original_v1` | 7,500 | 305 | 4.0667% | 1.1324% | 0.4467% |
| 1% | `sender_location_history_v1` | 7,500 | 378 | 5.0400% | 1.4035% | 1.2834% |
| 5% | `original_v1` | 37,500 | 1,669 | 4.4507% | 6.1969% | 2.9031% |
| 5% | `sender_location_history_v1` | 37,500 | 1,790 | 4.7733% | 6.6461% | 6.8003% |
| 10% | `original_v1` | 75,000 | 3,333 | 4.4440% | 12.3752% | 6.8335% |
| 10% | `sender_location_history_v1` | 75,000 | 3,569 | 4.7587% | 13.2514% | 12.9320% |

At the predeclared 5% workload, the candidate found 1,790 fraud cases compared
with 1,669 for the control: 121 additional cases with the same 37,500 alerts.
It also reduced legitimate alerts from 35,831 to 35,710. Fraudulent transaction
value captured increased from 281,130.46 to 658,524.71, although transaction
amount is only a proxy for institutional loss.

## Interpretation

The full candidate satisfied every part of the frozen rule:

1. Average Precision improved by 5.69% relative to the control.
2. Fraud-count recall at the 5% workload improved rather than declining.
3. Fraud-value recall at the 5% workload improved rather than declining.
4. Both operational measures improved, exceeding the requirement that at least
   one must improve.

The candidate also improved ROC-AUC and F1. Recall at each model's maximum-F1
threshold was lower, but those thresholds produced different alert volumes and
are therefore not the fair operational comparison. At equal 1%, 5%, and 10%
alert volumes, the history candidate captured more fraud cases in every
scenario.

`sender_location_history_v1` should proceed to later validation-only
oversampling and model-family experiments. The result does not make it a final
model, does not authorise test evaluation, and does not yet demonstrate Feast
online serving. The three features now require a versioned feature contract and
consistent offline/online calculation before a serving claim is made.
