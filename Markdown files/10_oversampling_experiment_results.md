# Oversampling Experiment Results

## Status

The automated safeguards, 50,000-row smoke run and complete five-million-row
comparison passed. Full validation evidence retained the class-weight
reference. No oversampling strategy satisfied the predeclared progression rule.

## Frozen comparison

- Feature set: ten original predictors plus three retained sender-location
  history features
- Feature contract: `fraud_behaviour_features_v2`
- Model: fixed Random Forest with 100 trees and maximum depth 12
- Split: frozen stratified random 70/15/15 assignment
- Selection metric: validation Average Precision
- Operational comparison: equal 1%, 5% and 10% alert shares
- Test evaluated: no

## Automated tests

Status: passed.

| Item | Value |
| --- | --- |
| Stage 10 tests run | 11 |
| Stage 10 tests passed | 11 |
| Complete repository tests run | 48 |
| Complete repository tests passed | 48 |
| Technical result | All safeguards passed |

The checks confirmed the frozen strategies and ratios, random-oversampling row
duplication, deterministic SMOTENC output, valid synthetic categories, a frozen
category vocabulary, fixed Random Forest parameters, validation mutation
detection, test-metric rejection and the prohibition on selecting a strategy
from smoke results.

## Smoke run

Status: passed.

| Strategy | Training rows after resampling | Validation Average Precision | 5% fraud-count recall | 5% fraud-value recall | MLflow run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| Class-weight reference | 35,000 | 0.045428 | 5.9259% | 2.3391% | `8df85c269c154db39ad2bbb8f2798d4d` |
| 5:1 undersampling | 7,542 | 0.046666 | 5.1852% | 3.1616% | `4efddce7c47442aa984c753f0f61e642` |
| 10:1 random oversampling | 37,117 | 0.047002 | 8.1481% | 7.2472% | `8361610830934c679a7289d7c08f5928` |
| 5:1 random oversampling | 40,491 | 0.048027 | 6.6667% | 4.1955% | `763598043d234cc7a1f6695f014f73ed` |
| 10:1 SMOTENC | 37,117 | 0.044617 | 7.0370% | 6.7997% | `0cabc352b40d4d9f8adb8b9bc3e0c7a6` |
| 5:1 SMOTENC | 40,491 | 0.043998 | 5.1852% | 3.7392% | `bc1168e973264457a9cd041fec7abe69` |

| Summary item | Value |
| --- | --- |
| Summary run ID | `cbf1c503882b429895ad6e84ab6c3cdf` |
| Smoke split hash | `d7cd28f34aac15908701984877b2593634ce1dafe0b6262c79c4a611f6560ee7` |
| Training rows | 35,000: 33,743 legitimate and 1,257 fraud |
| Validation rows | 7,500 |
| Test rows held aside | 7,500 |
| Validation mutation check | Passed |
| SMOTENC categorical validity | Passed for both ratios |
| SMOTENC reproducibility with seed 42 | Passed for both ratios |
| Strategy selected | None; smoke runs cannot select a strategy |

The 5:1 random-oversampling configuration produced the highest smoke Average
Precision. The 10:1 random-oversampling configuration produced stronger
fraud-count and fraud-value recall at the 5% workload. This disagreement is one
reason not to turn the smoke ranking into a decision. The complete validation
comparison must apply the frozen progression rule.

Both SMOTENC runs completed with finite numeric values, valid categorical
outputs, the requested class ratios and reproducible results. They therefore
pass the technical smoke gate, even though their smoke Average Precision was
below the class-weight reference.

## Complete comparison

Status: complete.

| Strategy | Training rows | Average Precision | ROC-AUC | Maximum-F1 precision | Maximum-F1 recall | 5% fraud-count recall | 5% fraud-value recall | Training time | MLflow run ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Class-weight reference | 3,500,000 | 0.046448 | 0.612248 | 4.5476% | 85.5159% | 6.6461% | 6.8003% | 1,204.0 s | `2895b10806e24b22aed35dd227d6d681` |
| 5:1 undersampling | 754,122 | 0.046558 | 0.612929 | 4.5481% | 86.3216% | 6.6647% | 6.2659% | 218.9 s | `baa9d55e54ad4ddd9924e1ec196e4efb` |
| 10:1 random oversampling | 3,711,744 | 0.046595 | 0.612532 | 4.5466% | 86.1211% | 6.7575% | 6.5154% | 1,326.6 s | `4a8fd3e29ae248de8ca0cd5b4905d616` |
| 5:1 random oversampling | 4,049,175 | 0.046517 | 0.612717 | 4.5474% | 85.8129% | 6.6201% | 6.7261% | 1,368.1 s | `5ab9ed8f841745a094a228929a6f9e4d` |
| 10:1 SMOTENC | 3,711,744 | 0.046314 | 0.611162 | 4.5727% | 80.6557% | 6.6721% | 4.7379% | 1,199.0 s | `8c720996f1eb469eb75df184455a280d` |
| 5:1 SMOTENC | 4,049,175 | 0.046113 | 0.606793 | 4.5688% | 74.4811% | 6.7501% | 5.2944% | 1,260.9 s | `506328322d9c452fb172986348ec232e` |

| Summary item | Value |
| --- | --- |
| Summary run ID | `7d1e57bd9c0a49a59e0898bb3c7bad21` |
| Split hash | `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242` |
| Training rows | 3,500,000 |
| Validation rows | 750,000 |
| Held-out test rows | 750,000 |
| Feasibility exclusions | None |
| Selected strategy | `class_weight_reference` |
| Progression decision | `retain_class_weight_reference` |
| Test evaluated | No |

Random oversampling at 10:1 produced the highest Average Precision and the
highest fraud-count recall at the 5% workload. However, its fraud-value recall
was lower than the class-weight reference. The 5:1 random-oversampling result
also failed to improve fraud-count recall and fraud-value recall together.
Undersampling and both SMOTENC configurations failed the same operational
condition.

The differences in Average Precision were small, and none of the alternatives
preserved both operational recall measures while improving at least one of
them. The class-weight reference was therefore retained for Stage 11. This is
not a claim that class weighting solves the predictive-quality problem; it only
means that the tested resampling alternatives did not provide a sufficiently
consistent validation improvement.

The smoke ranking did not reproduce on the full dataset: 5:1 random
oversampling ranked highest in smoke, while 10:1 random oversampling had the
highest full-data Average Precision. This confirms why smoke scores were used
only for technical verification.
