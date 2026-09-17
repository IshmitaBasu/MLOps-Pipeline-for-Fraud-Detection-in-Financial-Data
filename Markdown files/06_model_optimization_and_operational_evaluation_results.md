# Model Optimisation and Operational Evaluation Results

This is the running record for stage 06. It records what was executed, the values produced by MLflow, and the conclusion that can reasonably be drawn from each experiment. Smoke runs are kept because they show that the implementation worked, but they are not used as thesis evidence. Full-data validation runs are recorded separately. The document should be extended after each new experiment rather than replacing earlier results.

## Current position

The model-family, imbalance-strategy, and fixed nine-configuration tuning experiments are complete. Random Forest remained the model-family winner, and 5:1 training-only undersampling produced the strongest imbalance result. The full tuning search selected `rf_leaf_50` by validation Average Precision, but its 0.15% relative improvement over the tuning reference was small and it did not improve the fixed-capacity results. It is therefore a validation-selected research candidate rather than an accepted champion. The held-out test period has not been evaluated in stage 06.

| Item | Current status |
| --- | --- |
| Data source | Feast historical retrieval, version `v1` |
| Feature set | `original_v1`, containing ten predictors |
| Split | Chronological 70% training, 15% validation, 15% held-out test |
| Selection metric | Validation Average Precision, logged as `validation_pr_auc` |
| Imbalance methods tested so far | Class weighting; 10:1 and 5:1 training-only undersampling |
| External model | LightGBM 4.7.0 |
| Full-data metric winner | Random Forest |
| Test split | Not evaluated in stage 06 |
| Selected tuning candidate | `rf_leaf_50`: Random Forest with 5:1 undersampling, 100 trees, depth 12, and minimum leaf size 50 |
| Tuning status | Smoke and full-data validation runs complete; selection made only from full validation evidence |
| Current decision | Research candidate pending supervisor review of the acceptance rule and operating threshold |
| Next controlled evaluation | One held-out test evaluation only if the candidate and threshold are frozen and approved |

## Fixed experiment conditions

The full dataset contains 5,000,000 transactions. The stage-06 split produced:

| Split | Rows | Fraud cases | Fraud rate | Period |
| --- | ---: | ---: | ---: | --- |
| Training | 3,500,000 | 125,453 | 3.58437% | 1 January to 14 September 2023 |
| Validation | 750,000 | 27,019 | 3.60253% | 14 September to 7 November 2023 |
| Held-out test | 750,000 | Not inspected in these runs | Not inspected | After the validation period |

The source dataset hash recorded by the model runs was:

```text
db1c01acc7c6425113a4d8d41df753f4c9fc49b5c1a0c37479cc258ac4e5354c
```

The same original five categorical and five numerical predictors were used for every model. The target, transaction identifier, and timestamp were excluded from the model input. Random state 42 and the fixed starting parameters were retained.

## Experiment 1: control-model smoke run

### Purpose

The first 50,000-row run checked whether the new stage-06 workflow could load Feast data, make the chronological split, train both controls, calculate threshold scenarios, write MLflow artifacts, and leave the test split untouched.

### Command

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" --sample-rows 50000 --skip-data-hash
```

### Results

| Model | Validation Average Precision | ROC-AUC | Training time | Run ID |
| --- | ---: | ---: | ---: | --- |
| Random Forest | 0.04121475 | 0.58625065 | 5.72 s | `564f916262cb44b598ca578de883311c` |
| Histogram Gradient Boosting | 0.04108453 | 0.57506533 | 2.89 s | `512bcf4292e84531abb5fb1cb3fffb1e` |

Summary run: `e6fa409ca7a54768b1d854e80c9152d9`

### Inference

The workflow completed correctly and selected Random Forest on the smoke validation sample. The difference was only 0.00013022 Average Precision, so it was not evidence that Random Forest was generally better. The purpose of this run was technical validation, not model selection.

## Experiment 2: LightGBM smoke comparison

### Purpose

LightGBM 4.7.0 was added as the selected external boosting library. The second smoke run checked that its preprocessing, class weighting, probability output, threshold calculations, and MLflow logging worked alongside the two controls.

### Command

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --models random_forest_depth_12 hist_gradient_boosting_leaves_31 lightgbm `
  --imbalance-strategies class_weight `
  --sample-rows 50000 `
  --skip-data-hash
```

### Results

| Model | Validation Average Precision | ROC-AUC | Training time | Run ID |
| --- | ---: | ---: | ---: | --- |
| Random Forest | 0.04121475 | 0.58625065 | 6.25 s | `9a85252c973b416bb5a0977f0d2787c5` |
| Histogram Gradient Boosting | 0.04108453 | 0.57506533 | 4.28 s | `b4c0dbeb74bf42fcbfcdc1208fe6f20f` |
| LightGBM | **0.04401866** | **0.58903914** | **2.22 s** | `7a143585f5334b6ba9764d0f61c298a5` |

Summary run: `37160949d91f4a889aa35a8bc8fbae15`

### Inference

LightGBM was about 6.80% above Random Forest and 7.14% above Histogram Gradient Boosting on this particular sample. It was also the fastest model in the run. This justified taking LightGBM into the full-data comparison, but the result remained smoke evidence and was not treated as a model improvement.

The two control scores were identical to the first smoke run despite being executed again. This confirms that sampling, model construction, and evaluation were deterministic under random state 42. Differences in elapsed time reflect normal runtime variation rather than a different fitted result.

## Experiment 3: full-data model-family comparison

### Purpose

The full-data run answered whether LightGBM could beat the two frozen controls when all three used the same five-million-row dataset, original feature set, chronological split, and class-weighting approach.

### Command

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --models random_forest_depth_12 hist_gradient_boosting_leaves_31 lightgbm `
  --imbalance-strategies class_weight
```

### Ranking results

| Model | Validation Average Precision | Relative difference from RF | ROC-AUC | Run ID |
| --- | ---: | ---: | ---: | --- |
| Random Forest | **0.04395982** | — | **0.59331067** | `d6c53c5d8684468682385000ab96d136` |
| Histogram Gradient Boosting | 0.04384216 | −0.27% | 0.59269045 | `da55d6f473254e35b3dc3b5465177aa8` |
| LightGBM | 0.04350242 | −1.04% | 0.59111932 | `79d3f0b111f54057b0407d6a462132e2` |

Summary run: `676ddf7de6264d8a872ac3d3392bf124`

Random Forest reproduced the earlier frozen validation Average Precision of 0.0439598. The difference is below the displayed precision, which supports the reproducibility of the Feast data handoff and fixed pipeline.

### Runtime results

| Model | Training time | Validation inference time | Validation rows per second |
| --- | ---: | ---: | ---: |
| Random Forest | 1,136.75 s (18.95 min) | 12.05 s | 62,246 |
| Histogram Gradient Boosting | 72.53 s (1.21 min) | **10.86 s** | **69,077** |
| LightGBM | 95.89 s (1.60 min) | 12.98 s | 57,787 |

Histogram Gradient Boosting trained about 15.7 times faster than Random Forest while its Average Precision was only about 0.27% lower. LightGBM trained about 11.9 times faster than Random Forest but had the lowest ranking score of the three.

### Operational thresholds

The following tables use the natural 750,000-row validation period. Precision and recall are shown as percentages for readability.

#### Random Forest

| Threshold rule | Threshold | Alert rate | Precision | Recall | Fraud found | Legitimate alerts | Fraud missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default 0.5 | 0.500000 | 75.88% | 4.38% | 92.35% | 24,953 | 544,171 | 2,066 |
| Maximum F1 | 0.428590 | 82.00% | 4.39% | 99.90% | 26,993 | 587,980 | 26 |
| 1% capacity | 0.544708 | 1.00% | 4.39% | 1.22% | 329 | 7,171 | 26,690 |
| 5% capacity | 0.537741 | 5.00% | 4.35% | 6.04% | 1,633 | 35,867 | 25,386 |
| 10% capacity | 0.534036 | 10.00% | 4.43% | 12.30% | 3,324 | 71,676 | 23,695 |

#### Histogram Gradient Boosting

| Threshold rule | Threshold | Alert rate | Precision | Recall | Fraud found | Legitimate alerts | Fraud missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default 0.5 | 0.500000 | 76.39% | 4.39% | 93.09% | 25,151 | 547,758 | 1,868 |
| Maximum F1 | 0.272284 | 81.83% | 4.39% | 99.64% | 26,922 | 586,784 | 97 |
| 1% capacity | 0.580794 | 1.00% | 4.40% | 1.22% | 330 | 7,170 | 26,689 |
| 5% capacity | 0.567775 | 5.00% | 4.40% | 6.11% | 1,651 | 35,849 | 25,368 |
| 10% capacity | 0.562409 | 10.00% | 4.37% | 12.13% | 3,277 | 71,723 | 23,742 |

#### LightGBM

| Threshold rule | Threshold | Alert rate | Precision | Recall | Fraud found | Legitimate alerts | Fraud missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default 0.5 | 0.500000 | 79.23% | 4.39% | 96.56% | 26,090 | 568,101 | 929 |
| Maximum F1 | 0.421125 | 81.99% | 4.39% | 99.89% | 26,989 | 587,914 | 30 |
| 1% capacity | 0.581058 | 1.00% | 4.37% | 1.21% | 328 | 7,172 | 26,691 |
| 5% capacity | 0.568097 | 5.00% | 4.30% | 5.97% | 1,613 | 35,887 | 25,406 |
| 10% capacity | 0.562698 | 10.00% | 4.31% | 11.97% | 3,234 | 71,766 | 23,785 |

### Cost-sensitivity results

The cost figures are relative units, not measured bank costs. One false positive costs 1 unit. The false-negative cost is varied between 10, 50, and 100 units.

| Model | FN cost | Cost-selected alert rate | Fraud missed | False positives | Total relative cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 10 | 0.00% | 27,019 | 0 | 270,190 |
| Random Forest | 50 | 82.09% | 2 | 588,626 | 588,726 |
| Random Forest | 100 | 82.09% | 2 | 588,626 | 588,826 |
| Histogram Gradient Boosting | 10 | 0.0013% | 27,018 | 9 | 270,189 |
| Histogram Gradient Boosting | 50 | 81.85% | 92 | 586,913 | 591,513 |
| Histogram Gradient Boosting | 100 | 81.85% | 92 | 586,913 | 596,113 |
| LightGBM | 10 | 0.00% | 27,019 | 0 | 270,190 |
| LightGBM | 50 | 82.09% | 0 | 588,666 | 588,666 |
| LightGBM | 100 | 82.09% | 0 | 588,666 | 588,666 |

### Inference from the full-data experiment

The LightGBM advantage seen in the smoke sample disappeared on the full validation period. This is why smoke scores cannot be used for model selection. LightGBM remains a valid implementation result, but its fixed starting configuration does not justify replacing the Random Forest control.

Random Forest is the strict validation-metric winner. However, the result is not an operationally convincing model. Its maximum-F1 threshold flags about 82% of the validation transactions. At a 10% alert capacity, it captures only about 12.30% of fraud cases and produces 71,676 legitimate alerts for 3,324 detected fraud cases. Precision remains close to the underlying fraud prevalence, showing only modest concentration of fraud among the alerts.

The cost analysis also produces extreme decisions. When a missed fraud costs 10 units, the lowest calculated cost is obtained by creating almost no alerts. When a missed fraud costs 50 or 100 units, the lowest cost requires flagging about 82% of transactions. The assumed costs therefore do not identify a workable middle ground with the current model scores. They are useful as sensitivity examples but should not be presented as a real banking policy.

Histogram Gradient Boosting deserves an explicit efficiency note. It is only 0.27% below Random Forest in validation Average Precision and is substantially faster to train. The automated summary selected Random Forest because the coded rule ranks only by validation Average Precision. The plan also states that runtime should be considered when models are nearly tied, so this trade-off should be acknowledged when the family for the imbalance experiment is frozen.

No candidate reached the provisional target of a 5% relative improvement over the Random Forest reference. This is a legitimate negative result rather than a failed experiment.

## Experiment 4: Random Forest imbalance-strategy smoke run

### Purpose

This smoke run kept the Random Forest configuration and validation data fixed while changing only the treatment of class imbalance. It checked deterministic training-only undersampling, effective training counts, model fitting, operational threshold artifacts, and MLflow metadata before a full-data comparison.

### Command

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --models random_forest_depth_12 `
  --imbalance-strategies class_weight undersample_10_to_1 undersample_5_to_1 `
  --sample-rows 50000 `
  --skip-data-hash
```

### Effective training data

The smoke sample produced 35,000 training rows, including 1,237 fraud cases. Undersampling retained every fraud case and removed only legitimate training transactions.

| Strategy | Effective training rows | Legitimate rows | Fraud rows | Effective fraud rate | Rows removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Class weighting | 35,000 | 33,763 | 1,237 | 3.53% | 0 |
| 10:1 undersampling | 13,607 | 12,370 | 1,237 | 9.09% | 21,393 |
| 5:1 undersampling | 7,422 | 6,185 | 1,237 | 16.67% | 27,578 |

The validation and held-out test splits were not resampled. The validation period retained its natural class distribution, and the test period was not evaluated.

### Ranking and runtime results

| Strategy | Validation Average Precision | Relative change from class weighting | ROC-AUC | Training time | Run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| Class weighting | 0.04121475 | — | 0.58625065 | 5.65 s | `9c15ad34a54a4c8bb62129d5452f49dc` |
| 10:1 undersampling | **0.04279611** | **+3.84%** | 0.58278920 | 1.95 s | `1f2b99ce2f7446fd8d3da28dc0d9d531` |
| 5:1 undersampling | 0.04229588 | +2.62% | **0.58636786** | **1.06 s** | `a041497856894e61b1aec841d36d97d3` |

Summary run: `1619728b93d249b58dfd74e515f363dd`

Both undersampling strategies increased smoke-sample Average Precision and reduced training time. The 10:1 strategy used about 61% fewer training rows and trained about 2.9 times faster than class weighting. The 5:1 strategy used about 79% fewer rows and trained about 5.3 times faster.

### Operational threshold results

The validation sample contained 7,500 transactions and 269 fraud cases.

| Strategy | Threshold rule | Threshold | Alert rate | Precision | Recall | Fraud found | Legitimate alerts | Fraud missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Class weighting | Maximum F1 | 0.455289 | 39.08% | 4.81% | 52.42% | 141 | 2,790 | 128 |
| Class weighting | 1% capacity | 0.537043 | 1.00% | 1.33% | 0.37% | 1 | 74 | 268 |
| Class weighting | 5% capacity | 0.511836 | 5.00% | 1.87% | 2.60% | 7 | 368 | 262 |
| Class weighting | 10% capacity | 0.496891 | 10.00% | 3.07% | 8.55% | 23 | 727 | 246 |
| 10:1 undersampling | Maximum F1 | 0.088206 | 56.09% | 4.47% | 69.89% | 188 | 4,019 | 81 |
| 10:1 undersampling | 1% capacity | 0.120617 | 1.00% | 4.00% | 1.12% | 3 | 72 | 266 |
| 10:1 undersampling | 5% capacity | 0.109876 | 5.00% | 4.27% | 5.95% | 16 | 359 | 253 |
| 10:1 undersampling | 10% capacity | 0.105589 | 10.00% | 4.40% | 12.27% | 33 | 717 | 236 |
| 5:1 undersampling | Maximum F1 | 0.168250 | 45.89% | 4.56% | 58.36% | 157 | 3,285 | 112 |
| 5:1 undersampling | 1% capacity | 0.204550 | 1.00% | 1.33% | 0.37% | 1 | 74 | 268 |
| 5:1 undersampling | 5% capacity | 0.192488 | 5.00% | 3.47% | 4.83% | 13 | 362 | 256 |
| 5:1 undersampling | 10% capacity | 0.186933 | 10.00% | 3.60% | 10.04% | 27 | 723 | 242 |

### Inference

The implementation behaved correctly. Every fraud row remained in the training data, only legitimate rows were removed, and the validation and test distributions were not changed. The shorter training times are consistent with the smaller effective datasets.

The 10:1 strategy is the smoke winner by Average Precision. At the 10% alert capacity, it found 33 fraud cases compared with 23 for class weighting and 27 for 5:1 undersampling. These counts come from only 269 validation fraud cases, so the differences may be caused by the small sample. The strategy cannot be selected until the full validation period confirms the result.

Undersampling also changed the scale of the model scores. For example, the 10% capacity threshold fell from about 0.497 with class weighting to 0.106 with 10:1 undersampling. This is expected because the model was trained with a different class proportion. It means that a threshold from one imbalance strategy cannot be reused for another; every candidate needs a threshold chosen on the unchanged validation distribution.

Average Precision and maximum F1 do not give exactly the same ordering in this smoke run. The class-weighted model had the highest maximum F1, while 10:1 undersampling had the highest Average Precision. The experiment continues to use Average Precision as the predeclared ranking metric.

## Experiment 5: full-data Random Forest imbalance comparison

### Purpose

The full-data run tested whether the smoke-sample undersampling improvements remained on the complete validation period. The Random Forest parameters, original feature set, split, and random state remained fixed. Only the imbalance strategy and resulting training-row count changed.

### Command

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --models random_forest_depth_12 `
  --imbalance-strategies class_weight undersample_10_to_1 undersample_5_to_1
```

### Effective training data

All strategies started from 3,500,000 chronological training rows containing 125,453 fraud cases. Both undersampling strategies retained every fraud case.

| Strategy | Effective training rows | Legitimate rows | Fraud rows | Effective fraud rate | Rows removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Class weighting | 3,500,000 | 3,374,547 | 125,453 | 3.58% | 0 |
| 10:1 undersampling | 1,379,983 | 1,254,530 | 125,453 | 9.09% | 2,120,017 |
| 5:1 undersampling | 752,718 | 627,265 | 125,453 | 16.67% | 2,747,282 |

The 750,000-row validation period retained its natural 3.60% fraud rate. The 750,000-row test period was held aside and was not evaluated.

### Ranking and runtime results

| Strategy | Validation Average Precision | Relative change from class weighting | ROC-AUC | Training time | Run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| Class weighting | 0.04395982 | — | **0.59331067** | 1,027.49 s (17.12 min) | `d278226c4f9e4048bdb3b16fa86340c5` |
| 10:1 undersampling | 0.04365273 | −0.70% | 0.59222266 | 358.17 s (5.97 min) | `f7ed4b52e6574cff8e62af6fd6faf3a5` |
| 5:1 undersampling | **0.04422677** | **+0.61%** | 0.59281242 | **161.95 s (2.70 min)** | `5ac10c92f9004d438d55f7d6ba9910f2` |

Summary run: `1a0e31d0b14f427fbaad26de488efcea`

The 5:1 strategy trained about 6.3 times faster than class weighting. The 10:1 strategy trained about 2.9 times faster but did not retain its smoke-sample ranking advantage.

### Operational threshold results

| Strategy | Threshold rule | Threshold | Alert rate | Precision | Recall | Fraud found | Legitimate alerts | Fraud missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Class weighting | Maximum F1 | 0.428590 | 82.00% | 4.39% | 99.90% | 26,993 | 587,980 | 26 |
| Class weighting | 1% capacity | 0.544708 | 1.00% | 4.39% | 1.22% | 329 | 7,171 | 26,690 |
| Class weighting | 5% capacity | 0.537741 | 5.00% | 4.35% | 6.04% | 1,633 | 35,867 | 25,386 |
| Class weighting | 10% capacity | 0.534036 | 10.00% | 4.43% | 12.30% | 3,324 | 71,676 | 23,695 |
| 10:1 undersampling | Maximum F1 | 0.090407 | 81.75% | 4.39% | 99.62% | 26,917 | 586,201 | 102 |
| 10:1 undersampling | 1% capacity | 0.111079 | 1.00% | 3.97% | 1.10% | 298 | 7,202 | 26,721 |
| 10:1 undersampling | 5% capacity | 0.107755 | 5.00% | 4.30% | 5.97% | 1,612 | 35,888 | 25,407 |
| 10:1 undersampling | 10% capacity | 0.106224 | 10.00% | 4.32% | 12.00% | 3,241 | 71,759 | 23,778 |
| 5:1 undersampling | Maximum F1 | 0.118031 | 82.07% | 4.39% | 99.99% | 27,016 | 588,516 | 3 |
| 5:1 undersampling | 1% capacity | 0.195562 | 1.00% | 4.76% | 1.32% | 357 | 7,143 | 26,662 |
| 5:1 undersampling | 5% capacity | 0.191310 | 5.00% | 4.67% | 6.48% | 1,750 | 35,750 | 25,269 |
| 5:1 undersampling | 10% capacity | 0.189143 | 10.00% | 4.58% | 12.71% | 3,434 | 71,566 | 23,585 |

At the same operational capacities, 5:1 undersampling found 28 more fraud cases at 1%, 117 more at 5%, and 110 more at 10% than class weighting. It also produced fewer legitimate alerts at each capacity because the total alert count was fixed.

The default threshold of 0.5 produced no alerts for either undersampled model. Undersampling changed the training prevalence and therefore the probability scale. This confirms that 0.5 cannot be treated as a universal operating threshold after resampling. Thresholds must be selected from the unchanged validation distribution and stored with the model.

### Cost-sensitivity results

| Strategy | FN cost | Cost-selected alert rate | Fraud missed | False positives | Total relative cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Class weighting | 10 | 0.00% | 27,019 | 0 | 270,190 |
| Class weighting | 50 | 82.09% | 2 | 588,626 | 588,726 |
| Class weighting | 100 | 82.09% | 2 | 588,626 | 588,826 |
| 10:1 undersampling | 10 | 0.0055% | 27,014 | 36 | 270,176 |
| 10:1 undersampling | 50 | 82.08% | 3 | 588,585 | 588,735 |
| 10:1 undersampling | 100 | 82.08% | 3 | 588,585 | 588,885 |
| 5:1 undersampling | 10 | 0.00% | 27,019 | 0 | 270,190 |
| 5:1 undersampling | 50 | 82.08% | 1 | 588,574 | **588,624** |
| 5:1 undersampling | 100 | 82.08% | 1 | 588,574 | **588,674** |

The cost-selected policies remain extreme: almost no alerts for a false-negative cost of 10 and about 82% alerts for costs of 50 or 100. The 5:1 strategy has the lowest cost under the two higher missed-fraud assumptions, but the numerical difference is small and the resulting workload is still unrealistic.

### Inference

The smoke winner did not generalise: 10:1 undersampling moved from first place on the sample to last place on the full validation period. The 5:1 strategy became the full-data winner. This is another example of why smoke scores are retained only as implementation evidence.

The 5:1 improvement over class weighting is 0.00026695 Average Precision, or about 0.61% relative. It is consistent with slightly better fraud capture at the fixed alert capacities and substantially shorter training time. This makes Random Forest with 5:1 undersampling the appropriate combination to carry into a limited tuning experiment.

The improvement remains well below the provisional 5% acceptance target. The 5:1 combination is therefore a tuning candidate, not an accepted model. Maximum-F1 operation still flags about 82% of validation transactions, so the operational workload problem is unresolved.

## Decision after the imbalance-strategy comparison

At this point in the experiment sequence, the validation-based choice for further development was Random Forest with 5:1 training-only undersampling. This choice was based on the highest full-data validation Average Precision, better fraud capture at the fixed capacities, and shorter training time. It did not satisfy the model-acceptance target.

The fixed nine-configuration search is implemented in the stage-06 runner and documented in the experiment plan. It varies the number of trees, maximum depth, minimum leaf size, and maximum feature fraction while keeping Random Forest, 5:1 training-only undersampling, `original_v1`, the chronological split, and random state 42 fixed.

After the tuning smoke test passed, the next action at that checkpoint was the following full-data tuning command:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --mode tuning
```

The completed full run executed the same nine configurations without changing the search in response to the smoke scores. As planned, the full validation result rather than the smoke ranking selected the configuration. Experiment 7 below records that outcome.

## Tuning implementation checkpoint

The tuning workflow was implemented before any tuning score was observed. It adds `--mode tuning` to the stage-06 runner and uses the nine configurations listed in the experiment plan. The tuning function accepts training and validation frames but has no test-frame argument. Every model run also passes `test_df=None` to the shared evaluation function.

The selected 5:1 undersampling is applied once with random state 42, and the same resulting training rows are used for every tuning configuration. No additional class weight is applied. Each run records its parameters, validation metrics, runtime, threshold scenarios, cost scenarios, data lineage, and source snapshot in MLflow. Smoke summaries identify only the highest smoke score and explicitly record that no selection was made.

The new focused test module contains six safeguards covering the fixed configuration list, unique parameter sets, pipeline parameters, absence of additional class weighting, deterministic training-only resampling, validation/test preservation, rejection of test-metric selection, tuning CLI restrictions, and the absence of a test-frame path through the tuning function.

Verification completed before the tuning smoke run:

```text
Focused stage-06 tests: 6 passed
Complete repository tests: 17 passed
Python compilation: passed
In-memory fit and probability prediction: passed for all 9 configurations
```

No MLflow tuning experiment was started during implementation verification.

## Experiment 6: Random Forest tuning smoke run

### Purpose

The tuning smoke run checked that all nine predeclared Random Forest configurations could use the same 5:1 undersampled training data, fit successfully, produce validation scores and operational artifacts, and remain isolated from the held-out test period. It was not used to modify the search or select a configuration.

### Command

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --mode tuning `
  --sample-rows 50000 `
  --skip-data-hash
```

### Execution checks

| Check | Recorded value |
| --- | --- |
| Source sample | 50,000 transactions |
| Effective training rows after 5:1 undersampling | 7,422 |
| Validation rows | 7,500 |
| Held-out test rows | 7,500, not evaluated |
| Candidate runs | 9, all finished |
| Summary decision status | `technical_smoke_no_selection` |
| Summary selection basis | `none_smoke_test_only` |
| Summary run | `d7f5a31f420a4080be7ea7e1ca75d925` |

### Ranking and runtime results

| Configuration | Validation Average Precision | ROC-AUC | Training time | Validation inference | Run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| `rf_reference` | 0.04229588 | 0.58636786 | 1.09 s | 0.135 s | `978155c5875742ccb85991f43f258f7e` |
| `rf_trees_200` | 0.04273490 | 0.58888079 | 2.05 s | 0.207 s | `a64ccce3e39646ca8386873a59b4e44f` |
| `rf_trees_300` | 0.04247514 | 0.58636478 | 3.05 s | 0.296 s | `c9d5806ca67947f9ad0851d31806dd07` |
| `rf_depth_8` | 0.04428175 | **0.59464131** | 1.07 s | **0.109 s** | `0c127b3d91e04a869c4d495b4f53e86c` |
| `rf_depth_16` | 0.04239626 | 0.58821760 | 1.10 s | 0.127 s | `1769968a50be45bd89595480f442a88e` |
| `rf_leaf_50` | 0.04237636 | 0.58466773 | 1.24 s | 0.146 s | `58bc676f57224f7e9be2e9a2c8a4af72` |
| `rf_leaf_250` | 0.04090750 | 0.56414734 | **0.92 s** | 0.132 s | `8735fa5925164ff2813135b72ca6ab9c` |
| `rf_features_half` | 0.04502743 | 0.58353413 | 2.11 s | 0.126 s | `73e2862991c64f798950dd58fa07e927` |
| `rf_combined_flexible` | **0.04586158** | 0.58701872 | 5.00 s | 0.224 s | `986cb23d60684ffd8844d0f2ae54f610` |

The combined flexible configuration produced the highest smoke Average Precision, followed by half-feature splits and depth 8. These positions are diagnostic only. No comparison with the full-data 5:1 reference or provisional acceptance target was treated as valid in the smoke summary.

### Fixed-capacity results

The validation sample contained 269 fraud cases. Each capacity contains the same total number of alerts, so the table reports fraud cases found; the remaining alerts were legitimate transactions.

| Configuration | Maximum-F1 alert rate | Fraud found at 1% | Fraud found at 5% | Fraud found at 10% |
| --- | ---: | ---: | ---: | ---: |
| `rf_reference` | 45.89% | 1 | 13 | 27 |
| `rf_trees_200` | 59.63% | 1 | 15 | 32 |
| `rf_trees_300` | 35.17% | 1 | 16 | 27 |
| `rf_depth_8` | 43.27% | 3 | 12 | 31 |
| `rf_depth_16` | 65.39% | 1 | 12 | 27 |
| `rf_leaf_50` | 68.85% | 2 | 10 | 26 |
| `rf_leaf_250` | 64.81% | 2 | 16 | 28 |
| `rf_features_half` | 73.19% | **4** | **18** | **32** |
| `rf_combined_flexible` | 54.16% | **4** | 12 | 31 |

The Average Precision leader was not the strongest candidate at every fixed capacity. `rf_features_half` matched or exceeded the combined configuration at 1%, 5%, and 10% capacity. With only 269 fraud cases, one or two detected transactions can noticeably change these smoke percentages. The full validation period is required before interpreting this trade-off.

### Cost-sensitivity results

| Configuration | FN=10: cost / alert rate | FN=50: cost / alert rate | FN=100: cost / alert rate |
| --- | --- | --- | --- |
| `rf_reference` | 2,690 / 0.00% | 6,217 / 82.40% | 6,492 / 84.76% |
| `rf_trees_200` | 2,690 / 0.00% | 6,241 / 82.04% | 6,554 / 89.63% |
| `rf_trees_300` | 2,690 / 0.00% | 6,304 / 82.20% | 6,605 / 83.57% |
| `rf_depth_8` | 2,690 / 0.00% | 6,231 / 82.59% | 6,456 / 84.28% |
| `rf_depth_16` | 2,690 / 0.00% | 6,252 / 82.87% | 6,492 / 86.11% |
| `rf_leaf_50` | 2,690 / 0.00% | 6,322 / 83.12% | 6,672 / 83.12% |
| `rf_leaf_250` | 2,689 / 0.13% | 6,434 / 84.61% | 6,715 / 90.43% |
| `rf_features_half` | **2,677 / 0.12%** | 6,069 / 81.79% | 6,180 / 83.29% |
| `rf_combined_flexible` | 2,681 / 0.03% | **6,008 / 82.33%** | **6,108 / 82.33%** |

The cost scenarios continue to choose extreme alert rates. The smoke run therefore provides no new evidence that the operational workload problem has been solved.

### Inference

The smoke run confirms that every tuning configuration is technically usable. Increasing the number of trees from 100 to 200 raised the sample score, while 300 trees did not improve it further. Depth 8 performed better than both the reference and depth 16, suggesting that stronger structural regularisation may be useful. A minimum leaf size of 250 produced the weakest ranking result. Considering half of the transformed predictors per split produced a larger improvement, and the combined flexible configuration had the highest overall smoke Average Precision.

These observations are hypotheses for the already-fixed full-data search, not reasons to remove, add, or alter candidates. The smoke ranking may reverse on the full validation period, as already happened in the model-family and imbalance experiments. All nine configurations will therefore proceed unchanged.

The test period remains closed. No final model, threshold, champion registration, or deployment claim has been made.

## Experiment 7: Full-data Random Forest tuning

### Purpose

The full tuning run evaluated the same nine predeclared Random Forest configurations on the complete training and validation periods. Every configuration used the selected 5:1 training-only undersampling strategy and the unchanged `original_v1` feature set. The held-out test period remained closed.

### Command

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\06_model_optimization_and_operational_evaluation_with_mlflow.py" `
  --mode tuning
```

### Execution checks

| Check | Recorded value |
| --- | --- |
| Effective training rows after 5:1 undersampling | 752,718 |
| Validation rows | 750,000 |
| Held-out test rows | 750,000, not evaluated |
| Fraud cases in validation | 27,019 |
| Candidate runs | 9, all finished |
| Evidence scope | `full_data_validation` |
| Summary decision status | `selected_by_full_validation_pr_auc` |
| Summary run | `0653535d894b47e4af7c7db43a51c3cd` |

### Ranking and runtime results

| Configuration | Validation Average Precision | ROC-AUC | Training time | Validation inference | Run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| `rf_leaf_50` | **0.04429251** | **0.59452148** | 173.76 s | 11.03 s | `3cd51150881346bdb95faf37bb44be9e` |
| `rf_reference` | 0.04422677 | 0.59281242 | 177.07 s | 10.85 s | `bd56b33a2492486f9abc39d86ddd104f` |
| `rf_trees_200` | 0.04395807 | 0.59258532 | 346.69 s | 19.83 s | `0ff060db524747279265fd2a9a308d0e` |
| `rf_depth_16` | 0.04394979 | 0.59308899 | 202.74 s | 14.27 s | `25b3ada4ade748b3aa8dce2163c957b4` |
| `rf_combined_flexible` | 0.04392870 | 0.59374069 | 2,142.75 s | 24.18 s | `58435df551ef4f08ab91cc53b2f6f0e1` |
| `rf_trees_300` | 0.04391177 | 0.59279700 | 478.03 s | 24.09 s | `e038b70308954c25a55b5b3b5519c35b` |
| `rf_leaf_250` | 0.04381318 | 0.59242614 | 138.82 s | 8.96 s | `1446614fa0e745dab830cbedf9b02b63` |
| `rf_depth_8` | 0.04363318 | 0.59077670 | **131.51 s** | **7.60 s** | `28d1e0988be047e0ad27542ffa76e8f2` |
| `rf_features_half` | 0.04359319 | 0.59195939 | 391.35 s | 7.75 s | `a95804cedab34f95801c9671ddd39c46` |

The validation selection rule identifies `rf_leaf_50`: 100 trees, maximum depth 12, minimum leaf size 50, and `sqrt` feature sampling. Its Average Precision is 0.00006573 above the 5:1 reference, a relative improvement of approximately **0.15%**. Compared with the frozen original reference of 0.04395982, the relative improvement is approximately **0.76%**.

The selected score remains below the provisional acceptance value of 0.04615781, which represents a 5% relative improvement over the frozen original reference. Therefore, `rf_leaf_50` is the tuning winner but does not meet the current provisional improvement condition.

### Fixed-capacity results

Each candidate produced exactly 7,500, 37,500, and 75,000 alerts at the 1%, 5%, and 10% capacities. The table reports the number of fraud cases found within those alerts.

| Configuration | Maximum-F1 alert rate | Fraud found at 1% | Fraud found at 5% | Fraud found at 10% |
| --- | ---: | ---: | ---: | ---: |
| `rf_reference` | 82.07% | **357** | **1,750** | **3,434** |
| `rf_trees_200` | 82.07% | 315 | 1,684 | 3,374 |
| `rf_trees_300` | 82.08% | 306 | 1,681 | 3,388 |
| `rf_depth_8` | 82.05% | 332 | 1,657 | 3,261 |
| `rf_depth_16` | 82.02% | 317 | 1,651 | 3,313 |
| `rf_leaf_50` | 82.02% | 307 | 1,749 | 3,417 |
| `rf_leaf_250` | 82.03% | 306 | 1,658 | 3,318 |
| `rf_features_half` | **80.61%** | 319 | 1,584 | 3,245 |
| `rf_combined_flexible` | 82.03% | 324 | 1,642 | 3,316 |

The small Average Precision increase from `rf_leaf_50` does not produce a fixed-capacity improvement. The reference finds 50 more fraud cases at 1% capacity, one more at 5%, and 17 more at 10%. The selected candidate's maximum-F1 threshold would also alert on about 82% of validation transactions, so it does not solve the workload problem seen in the earlier experiments.

### Cost-sensitivity results

| Configuration | FN=10: cost / alert rate | FN=50: cost / alert rate | FN=100: cost / alert rate |
| --- | --- | --- | --- |
| `rf_reference` | 270,190 / 0.00% | **588,624 / 82.08%** | **588,674 / 82.08%** |
| `rf_trees_200` | 270,190 / 0.00% | 588,651 / 82.08% | 588,701 / 82.08% |
| `rf_trees_300` | 270,190 / 0.00% | 588,649 / 82.08% | 588,708 / 82.08% |
| `rf_depth_8` | **270,180 / 0.0001%** | 588,681 / 82.09% | 588,731 / 82.09% |
| `rf_depth_16` | 270,190 / 0.00% | 588,903 / 82.07% | 589,136 / 82.11% |
| `rf_leaf_50` | 270,190 / 0.00% | 588,630 / 82.07% | 588,713 / 82.08% |
| `rf_leaf_250` | 270,185 / 0.0008% | 588,785 / 82.07% | 588,951 / 82.09% |
| `rf_features_half` | 270,190 / 0.00% | 588,656 / 82.09% | 588,656 / 82.09% |
| `rf_combined_flexible` | 270,190 / 0.00% | 588,656 / 82.09% | 588,656 / 82.09% |

The assumed cost ratios still favour operational extremes: almost no alerts when a missed fraud costs 10 units and roughly 82% of transactions when it costs 50 or 100 units. These sensitivity assumptions do not identify a practical operating threshold, and the selected candidate is not clearly better than the reference under them.

### Comparison with the smoke run

The smoke ranking did not generalise to the complete validation period. `rf_combined_flexible`, `rf_features_half`, and `rf_depth_8` were the three strongest smoke configurations, but they ranked fifth, ninth, and eighth respectively on full data. `rf_leaf_50` ranked sixth in the smoke run and first on full data. This reversal confirms why smoke scores were restricted to technical verification and were not used to alter the predeclared search.

### Inference and decision

The full search selects `rf_leaf_50` according to the predeclared validation Average Precision rule. The result is technically reproducible and feasible on the local environment, but its improvement is very small and it does not improve the fixed-capacity or maximum-F1 workload results. It therefore remains a **validation-selected research candidate**, not an accepted prototype champion.

No operating threshold has been frozen. Under the current provisional rules, a final held-out test evaluation should not be run until the 5% improvement requirement and the operational threshold expectations have been reviewed with the supervisor. If a one-time final evaluation of the strongest research candidate is approved despite missing the provisional target, that exception should be recorded before the test period is opened.

The test period remains unevaluated, and no model has been registered or promoted.

## MLflow location

Experiment:

```text
financial-fraud-feast-model-optimization-operational-evaluation
```

Each candidate run contains:

```text
operational_evaluation/validation_threshold_summary.csv
operational_evaluation/validation_threshold_summary.json
evaluation/metrics_summary.json
reproducibility/split_summary.json
reproducibility/feature_set.json
reproducibility/environment.json
```

The summary runs contain the candidates, selected validation result, relative improvement, and embedded threshold scenarios. MLflow remains the source of exact machine-readable values; this document is the readable experiment record and interpretation.
