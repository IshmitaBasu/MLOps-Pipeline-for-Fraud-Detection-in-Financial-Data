# Plan for Model Optimisation and Operational Evaluation

This document describes the work planned after the baseline and feature-engineering experiments. The next stage has a clear order: reproduce the strongest controls, compare one additional model, study class imbalance, and only then tune the strongest option. The results will be written separately after the experiments have been completed.

## Starting point

The data preparation, Feast handoff, baseline comparison, and first feature experiment are complete. Random Forest is currently the strongest original-feature baseline, although its performance is still too weak for practical use.

| Item | Current reference |
| --- | --- |
| Data source | Feast historical retrieval, version `v1` |
| Feature set | `original_v1`, containing ten predictors |
| Dataset size | 5,000,000 transactions |
| Fraud cases | 179,553 (3.59106%) |
| Data split | Chronological 70% training, 15% validation, 15% test |
| Random state | 42 |
| Reference model | Random Forest with maximum depth 12 |
| Validation Average Precision | 0.0439598 |
| Test Average Precision | 0.0441134 |
| Test ROC-AUC | 0.5942102 |
| Test alert rate at the F1-selected threshold | About 82.1% |

The feature experiment did not provide enough evidence to replace the original feature set. The temporal features looked slightly better on validation data, but that improvement disappeared on the test period. For this reason, `original_v1` will be used for the model-optimisation experiments. This keeps the comparison understandable and avoids changing the model and features at the same time.

## Questions for this stage

The main question is whether a stronger model and a better treatment of class imbalance can improve fraud ranking without producing an unrealistic number of alerts.

More specifically, the experiments will examine:

1. whether a boosted-tree model performs better than the current Random Forest;
2. whether class weighting is sufficient or whether undersampling is useful;
3. whether careful tuning produces a meaningful improvement rather than a very small numerical change; and
4. whether a threshold can be found that balances detected fraud against the number of transactions sent for investigation.

A negative result is also useful. If the optimisation experiments do not improve the benchmark, that outcome will be reported instead of selecting a model simply because it is more complex.

## What will remain unchanged

The following parts will remain fixed throughout this stage:

- the Feast `v1` data and recorded hashes;
- the `original_v1` feature set;
- the chronological 70/15/15 split;
- random state 42 where the library supports it;
- training-only fitting of imputation, encoding, resampling, and model parameters; and
- validation Average Precision as the main model-selection metric.

The validation data will be used for choosing the model, imbalance strategy, hyperparameters, and operating threshold. The test data will not be used to revise those choices.

The test results from the earlier baseline and feature experiments are already known. Those values will be treated only as historical references, and the new models will not be tuned to beat them. The final candidate will be fixed using training and validation evidence before it is evaluated on the test period.

## Models to compare

The comparison will remain small enough that every model has a clear reason for being included.

| Model | Reason for inclusion |
| --- | --- |
| Random Forest | It is the frozen control and shows whether the new experiment reproduces the existing benchmark. |
| Histogram Gradient Boosting | It was almost as strong as Random Forest in the baseline experiment and trained much faster. |
| LightGBM | The selected external boosting model provides a stronger tabular candidate with support for weighted learning. |

LightGBM has been selected for implementation because training speed and memory use matter for five million rows. Version `4.7.0` is pinned in `requirements.txt` so that the experiment can be reproduced. The choice should still be supported by the literature review and, if possible, confirmed with the supervisor. XGBoost has not been added as a second dependency.

The Dummy Classifier and Logistic SGD results are already available from the baseline stage. They will remain in the written comparison, but reruns are only needed if the data interface changes.

## Experiment sequence

### 1. Start with smoke runs

Each new experiment family will first run on a reproducible 50,000-row sample. The smoke run is only a technical check. It should confirm that the data loads correctly, the pipeline trains, probabilities are produced, MLflow receives the expected information, and the cost and alert calculations work.

Smoke-run scores will not be used as thesis findings.

### 2. Compare the model families

Random Forest, Histogram Gradient Boosting, and the selected external boosting model will be run on the full dataset. Each model will begin with one documented configuration and its normal class-weighting method.

The main comparison will use validation Average Precision. Training time, inference speed, memory requirements, and the number of alerts produced at useful thresholds will also be considered. If two models perform almost identically, the simpler or substantially faster one will be preferred and the trade-off will be documented.

### 3. Compare imbalance strategies

After the most suitable model family has been selected, it will remain fixed while different ways of handling the rare fraud class are compared.

The main strategies will be:

- native class weighting;
- random undersampling with legitimate-to-fraud ratios of 10:1 and 5:1; and
- SMOTE or SMOTENC only if it is technically and semantically appropriate.

Resampling will be applied only to the training data. Validation and test data must keep their natural fraud rates.

Ordinary SMOTE will not be applied blindly after one-hot encoding. The dataset contains both numerical and categorical variables, and it has five million rows. If the categorical variables cannot be handled correctly or the memory cost is unreasonable, SMOTE will be excluded with an explanation. A justified exclusion is better than adding an unsuitable method only because it appeared in the proposal.

### 4. Tune only the selected approach

Only the selected model family and imbalance strategy will be tuned. This avoids an unnecessarily large search across every possible combination.

Depending on the model, the search may include:

- learning rate;
- number of trees or boosting iterations;
- tree depth or number of leaves;
- minimum leaf size;
- row and feature subsampling;
- regularisation; and
- class weight or positive-class scale.

The exact ranges will be documented before a full search begins. Tuning will start with a limited random search or a development sample. Only the strongest configurations will be trained on all five million rows.

### 5. Choose a useful operating threshold

The baseline threshold maximised F1, but it flagged about 82% of the test transactions. That is not realistic for a fraud-investigation process. For this reason, several validation thresholds will be examined for the selected model instead of relying only on maximum F1.

The analysis will report:

- the default probability threshold;
- the threshold that maximises validation F1;
- thresholds that flag 1%, 5%, and 10% of transactions; and
- thresholds that minimise the cost scenarios described below.

The chosen threshold and the reason for choosing it will be logged separately from the fitted model.

### 6. Evaluate one final candidate

Before the test data is used, the feature set, model family, preprocessing steps, imbalance method, hyperparameters, threshold, and acceptance criteria will be frozen. That one candidate will then be evaluated on the test period, and its fitted pipeline, model signature, input example, threshold, and final metrics will be stored in MLflow.

## Metrics to report

The main ranking metric will remain Average Precision. The current code logs this value under the name `pr_auc`, but the written thesis should describe it accurately as Average Precision, used as a summary of the precision-recall curve.

The report will also include:

- ROC-AUC;
- Precision, Recall, F1, and Specificity;
- the confusion matrix;
- alert rate and alerts per 10,000 transactions;
- fraud cases captured per 10,000 transactions;
- training and inference time; and
- inference rows per second.

Accuracy may be recorded, but it will not be used to decide which model is best because only about 3.59% of the transactions are fraudulent.

## Alert capacity and cost assumptions

The dataset does not contain real investigation costs or verified losses from missed fraud. No threshold can therefore be presented as the actual policy of a financial institution. Instead, transparent scenarios will be used and discussed as assumptions.

For investigation capacity, thresholds that flag 1%, 5%, and 10% of transactions will be compared. Each scenario will show how many fraud cases are detected, how many legitimate transactions are flagged, and how many fraud cases are missed.

For the cost comparison, the cost of one false positive will be set to 1 unit and three missed-fraud scenarios will be examined:

- false negative cost = 10 units;
- false negative cost = 50 units; and
- false negative cost = 100 units.

The calculation will be:

```text
expected cost = (false positives x false-positive cost)
              + (false negatives x false-negative cost)
```

These values are sensitivity assumptions, not real bank costs. If the supervisor suggests more suitable values, they will be updated before the final evaluation.

## Conditions for a model to be considered good enough

A clear rule is needed before looking at the final test result. The following conditions are provisional:

1. The model must beat the original validation Average Precision of 0.0439598.
2. The difference should be large enough to matter. A provisional target of at least 5% relative improvement will be used, subject to confirmation with the supervisor.
3. The model must offer a documented threshold for one of the 1%, 5%, or 10% alert-capacity scenarios.
4. Its alert or cost behaviour must be clearly better than the current 82.1% alert-rate result.
5. Training and inference must be feasible on the local thesis environment.
6. MLflow must contain the data lineage, parameters, metrics, artifacts, and source files needed to reproduce the result.
7. The automated tests must pass.

Meeting these conditions allows the candidate to receive one final test evaluation. It does not automatically make the model ready for deployment.

After the test evaluation, one of four decisions will be recorded:

- accept it as the prototype champion;
- retain it only as a research candidate;
- reject it; or
- continue development with one specifically justified experiment.

If no model reaches the provisional 5% target, that result will be reported clearly. The strongest model may still be useful as a research candidate, but it will not be described as operationally accepted.

## MLflow organisation

The model-optimisation experiments will use a new MLflow experiment:

```text
financial-fraud-feast-model-optimization-operational-evaluation
```

Run names will show the purpose of each run, for example:

```text
smoke__<model>__<strategy>
model_family__<model>
imbalance__<model>__<strategy>
tuning__<model>__<configuration>
selected_candidate__final_test
model_optimization_operational_evaluation_summary
```

Each full experiment should record the data and Feast hashes, feature list, split information, model settings, imbalance method, effective training counts, validation metrics, threshold tables, cost scenarios, runtime, plots, environment versions, and source snapshots.

A separate fitted model does not need to be stored for every screening run. The final frozen candidate should contain the complete model artifact and serving signature.

## Tests and files to add

The implementation should add tests for pipeline construction, deterministic configurations, train-only resampling, unchanged validation and test distributions, cost calculations, alert-rate thresholds, validation-only selection, and final-test access.

The planned files are:

```text
model_optimization_utils.py
06_model_optimization_and_operational_evaluation_with_mlflow.py
Markdown files/06_model_optimization_and_operational_evaluation_plan.md
Markdown files/06_model_optimization_and_operational_evaluation_implementation_guide.md
Markdown files/06_model_optimization_and_operational_evaluation_results.md
tests/test_model_optimization_and_operational_evaluation.py
```

Shared threshold and cost functions can be added to `fraud_modeling_utils.py` because they will also be useful later for serving and monitoring.

## What is not part of this step

This experiment will not introduce more engineered features, create a Feast `v2` handoff, register a champion before evaluation, build the FastAPI service, implement monitoring, or package the system with Docker. Those tasks come after the model decision.

## Points to review before the full runs

Before expensive full-data experiments begin, the following points should be confirmed with the supervisor where possible:

1. Is a 5% relative improvement in validation Average Precision a reasonable practical target for this thesis?
2. Are alert-rate scenarios of 1%, 5%, and 10% suitable for the prototype evaluation?
3. Are false-negative cost ratios of 10, 50, and 100 acceptable as sensitivity assumptions?
4. Is LightGBM acceptable as the selected external boosting model?
5. Is a documented exclusion of SMOTE acceptable if it is unsuitable for the mixed, five-million-row dataset?

The experiment structure and smoke tests can be implemented while these questions are being reviewed. No final full-data selection or acceptance claim should be made until the evaluation rules are agreed.
