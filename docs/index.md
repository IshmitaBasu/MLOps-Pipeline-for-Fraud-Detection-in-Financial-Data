# Documentation guide

This page is the starting point for the project documentation. The repository root contains only configuration and entry-point files; implementation code, experiments, notebooks, diagrams and generated evidence now have separate locations.

## Where to look

| Need | Location |
| --- | --- |
| Current project status and common commands | [`README.md`](../README.md) |
| Chronological experiment plans, implementation notes and results | [`docs/experiments/`](experiments/) |
| Architecture diagrams | [`docs/architecture/`](architecture/) |
| Documentation figures | [`docs/images/`](images/) |
| Notebook-generated reports | [`docs/reports/`](reports/) |
| Directly runnable workflow files | [`stages/`](../stages/) |
| Reusable implementation code | [`fraud_detection/`](../fraud_detection/) |
| Automated safeguards | [`tests/`](../tests/) |

## Experiment documentation by topic

### Data understanding and preparation

- Stages 01–03: raw EDA, preprocessing and post-cleaning EDA.
- Stage 04: original-feature model benchmark.
- [`02_data_pipeline_preprocessing.md`](experiments/02_data_pipeline_preprocessing.md) explains the cleaned-data handoff.
- [`original_feature_benchmark_v1.md`](experiments/original_feature_benchmark_v1.md) records the original modelling reference.

### Feature and model development

- Stages 05–09: feature experiments, behavioural diagnostics and Feast serving.
- Stages 10–14: imbalance handling, model families, LightGBM tuning, voting and final evaluation.
- Files ending in `_plan.md` state the question and decision rule.
- Files ending in `_implementation_guide.md` explain the code and commands.
- Files ending in `_results.md` record executed runs, values and interpretation.

### Serving, registration and monitoring

- [`15_model_serving_feature_contract.md`](experiments/15_model_serving_feature_contract.md): API and feature inputs.
- [`15_research_prediction_api_implementation_guide.md`](experiments/15_research_prediction_api_implementation_guide.md): local API operation.
- [`16_research_model_registration.md`](experiments/16_research_model_registration.md): MLflow registration and future versions.
- [`18_batch_monitoring_implementation_guide.md`](experiments/18_batch_monitoring_implementation_guide.md): monitoring inputs and commands.

### Predictive signal and alternative dataset work

- [`17_predictive_signal_diagnostics_results.md`](experiments/17_predictive_signal_diagnostics_results.md): training-only univariate evidence.
- [`19_multivariate_signal_diagnostics.md`](experiments/19_multivariate_signal_diagnostics.md): multivariate evidence.
- [`20_ibm_aml_dataset_feasibility_audit.md`](experiments/20_ibm_aml_dataset_feasibility_audit.md): IBM HI-Small feasibility.
- [`21_ibm_aml_baseline_plan.md`](experiments/21_ibm_aml_baseline_plan.md): fixed validation baseline and its result.

## Reading one experiment

For a numbered experiment, begin with its plan, then read the implementation guide, and finish with the results. Some later stages use one combined document when separating those sections would add repetition.

MLflow remains the source for run-level parameters and artifacts. The Markdown files provide the readable record of what was run, what was observed and what decision followed.
