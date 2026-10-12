# Batch monitoring and investigation rules

## Purpose

This stage adds monitoring around the frozen research model. It answers a different question from model evaluation: **does a later batch still look and behave like the reference population?** It does not try another classifier, change the threshold or retrain the model.

In simple terms, the reference profile is a photograph of a known batch. A later batch is compared with that photograph. A difference is a warning to investigate, not proof that the model has failed.

## What is monitored

| Area | Check | Meaning |
| --- | --- | --- |
| Data quality | Required columns, invalid numbers, score range, duplicate IDs and missingness | Can the batch be trusted technically? |
| Input drift | Population Stability Index (PSI) for every model feature | Have transaction characteristics changed? |
| Prediction drift | Score PSI and alert-rate change | Is the frozen model behaving differently? |
| Performance | Average Precision, precision, recall and confusion counts when labels arrive | Is predictive quality measurable on this batch? |
| Fraud value | Captured and missed fraud amount | Are high-value fraudulent transactions being missed? |
| Scenario cost | False-positive review cost plus missed fraud amount multiplier | What happens under stated cost assumptions? |

Scores are uncalibrated ranking scores, not verified fraud probabilities. Amount is treated as exposure in dataset units, not proven institutional loss. Scenario-cost outputs must always retain those assumptions.

## Reference and batch boundaries

A reference must be explicitly declared as a training or pre-deployment operational reference. Validation and the consumed final test partition are not accepted as monitoring-reference material. A later monitored CSV must be declared as an operational batch. The local API ledger can also be read directly and read-only.

The profile records the frozen Stage 14 model run, threshold, feature distributions, score distribution, alert rate, state label and rule thresholds. The monitor refuses a different model run or decision threshold. Reference creation rejects invalid scores, invalid numerics and duplicate transaction identifiers.

No canonical reference profile is frozen yet. A meaningful reference needs a sufficiently large, appropriately scoped scored population. The few manual API calls are useful endpoint demonstrations but not a distributional baseline. A synthetic profile is used only to exercise the software.

## Rule interpretation

The current rules are transparent research heuristics, not universal banking standards:

- PSI from 0.10 to below 0.25 is a warning; PSI of 0.25 or more is critical.
- An absolute missing-rate change of at least five percentage points is flagged.
- An alert-rate change by a factor of 1.5 or more in either direction is flagged.
- Label-based results require at least 100 labelled rows and 30 fraud cases.
- Fraud-value recall below 80% is a critical research flag when sufficient labels exist.

A critical result says `investigate_before_retraining_decision`. A warning says `investigate`. Neither result starts training. The report always records `automatic_retraining_triggered: false`; material action requires a separate decision and evidence check.

## Evidence boundary

Drift does not show why a population changed, and it does not by itself prove performance degradation. Performance remains explicitly unavailable until delayed labels exist. A stable distribution also does not prove that the weak research model is suitable for production.

This implementation is local and research-only. Authentication, retention automation, scheduled orchestration, production alert delivery and final business thresholds are outside its current scope.
