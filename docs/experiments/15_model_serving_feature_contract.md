# Model-serving feature contract

## Status and purpose

Design recorded on 6 October 2026. The local feature adapter and FastAPI implementation now follow this contract; see [the implementation guide](15_research_prediction_api_implementation_guide.md) and [verification results](15_research_prediction_api_results.md). Model registration remains pending. The evaluated model and final test results are unchanged.

The serving workflow must assemble the same predictors with the same definitions used during training. A successful HTTP response alone is not evidence of train-serving consistency.

The evaluated model has weak predictive performance. Serving is a research demonstration, not approval for live fraud decisions. Completed predictive results remain in [the final evaluation report](14_final_test_evaluation_results.md); they are not repeated here.

## Fixed model and feature references

- Evaluated MLflow run: `37b41018f28646598344fbcfe4ea4db6`.
- Artifact: `evaluated_model/evaluated_lightgbm_pipeline.joblib` in that run.
- Feature set: `original_v1_plus_fraud_behaviour_features_v2`.
- Behaviour service: `fraud_behaviour_features_v2`.
- History population: the original 3,500,000 training transactions only.
- Frozen split SHA-256: `e8f58ca7f8c713df27f977c2661f296ea57a118678daebabd79bc4a39c3e9242`.
- Research decision: `score >= 0.5226022799524157`.
- Scores are not established calibrated fraud probabilities.

The exported pipeline contains the fitted imputation, categorical encoding and LightGBM model. Serving must load this artifact, not rebuild or refit its preprocessing. Model registration is a later step and must preserve this run and artifact reference.

## What the request supplies

The planned single-transaction request contains these fields. The first three identify the request and historical state; only `location` among the lookup keys is also an original model predictor.

| Field | Planned validation | Purpose |
| --- | --- | --- |
| `transaction_id` | Non-empty string | Traceability and duplicate-request handling; not a model feature |
| `event_timestamp` | Timezone-aware ISO-8601; normalise to UTC; whole-second precision for the initial interface | Select the appropriate historical state; not a direct model feature |
| `sender_account` | Non-empty string | Sender and sender-location lookup; not a direct model feature |
| `transaction_type` | String or explicit null | Original categorical predictor |
| `merchant_category` | String or explicit null | Original categorical predictor |
| `location` | Non-empty string | Original predictor and history key |
| `device_used` | String or explicit null | Original categorical predictor |
| `payment_channel` | String or explicit null | Original categorical predictor |
| `amount` | Finite, non-negative number | Original numeric predictor and logged transaction value |
| `time_since_last_transaction` | Finite, non-negative number or explicit null | Original numeric predictor |
| `spending_deviation_score` | Finite number or explicit null | Original numeric predictor |
| `velocity_score` | Integer or explicit null | Original numeric predictor |
| `geo_anomaly_score` | Finite number or explicit null | Original numeric predictor |

All listed fields must be present. Explicitly allowed nulls are converted to the missing-value representation expected by the saved pipeline. Required keys and amount cannot be null. Reject NaN/Infinity, booleans supplied as numbers, empty categorical strings, unexpected fields and client-supplied history counts. Unknown non-empty categories are handled by the fitted encoder's existing `handle_unknown="ignore"` setting.

Do not lowercase, trim or otherwise rewrite sender/location keys silently. Apply the same established preparation rules as the source handoff, and reject malformed keys. The existing preparation and model code remain the reference for units and meanings; do not invent new score scales.

The original `v1` Feast transaction view is offline-only. The initial API therefore receives the ten original predictors in the request; it does not assume an online transaction-ID lookup exists.

`time_since_last_transaction`, `spending_deviation_score`, `velocity_score` and `geo_anomaly_score` are already supplied in the dataset. Their real-world upstream calculation is not implemented. The prototype accepts prepared values and must state that dependency; supplying amount, sender and location alone is insufficient for this model.

Neither `is_fraud` nor `fraud_type` is accepted by the prediction endpoint. Later observed labels belong to a separate monitoring interface, not model inputs.

### Illustrative request

```json
{
  "transaction_id": "demo-001",
  "event_timestamp": "2026-10-06T12:00:00Z",
  "sender_account": "example-account",
  "transaction_type": "example-type",
  "merchant_category": "example-category",
  "location": "Berlin",
  "device_used": "example-device",
  "payment_channel": "example-channel",
  "amount": 120.0,
  "time_since_last_transaction": null,
  "spending_deviation_score": 0.2,
  "velocity_score": 2,
  "geo_anomaly_score": 0.1
}
```

These are invented interface examples, not dataset observations or evidence of prediction quality. No model score has been calculated for this request.

## What Feast supplies and what the adapter derives

| Model feature | Source / definition |
| --- | --- |
| `sender_prior_transaction_count` | `sender_history_features_v2`: number of training transactions for this sender with event time strictly earlier than the request |
| `sender_location_prior_transaction_count` | `sender_location_history_features_v2`: number of training transactions for this exact sender-location pair strictly earlier than the request |
| `is_new_location_for_returning_sender` | Adapter: integer 1 when sender count is greater than zero and pair count is zero; otherwise 0 |

Compose `sender_location_key` with the existing `compose_sender_location_key` helper: sender length, a colon, the unchanged sender, a vertical bar, and the unchanged location. For sender `A` and location `Berlin`, the key is `1:A|Berlin`. Do not substitute a different concatenation scheme.

After retrieval, require integer counts with `0 <= pair_count <= sender_count`. Convert the derived boolean flag to the model's 0/1 representation. Never let the client override these features.

Simple examples:

- Sender has five earlier training transactions, two in Berlin: counts `(5, 2)`, new-location flag `0`.
- Same sender visits a location absent from its training history: counts `(5, 0)`, flag `1`.
- Sender has no earlier training history: counts `(0, 0)`, flag `0`. A first transaction is not a new location for a *returning* sender.

## Training population and timestamp rules

Inspection of `build_behaviour_feature_artifacts` and the persisted v2 metadata confirms that Feast count snapshots already use training events only. Validation events do not update counts and test events are excluded. The remaining verification concerns retrieval time, defaults and model input assembly, not an assumed all-data history population.

For an event at time `t`, count only training events with `event_time < t`. Training transactions at exactly `t` must not count, even if their transaction IDs sort first. Labels never participate in the calculation.

Feast source snapshots contain post-event counts effective one microsecond after each event timestamp. They exclude equal-time events at that query timestamp. The source has sub-second timestamps; the initial future-online request interface deliberately accepts whole-second timestamps only. Historical replay at finer granularity requires separately tested timing rules, not an assumption that the source has whole-second precision.

### Historical replay

A replay request within the training-history time range needs point-in-time retrieval from the Parquet snapshot sources. Do not feed it the latest SQLite online counts: those could include training events occurring after the request.

Historical replay is a separate explicit path, not a silent fallback in the online endpoint. Start parity checks with synthetic fixtures and selected training/validation identifiers. Do not rescore the consumed final test or use replay to choose a new threshold.

### Future online requests

The initial online prototype uses a frozen training-history store. Accept online requests only after the last effective snapshot timestamp across both behavioural source tables, recorded in a verified deployment manifest. At those times, the latest fully materialised counts represent all strictly earlier training events.

At startup, verify the snapshot hashes and frozen split, both feature views, complete materialisation, and matching state versions. Readiness must fail if the two views are not from the same verified build. Record the source cutoff and materialisation boundary; do not infer them from the current clock or an arbitrary transaction.

Prediction requests, ingestion batches, delayed labels and validation/test events do not update the frozen online state. Repeated requests therefore do not increase counts. Demonstrating evolving history would need a separately versioned state policy and independent validation; it must not silently replace this contract.

This frozen state becomes stale for real behavioural monitoring. An online demonstration establishes serving mechanics, not predictive validity for transactions years after the dataset period.

## Missing entities and retrieval failures

Zero is appropriate for a genuinely unseen sender or sender-location pair. It is not a general response to a broken feature store.

The serving adapter must distinguish an absent entity from an unavailable, expired, corrupt or incompletely materialised store. Use verified immutable entity membership and build metadata, plus retrieval status checks, to justify cold-start defaults. Known entities unexpectedly missing their required state must fail rather than receive invented zero counts.

| Situation | Planned behaviour |
| --- | --- |
| Unknown sender, healthy verified store | Counts `(0, 0)`, flag `0`; log cold start |
| Known sender, unseen location pair | Retrieve sender count; pair count `0`; derive flag |
| Known pair unexpectedly missing or inconsistent counts | Fail prediction and log the state error |
| Feature store unavailable, incomplete materialisation or mismatched state version | Service unavailable; no score or decision |
| Timestamp invalid or not after the online state boundary | Reject online request; identify historical replay as the separate route |

The existing `normalise_served_features` helper converts missing/unparseable counts to zero for consistency checks. It is not, by itself, a sufficient serving failure policy. Implement status and integrity validation before reusing normalisation.

## Assemble the model input and response

Build a one-row DataFrame in the existing `MODEL_FEATURES` order:

```text
transaction_type, merchant_category, location, device_used, payment_channel,
amount, time_since_last_transaction, spending_deviation_score, velocity_score,
geo_anomaly_score, sender_prior_transaction_count,
sender_location_prior_transaction_count, is_new_location_for_returning_sender
```

Supply raw predictors and retrieved integer history features to the saved pipeline. Do not manually encode categories, compute new medians, scale inputs, resample requests or change LightGBM parameters.

The planned response contains `transaction_id`, `score`, `decision`, the exact threshold, model run/registered version, feature contract/state version, event timestamp and cold-start status. Name the output `score`, not a verified probability. `decision` means a research review flag, not confirmed fraud.

The top 1%, 5% and 10% workloads in the evaluation are batch ranking scenarios. They cannot be applied independently to each API request and must not be advertised as additional single-request threshold rules.

Prediction logs also record the assembled predictors, derived history values, state cutoff, request/processing times, latency and failures. Sender/account identifiers require a documented local access and retention policy; do not expose full request records indiscriminately in a dashboard. Delayed labels remain separate until explicitly joined for monitoring.

## Acceptance checks before registry-to-API integration

The following checks define acceptance. Completed checks and remaining historical-replay work are distinguished in the serving results document:

1. On synthetic events, compare the model's strict-before history helper with Feast historical retrieval at first events, repeated locations, unseen pairs and equal-time events. Require exact count and flag equality.
2. On selected training/validation query rows, compare directly computed histories with Feast point-in-time values using the same frozen state and timestamp. No final-test scoring.
3. After a complete materialisation in a temporary store, compare online counts against direct training-history calculations for future queries. Include cold starts, an unseen pair and repeated requests.
4. Demonstrate that a historical request is rejected by the online route rather than receiving future history. Verify timezone and timestamp-precision handling.
5. Test missing known entities, unavailable stores, invalid counts, partial materialisation and mismatched versions. No successful response may silently hide a store failure.
6. For a synthetic valid request, require identical assembled features and score from direct saved-pipeline inference and the serving adapter. The threshold decision must use the unchanged `>=` rule.
7. Verify request validation, label/history-field rejection, unknown categories, allowed nulls and duplicate-ID handling. A repeated identical request must not mutate state; a conflicting reuse of its ID must be rejected.
8. Confirm that logs and responses identify the exact model, protocol and feature-state build. Loading a registered model must not trigger training or test evaluation.

The existing 50-row Stage 09 check demonstrated historical/online equality at a shared latest-state query time. It did not execute all of the acceptance checks above or validate a running API.

## Next implementation and expected outcome

The serving-feature adapter, tests, isolated state bundle and verified manifest are implemented. The adapter remains independent of FastAPI. It supports future requests against the frozen latest training state, not arbitrary historical replay.

FastAPI and local versioned prediction logging are implemented. Explicit registry integration, monitoring, Streamlit and Docker remain next steps. Historical replay and broader training/validation point-in-time parity evidence remain separate work; no production champion promotion is implied.

The expected result is a verified mapping from a transaction request to the model's 13 predictors, with explicit timing and error rules. It is not a new predictive score, a new model-selection result, or an improvement in fraud detection.
