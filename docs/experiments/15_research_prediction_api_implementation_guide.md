# Local research prediction API

## What this adds

The saved LightGBM pipeline can now receive one prepared transaction, retrieve frozen sender history from Feast, and return an uncalibrated score and research review flag. The model is loaded from its evaluated artifact; it is not trained again.

The [feature contract](15_model_serving_feature_contract.md) explains request fields and timing. The [results document](15_research_prediction_api_results.md) records completed checks and limitations. These documents serve different purposes; predictive evidence remains in the Stage 14 report.

| File | Responsibility |
| --- | --- |
| `stages/15_research_prediction_api.py` | Cell-based entry point: prepare the isolated bundle, verify readiness, or start the local server |
| `fraud_detection/serving/service.py` | Artifact verification, disk-based entity/count index, strict Feast status checks, feature assembly and persistent prediction ledger |
| `fraud_detection/serving/api.py` | Request validation, application startup, readiness endpoints and `/predict` |
| `tests/test_research_prediction_api.py` | Synthetic request, failure, timing, duplicate and Feast integration checks |

## Local bundle and safeguards

`serving_artifacts/research_v1/` holds the manifest, exact model copy, Feast registry copy, online-store backup, expected-count index, feature metadata and prediction ledger. The folder is Git-ignored. Preparation reads existing snapshots and copies existing state; it does not rebuild training data or materialise into the canonical store.

The current isolated bundle occupies approximately 1.51 GB, mostly the online-store copy and disk index. This additional local storage is intentional; it protects the canonical Feast state from serving operations. Restore or regenerate the ignored bundle when moving the project to another machine rather than committing it.

Source snapshot hashes, split identity, feature schema and recorded model library versions are verified. Both copied online tables must have the expected entity counts. Each requested known entity must return the exact indexed training-history count with a PRESENT status; genuinely unseen keys may receive zero only with NOT_FOUND status. Startup checks are followed by immutable-file fingerprints and known-state health probes. No state updates occur during prediction.

The manifest pins the actual latest snapshot boundary. Online timestamps must be strictly after it. Historical replay is not implemented as an endpoint; historical requests are rejected instead of receiving future state.

Preparation refuses to overwrite an existing bundle. An interrupted bundle is retained for diagnosis; do not delete it blindly or rerun final evaluation. Prepare a separately named local bundle after diagnosing the error. A manifest is not written until source/index/copy checks pass, and readiness failure during preparation removes that generated manifest.

The manifest is a local integrity record, not a cryptographic signature or an approval process. Load only trusted local artifacts: joblib deserialisation is unsafe for untrusted uploads. There is no upload endpoint.

## Commands

Run these from the thesis workspace. The current `research_v1` bundle has already been prepared and verified; do not repeat preparation for that existing directory.

Verify it without making a prediction:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\15_research_prediction_api.py" --mode verify
```

Start the server when needed:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\15_research_prediction_api.py" --mode serve
```

Open `http://127.0.0.1:8000/docs` to inspect the request schema and try `/predict`. `/health/live` checks that the application is running; `/health/ready` checks verified model/history readiness. Stop the foreground server with Ctrl+C.

For a fresh checkout with the required ignored local model/state artifacts restored, prepare a new bundle once:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\15_research_prediction_api.py" --mode prepare
```

The process deliberately uses one worker and binds to `127.0.0.1`. It has no authentication and is not intended for public, LAN or banking exposure. No background server is left running by the implementation checks.

## Try a request

The JSON example in the feature contract can be pasted into Swagger's `/predict` request. It contains made-up values, not a real fraud case. The API still needs the original upstream predictors; it cannot infer spending-deviation or geo-anomaly scores from amount and sender alone.

The response includes `score`, `decision`, the frozen threshold, model run, contract/state versions, timestamp, cold-start flag and processing latency. `not_flagged` does not establish that a transaction is legitimate. `review` does not establish fraud.

Repeating an identical transaction ID and payload returns the stored response without rescoring or increasing history counts, provided state remains healthy. Reusing that ID for different data returns HTTP 409. Malformed inputs return HTTP 422; history/model failures return HTTP 503 without a fabricated score. History features and labels supplied by a client are rejected.

## Tests and privacy

Run the complete suite from the code repository, because two existing experiment tests use repository-relative paths:

```powershell
Push-Location ".\Code snippets"
try {
    & "..\masters_thesis\Scripts\python.exe" -m unittest discover -s tests -v
} finally {
    Pop-Location
}
```

Successful requests, assembled predictors, responses and model/state references are stored in local `predictions.sqlite`. Validation failures are logged without their raw invalid payloads. Successful request records can contain account identifiers and transaction values: keep the ignored bundle private, do not put the ledger on GitHub, and use invented inputs for demonstrations. Automatic retention/deletion and authentication are not implemented; the ledger persists until explicitly managed.

Standalone research-model registration is complete; registry-backed API loading, monitoring, Streamlit, Docker and a formal serving-latency evaluation remain separate work. The [future-version procedure](16_research_model_registration.md#registering-a-future-model-version) covers candidate evidence, feature/runtime changes, explicit serving switches, rollback and cross-version ledger boundaries, distinguishing implemented guards from planned mechanisms. Any future model replacement requires explicit provenance and a separately reviewed serving contract; this API never retrains or chooses a new threshold.
