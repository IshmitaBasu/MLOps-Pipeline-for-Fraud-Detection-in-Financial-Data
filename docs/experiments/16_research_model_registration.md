# Research model registration

## Purpose

Registration gives the evaluated LightGBM model a named, traceable MLflow version. It does not improve predictions or approve the model for production. The recorded low precision and high false-alert burden remain limitations.

For example, `models:/financial_fraud_lightgbm_research/1` identifies one exact research package. A later model needs a separate version; it must not overwrite the evaluated source or inherit its test evidence.

## Files and controls

- `stages/16_register_research_model.py`: cell-based registration and read-only verification runner. Defaults to verification; creation requires `--mode register`.
- `fraud_detection/serving/registered_model.py`: portable MLflow Python-function wrapper. It loads the original evaluated joblib bytes after checking their SHA-256 hash. It does not fit the pipeline or retrieve features.
- `tests/test_research_model_registration.py`: synthetic tests for source integrity, fixed-threshold behaviour, version lookup, and repeat-registration safeguards.

The source is final-evaluation run `37b41018f28646598344fbcfe4ea4db6`. Registration requires its completed receipt and frozen protocol, checks the evaluated runtime versions, and verifies byte identity with the Stage 15 export. The numerical threshold remains `0.5226022799524157`. Registration and verification load the package but make no predictions or test-set calls.

The registered package accepts the thirteen already-computed model features, including the three training-history features. It returns an uncalibrated score and the frozen `flagged` / `not_flagged` decision. It is not a replacement for the raw-transaction API: Feast retrieval and request validation still belong to the serving layer. The model signature describes string categories and double-valued numerical inputs; callers should supply numeric feature columns as floating point.

## Commands

From the thesis root:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\16_register_research_model.py" --mode register
```

Re-running this command verifies the existing matching version instead of creating a duplicate. An interrupted workflow that created a registry version can be verified on retry; multiple matching versions require manual inspection. Registration is intended to run serially, not through concurrent writers.

Read-only package verification:

```powershell
.\masters_thesis\Scripts\python.exe ".\Code snippets\stages\16_register_research_model.py" --mode verify
```

The registry shares the existing local MLflow SQLite tracking backend. The registration experiment is `financial-fraud-stage-16-research-model-registration`. The new run retains the source hash, input/output signature, pinned model runtime, original evaluation evidence and hashes, frozen contract, source snapshot, and registration receipt. Historical evidence is copied and linked, not changed or relabelled as a new evaluation. No alias or deployment stage is assigned. Version tags explicitly set `usage=research_only` and `production_approved=false`.

This follows MLflow's documented [Python-function packaging](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html) and [model registry workflow](https://www.mlflow.org/docs/latest/ml/model-registry/workflow/).

## Execution results

Completed on 8 October 2026 in the existing local SQLite-backed MLflow registry.

| Item | Recorded value |
| --- | --- |
| Registered model | `financial_fraud_lightgbm_research` |
| Version | `1` |
| Pinned model URI | `models:/financial_fraud_lightgbm_research/1` |
| Registration run | `90f3f5b7f3f04a27abd5c409396f25e6` |
| Original evaluated run | `37b41018f28646598344fbcfe4ea4db6` |
| Original pipeline SHA-256 | `f89c50b5fba382a1312fa60f57dab7d305cce4bf711090fb10c28c970226b755` |
| Fixed threshold | `0.5226022799524157` |
| Usage / production approval | Research only / false |
| Aliases or production promotion | None |
| Model fitting / predictions / repeated test evaluation | None |
| Existing API bundle changed | No |

Verification successfully loaded version 1, checked the embedded original joblib checksum, matched the complete frozen contract, and checked fitted feature names and binary class order without making predictions. The registration run saves `model_registration_receipt.json`. The original evaluation run and receipt remain unchanged.

A separate `--mode verify` execution also succeeded and resolved the same version and registration run. All 102 repository tests passed, including five new registration tests. README and the architecture diagram now distinguish completed research registration from pending registry-backed API loading.

MLflow warned about Python-object serialization and the absence of an input example. Only trusted local artifacts are loaded. The input/output signature was supplied explicitly; no input example was used because registration was designed not to make prediction calls. These warnings did not prevent successful registration or verification.

Observation: model provenance and version management are now implemented independently of predictive quality. Registration preserves and exposes the existing weak results; it does not make the model suitable for operational fraud detection. Look for the model under **Models** in the local MLflow interface; its registration run is under `financial-fraud-stage-16-research-model-registration`.

## Registering a future model version

This is a planned procedure, not an implemented automatic pipeline. Script 16 currently accepts only the recorded Stage 14 LightGBM source, frozen threshold, runtime, and existing serving export. Re-running it verifies version 1. It does not discover new training runs, accept an arbitrary replacement, or register version 2 automatically.

### Procedure

1. **Save the candidate separately.** Keep its fitted pipeline, training configuration, data/split identity, feature definitions, preprocessing, dependencies, and validation results in a new MLflow run. Do not overwrite version 1's source files or evaluation receipt.
2. **Define the evaluation evidence.** Freeze candidate and decision-rule choices using development data. The consumed Stage 14 test partition must not become another tuning or repeated final-evaluation tool. A changed model cannot inherit version 1's test results; establish an appropriate independent evaluation plan before making new final-performance claims. A validation-only candidate can be recorded as such, but must not be described as independently test-evaluated.
3. **Create the candidate's contract.** Record its input names/types, missing-value rules, feature-state requirements, fitted preprocessing, class ordering, output meaning, threshold, calibration status, runtime versions, source hashes, and evidence references. Retain research-only status unless a separate approval process establishes otherwise.
4. **Extend registration explicitly.** Implement and test a workflow that accepts a specified candidate run and contract rather than editing the historical frozen constants to bypass their guards. Validate the candidate against its own provenance and evaluation status. Identify an existing package by the complete model-and-contract identity, not just the weights: changed thresholds or feature semantics must not silently reuse an older version.
5. **Register the verified package deliberately.** For a compatible successor within this LightGBM research model, use the existing registered name. MLflow assigns the next available version; do not assume it will necessarily be 2 or use an unpinned latest-version reference. Record the actual version, registration run, source hash, contract, and evidence status. Registration alone grants no deployment approval.
6. **Verify before switching serving.** Load the explicitly numbered version in a separate candidate deployment. Check hashes, dependencies, input/output compatibility, feature retrieval, fixed-rule behaviour, readiness failures, and direct-model/API parity using synthetic fixtures and appropriately scoped development data, not the consumed test partition. Preserve the currently working bundle while doing this.
7. **Switch deliberately and retain rollback.** Only after the checks pass, select the exact version and its matching contract/state build for serving. Responses and prediction logs must identify the model version, threshold, and feature-state version. Preserve the previous package, runtime, state bundle, and logs so rollback restores the complete compatible deployment, not just older model weights.

Example: a new LightGBM candidate keeps the same thirteen inputs but uses a newly selected validation threshold. Its fitted artifact, threshold, and supporting evidence form a new package. Registering that package does not change requests served by version 1; an explicit, verified serving switch is still required.

### Changes and failure cases

| Situation | Required handling | Current implementation |
| --- | --- | --- |
| Current registration command is repeated | Verify the existing matching version; do not create a duplicate | Implemented for the frozen source |
| Registration was interrupted or duplicate matching versions exist | Inspect existing state; retry verification where possible; stop on ambiguity | Implemented guards; serial execution required |
| Model weights change | New source identity, candidate evidence, package and explicit registration | Future-candidate workflow not implemented |
| Threshold changes but weights stay the same | Treat the decision rule as a changed package; validate it on development data and preserve the old rule | Current script rejects a changed threshold; future package identity handling pending |
| Feature names, types, history population, timing or meaning change | Version the feature/serving contract and, where necessary, Feast definitions/state; verify training-serving consistency | Existing fixed-contract guards implemented; migration workflow pending |
| Runtime dependencies change | Validate a separate compatible runtime; do not assume the old serialized model still loads correctly | Evaluated-runtime checks implemented; migration workflow pending |
| Dataset or model family changes | Preserve distinct provenance and evaluation evidence; choose an appropriately named registry model if the LightGBM-specific name no longer fits | Not automatic; future workflow pending |
| Registry, artifact or feature-state verification fails | Do not select or serve the new candidate; keep the existing healthy deployment or report unavailable | Current registration integrity/load guards implemented; registry-backed serving failure policy pending |
| A serving switch or rollback occurs | Restore the matching model, threshold, contract, runtime and state; keep logs attributable to their original deployment | Planned; API switching and rollback are not yet implemented |

Prediction IDs and cached responses also need a defined deployment boundary. A response generated under one version must not silently appear to have been generated by another. The future switch should preserve old ledger provenance and explicitly define whether request IDs are scoped to a deployment or shared across versions; current Stage 15 has only one frozen deployment.

### What remains to be built

The future-candidate registration interface, complete package-identity deduplication, registry-backed API loading, feature/runtime migrations, explicit switch/rollback mechanism, and cross-version ledger policy remain unimplemented. This procedure documents their requirements; it does not claim those capabilities already exist. Changes to the current model or predictive analysis therefore do not automatically trigger registration, retraining, deployment, or promotion.

## Serving boundary

Registration alone does not change the existing API or its bundle. Stage 15 still loads the verified exported joblib file; its historical `model_registered=false` metadata has not been rewritten. This flag does not yet reflect the later standalone registry record. A future serving update should explicitly pin and verify a registry version, then accurately expose that version in readiness and prediction metadata. Do not silently switch to a latest version or overwrite the current frozen bundle.

Next: review the registry entry in MLflow. Registry-backed API loading is a separate implementation step; Docker, Streamlit, and monitoring remain later work.
