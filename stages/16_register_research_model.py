"""Register or verify the exact evaluated research model; never train or evaluate data."""

# %% Imports and fixed research references
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

# %% Repository bootstrap for direct script and VS Code cell execution
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fraud_detection.evaluation.final_test import FROZEN_THRESHOLD
from fraud_detection.common.modeling import CATEGORICAL_FEATURES, configure_mlflow
from fraud_detection.modeling.model_families import FROZEN_FULL_SPLIT_HASH
from fraud_detection.modeling.oversampling import FEATURE_SET_NAME, MODEL_FEATURES
from fraud_detection.common.io import sha256_file
from fraud_detection.serving.registered_model import FrozenResearchModel
from fraud_detection.paths import REPOSITORY_ROOT

PROJECT_DIR = REPOSITORY_ROOT
FINAL_RUN_ID = "37b41018f28646598344fbcfe4ea4db6"
MODEL_NAME = "financial_fraud_lightgbm_research"
EXPERIMENT = "financial-fraud-stage-16-research-model-registration"
PACKAGES = ("mlflow", "scikit-learn", "lightgbm", "joblib", "pandas", "numpy")


# %% Source preflight: require completed evidence and byte identity with the serving export
def validate_source(project_dir: Path) -> tuple[Path, dict]:
    artifacts = project_dir / "mlartifacts" / FINAL_RUN_ID / "artifacts"
    result = json.loads((artifacts / "final_test_evaluation.json").read_text(encoding="utf-8"))
    protocol = json.loads((artifacts / "frozen_evaluation_protocol.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (project_dir / "final_test_evaluation" / f"{FROZEN_FULL_SPLIT_HASH}.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((project_dir / "serving_artifacts/research_v1/manifest.json").read_text(encoding="utf-8"))
    if (
        result.get("run_id") != FINAL_RUN_ID
        or result.get("evidence_scope") != "full_data"
        or result.get("canonical_test_evaluated") is not True
        or result.get("test_split_evaluated") is not True
        or receipt.get("status") != "complete"
        or receipt.get("run_id") != FINAL_RUN_ID
        or result.get("protocol") != protocol
        or receipt.get("protocol") != protocol
        or protocol.get("threshold") != FROZEN_THRESHOLD
        or protocol.get("split_assignment_sha256") != FROZEN_FULL_SPLIT_HASH
        or protocol.get("feature_set") != FEATURE_SET_NAME
        or protocol.get("configuration") != "lgbm_reference"
        or manifest.get("model_run_id") != FINAL_RUN_ID
        or manifest.get("threshold") != FROZEN_THRESHOLD
        or manifest.get("split_sha256") != FROZEN_FULL_SPLIT_HASH
        or manifest.get("model_features") != MODEL_FEATURES
        or manifest.get("production_approved") is not False
    ):
        raise ValueError("Registration source does not match the completed frozen research protocol.")
    versions = json.loads((artifacts / "library_versions.json").read_text(encoding="utf-8"))
    for package in PACKAGES:
        if version(package) != versions[package]:
            raise ValueError(f"Registration runtime differs from evaluated runtime: {package}.")
    model_path = artifacts / "evaluated_model/evaluated_lightgbm_pipeline.joblib"
    model_hash = sha256_file(model_path)
    if (
        model_hash != manifest["files"]["model.joblib"]
        or sha256_file(project_dir / "serving_artifacts/research_v1/model.joblib") != model_hash
    ):
        raise ValueError("Evaluated pipeline and verified serving export differ.")
    contract = {
        "evaluated_run_id": FINAL_RUN_ID,
        "source_model_sha256": model_hash,
        "threshold": FROZEN_THRESHOLD,
        "model_features": MODEL_FEATURES,
        "feature_set": FEATURE_SET_NAME,
        "split_sha256": FROZEN_FULL_SPLIT_HASH,
        "production_approved": False,
        "probabilities_calibrated": False,
        "usage": "research_only",
        "feature_input": "already_computed_features; no Feast lookup inside the registered model",
        "history_population": "strictly_earlier_training_transactions_only",
        "protocol": protocol,
        "library_versions": versions,
        "evidence": {
            name: sha256_file(artifacts / name)
            for name in ("final_test_evaluation.json", "frozen_evaluation_protocol.json", "library_versions.json")
        },
    }
    return model_path, contract


# %% Registry identity and read-only loading checks
def matching_versions(client, contract: dict) -> list:
    matches = []
    token = None
    while True:
        page = client.search_model_versions(f"name = '{MODEL_NAME}'", page_token=token)
        for entry in page:
            if (
                entry.tags.get("evaluated_run_id") == FINAL_RUN_ID
                and entry.tags.get("source_model_sha256") == contract["source_model_sha256"]
            ):
                matches.append(entry)
        token = page.token
        if not token:
            break
    if len(matches) > 1:
        raise ValueError("Multiple matching versions exist; inspect the registry before proceeding.")
    return matches


def verify_version(client, entry, contract: dict) -> dict:
    if str(entry.status) != "READY" or entry.tags.get("production_approved") != "false":
        raise ValueError("Registered version is not ready or lacks the research-only approval guard.")
    uri = f"models:/{MODEL_NAME}/{entry.version}"
    loaded = mlflow.pyfunc.load_model(uri)
    wrapper = loaded.unwrap_python_model()
    if wrapper.source_sha256 != contract["source_model_sha256"] or wrapper.contract != contract:
        raise ValueError("Registry artifact differs from the verified evaluated source contract.")
    return {
        "registered_model": MODEL_NAME,
        "version": str(entry.version),
        "model_uri": uri,
        "registration_run_id": entry.run_id,
        "evaluated_run_id": FINAL_RUN_ID,
        "source_model_sha256": wrapper.source_sha256,
        "production_approved": False,
        "threshold": FROZEN_THRESHOLD,
        "verification": "load_and_checksum_only; no predictions",
        "api_changed": False,
        "test_evaluation_repeated": False,
    }


# %% Explicit registration; reruns verify the existing matching version
def register_model(client, model_path: Path, contract: dict) -> dict:
    matches = matching_versions(client, contract)
    if matches:
        return verify_version(client, matches[0], contract)
    configure_mlflow(EXPERIMENT)
    tags = {
        "evaluated_run_id": FINAL_RUN_ID,
        "source_model_sha256": contract["source_model_sha256"],
        "usage": "research_only",
        "production_approved": "false",
        "stage": "16_model_registration",
        "threshold": str(FROZEN_THRESHOLD),
        "feature_set": FEATURE_SET_NAME,
        "split_sha256": FROZEN_FULL_SPLIT_HASH,
        "retrained": "false",
        "test_evaluation_repeated": "false",
    }
    with mlflow.start_run(run_name="register_frozen_lightgbm_research") as run:
        mlflow.set_tags(tags)
        mlflow.log_dict(contract, "registration_contract.json")
        mlflow.log_params(
            {
                "source_run": FINAL_RUN_ID,
                "registered_model_name": MODEL_NAME,
                "source_model_sha256": contract["source_model_sha256"],
                "threshold": FROZEN_THRESHOLD,
            }
        )
        for name in ("final_test_evaluation.json", "frozen_evaluation_protocol.json", "library_versions.json"):
            mlflow.log_artifact(str(model_path.parent.parent / name), artifact_path="recorded_evidence")
        for name in (
            "stages/16_register_research_model.py",
            "fraud_detection/serving/registered_model.py",
            "requirements.txt",
        ):
            mlflow.log_artifact(str(PROJECT_DIR / name), artifact_path="source_snapshot")
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec("string" if feature in CATEGORICAL_FEATURES else "double", feature)
                    for feature in MODEL_FEATURES
                ]
            ),
            outputs=Schema([ColSpec("double", "score"), ColSpec("string", "decision")]),
        )
        with TemporaryDirectory(prefix="fraud_registration_") as temporary:
            contract_path = Path(temporary) / "registration_contract.json"
            contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
            info = mlflow.pyfunc.log_model(
                name="frozen_research_model",
                python_model=FrozenResearchModel(),
                artifacts={"evaluated_pipeline": str(model_path), "contract": str(contract_path)},
                code_paths=[str(PROJECT_DIR / "fraud_detection")],
                signature=signature,
                pip_requirements=[f"{package}=={version(package)}" for package in PACKAGES],
                metadata=tags,
            )
        entry = mlflow.register_model(info.model_uri, MODEL_NAME, tags=tags)
        client.update_model_version(
            MODEL_NAME,
            entry.version,
            description="Frozen evaluated LightGBM research model. Uncalibrated scores; weak predictive quality. "
            "Not production approved. No training, test scoring, alias assignment or API switch during registration.",
        )
        report = verify_version(client, client.get_model_version(MODEL_NAME, entry.version), contract)
        mlflow.log_dict(report, "model_registration_receipt.json")
    return report


# %% Cell-aware command-line entry point
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "register"), default="verify")
    args = parser.parse_args()
    uri = f"sqlite:///{(PROJECT_DIR / 'mlflow_tracking.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = mlflow.tracking.MlflowClient()
    model_path, contract = validate_source(PROJECT_DIR)
    if args.mode == "register":
        report = register_model(client, model_path, contract)
    else:
        matches = matching_versions(client, contract)
        if not matches:
            raise ValueError("No matching research version exists. Use --mode register to create it explicitly.")
        report = verify_version(client, matches[0], contract)
    print(json.dumps(report, indent=2))
    print("No retraining, predictions, test evaluation, promotion or API bundle changes.")


if __name__ == "__main__":
    main()
