"""Prepare training-only monitoring-reference history; model scoring comes later."""

# %% Imports and fixed evidence references
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import mlflow

# %% Repository bootstrap for direct script and VS Code cell execution
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fraud_detection.common.modeling import PROJECT_DIR, configure_mlflow
from fraud_detection.monitoring.reference import history_preparation_receipt, prepare_reference_history
from fraud_detection.common.io import sha256_file

EXPERIMENT = "financial-fraud-stage-18-batch-monitoring"
RAW_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
REFERENCE_PATH = PROJECT_DIR / "mlartifacts/5920c17849574c11b4854c3c5290bc41/artifacts/voting_full_data_summary.json"


# %% CLI and immutable source checks
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--raw-data-path", type=Path, default=RAW_PATH)
    return parser.parse_args()


def source_hash_for_mode(path: Path, mode: str) -> str:
    if mode == "smoke":
        return "not_calculated_smoke_only"
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    expected = reference["raw_dataset_info"]["dataset_sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("Raw source checksum differs from the frozen development source.")
    return actual


# %% Training-history preparation and MLflow receipt
def main() -> None:
    args = parse_args()
    scope = "smoke_test" if args.mode == "smoke" else "full_data"
    tracking_uri = configure_mlflow(EXPERIMENT)
    print(f"Preparing Stage 18 {args.mode} training history for the monitoring reference.", flush=True)
    source_hash = source_hash_for_mode(args.raw_data_path, args.mode)
    with mlflow.start_run(run_name=f"monitoring_reference_history_{scope}") as active:
        mlflow.set_tags(
            {
                "stage": "18_batch_monitoring",
                "run_role": "monitoring_reference_history_preparation",
                "evidence_scope": scope,
                "training_only": "true",
                "model_scored": "false",
                "reference_profile_created": "false",
                "test_analysed": "false",
            }
        )
        prepared, lineage = prepare_reference_history(args.raw_data_path, args.mode)
        lineage["source_sha256"] = source_hash
        if args.mode == "full" and sha256_file(args.raw_data_path) != source_hash:
            raise ValueError("Raw source bytes changed during reference-history preparation.")
        receipt = history_preparation_receipt(prepared, lineage)
        mlflow.log_dict(lineage, "monitoring_reference_history_lineage.json")
        mlflow.log_dict(receipt, "monitoring_reference_history_receipt.json")
        mlflow.log_dict(
            {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "mlflow")}, "library_versions.json"
        )
        mlflow.log_metrics(
            {
                "training_rows": receipt["training_rows"],
                "sender_prior_count_max": receipt["sender_prior_count_max"],
                "sender_location_prior_count_max": receipt["sender_location_prior_count_max"],
                "new_location_rows": receipt["new_location_rows"],
            }
        )
        for name in (
            "stages/18_prepare_monitoring_reference.py",
            "fraud_detection/monitoring/reference.py",
            "fraud_detection/common/training_data.py",
            "fraud_detection/diagnostics/predictive_signal.py",
            "fraud_detection/features/sender_location_history.py",
            "requirements.txt",
        ):
            mlflow.log_artifact(str(PROJECT_DIR / name), artifact_path="source_snapshot")
        print(f"Prepared {receipt['training_rows']:,} training rows with strictly earlier history.", flush=True)
        print("No model was loaded or scored; no reference profile was frozen.", flush=True)
        print("Validation and test features were not retained.", flush=True)
        print(f"Stage 18 history-preparation run: {active.info.run_id}", flush=True)
        print(f"Tracking URI: {tracking_uri}\nExperiment: {EXPERIMENT}", flush=True)
        if args.mode == "smoke":
            print("This was a smoke run and is not the canonical monitoring reference.", flush=True)


# %% Executable entry point
if __name__ == "__main__":
    main()
