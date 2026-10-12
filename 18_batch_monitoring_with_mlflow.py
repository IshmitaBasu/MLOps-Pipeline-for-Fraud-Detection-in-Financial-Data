"""Create or apply research batch-monitoring profiles; never retrain automatically."""

# %% Imports and fixed research references
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd

from final_test_evaluation_utils import FROZEN_THRESHOLD
from fraud_modeling_utils import PROJECT_DIR, configure_mlflow
from monitoring_utils import (
    MonitoringThresholds,
    build_reference_profile,
    compare_with_reference,
    labelled_performance,
    load_prediction_ledger,
    monitoring_recommendation,
    validate_scored_batch,
)
from project_io_utils import save_json_atomically, sha256_file
from research_serving_utils import FINAL_RUN_ID

EXPERIMENT = "financial-fraud-stage-18-batch-monitoring"
DEFAULT_LEDGER = PROJECT_DIR / "serving_artifacts/research_v1/predictions.sqlite"


# %% Deterministic software demonstration, not thesis performance evidence
def smoke_frames(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def make(rows: int, shifted: bool) -> pd.DataFrame:
        score = np.clip(rng.beta(2.2 + 0.8 * shifted, 3.0, rows), 0, 1)
        frame = pd.DataFrame(
            {
                "transaction_id": [f"SMOKE-{int(shifted)}-{i}" for i in range(rows)],
                "transaction_type": rng.choice(["payment", "transfer", "deposit", "withdrawal"], rows),
                "merchant_category": rng.choice(["retail", "travel", "food", "services"], rows),
                "location": rng.choice(["Berlin", "Hamburg", "Munich", "Frankfurt"], rows),
                "device_used": rng.choice(["mobile", "web", "atm", "pos"], rows),
                "payment_channel": rng.choice(["card", "bank", "ACH", "cash"], rows),
                "amount": rng.lognormal(5.0 + 0.45 * shifted, 1.0, rows),
                "time_since_last_transaction": rng.normal(10 + 8 * shifted, 5, rows),
                "spending_deviation_score": rng.normal(0.5 * shifted, 1, rows),
                "velocity_score": rng.integers(1 + 3 * shifted, 20, rows),
                "geo_anomaly_score": rng.uniform(0.2 * shifted, 1, rows),
                "sender_prior_transaction_count": rng.poisson(4 + 3 * shifted, rows),
                "sender_location_prior_transaction_count": rng.poisson(2, rows),
                "is_new_location_for_returning_sender": rng.binomial(1, 0.25 + 0.20 * shifted, rows),
                "score": score,
            }
        )
        probability = np.clip(0.02 + 0.12 * score + 0.03 * shifted, 0, 0.9)
        frame["is_fraud"] = rng.binomial(1, probability)
        return frame

    return make(10_000, False), make(5_000, True)


def versions() -> dict[str, str]:
    return {name: version(name) for name in ("mlflow", "pandas", "numpy", "scikit-learn")}


# %% Profile and monitoring workflows
def create_profile(frame: pd.DataFrame, output: Path, source: str, source_sha256: str) -> dict:
    profile = build_reference_profile(
        frame,
        threshold=FROZEN_THRESHOLD,
        model_run_id=FINAL_RUN_ID,
        state_version="explicit_batch_reference",
        thresholds=MonitoringThresholds(),
    )
    profile["source"] = source
    profile["source_sha256"] = source_sha256
    save_json_atomically(profile, output)
    return profile


def monitor(frame: pd.DataFrame, profile: dict) -> tuple[dict, pd.DataFrame]:
    if profile["model_run_id"] != FINAL_RUN_ID or profile["decision_threshold"] != FROZEN_THRESHOLD:
        raise ValueError("Monitoring reference does not match the frozen research model and threshold.")
    quality = validate_scored_batch(frame)
    drift = compare_with_reference(frame, profile)
    thresholds = MonitoringThresholds(**profile["thresholds"])
    performance = labelled_performance(frame, FROZEN_THRESHOLD, thresholds)
    recommendation = monitoring_recommendation(profile, frame, drift, quality, performance)
    report = {
        "scope": "research_batch_monitoring",
        "model_run_id": FINAL_RUN_ID,
        "decision_threshold": FROZEN_THRESHOLD,
        "rows": int(len(frame)),
        "quality": quality,
        "performance": performance,
        "decision": recommendation,
        "automatic_retraining": False,
        "test_partition_used": False,
        "production_approval": False,
    }
    return report, drift


def tracked_source_files() -> list[Path]:
    names = (
        "18_batch_monitoring_with_mlflow.py",
        "monitoring_utils.py",
        "tests/test_batch_monitoring.py",
        "Markdown files/18_batch_monitoring_plan.md",
        "Markdown files/18_batch_monitoring_implementation_guide.md",
        "requirements.txt",
    )
    return [PROJECT_DIR / name for name in names]


# %% CLI and MLflow evidence
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "create-reference", "monitor-csv", "monitor-ledger"), default="smoke"
    )
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--reference-profile", type=Path)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--input-scope",
        choices=("training_reference", "operational_reference", "operational_batch", "synthetic_fixture"),
        help="Explicit provenance declaration; validation and consumed test rows are not accepted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow(EXPERIMENT)
    with TemporaryDirectory() as temporary:
        output = args.output or Path(temporary)
        if args.output and output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to replace files in a non-empty monitoring output directory.")
        output.mkdir(parents=True, exist_ok=True)
        if args.mode == "smoke":
            reference, current = smoke_frames()
            profile_path = output / "monitoring_reference_profile.json"
            profile = create_profile(
                reference, profile_path, "deterministic_synthetic_smoke_reference", "deterministic_fixture"
            )
            evidence_scope = "smoke_test"
        elif args.mode == "create-reference":
            if args.input_csv is None:
                raise ValueError("--input-csv is required to create a monitoring reference.")
            if args.input_scope not in {"training_reference", "operational_reference"}:
                raise ValueError("Create-reference requires --input-scope training_reference or operational_reference.")
            reference = pd.read_csv(args.input_csv)
            profile_path = output / "monitoring_reference_profile.json"
            profile = create_profile(reference, profile_path, str(args.input_csv), sha256_file(args.input_csv))
            profile["source_scope"] = args.input_scope
            save_json_atomically(profile, profile_path)
            current = None
            evidence_scope = "explicit_reference_batch"
        else:
            if args.reference_profile is None:
                raise ValueError("--reference-profile is required for monitoring.")
            profile_path = args.reference_profile
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if args.mode == "monitor-csv":
                if args.input_csv is None:
                    raise ValueError("--input-csv is required for monitor-csv.")
                if args.input_scope != "operational_batch":
                    raise ValueError("monitor-csv requires --input-scope operational_batch.")
                current = pd.read_csv(args.input_csv)
                source = str(args.input_csv)
            else:
                current = load_prediction_ledger(args.ledger_path)
                source = str(args.ledger_path)
            evidence_scope = "monitored_batch"
            save_json_atomically(profile, output / "reference_profile_snapshot.json")
        with mlflow.start_run(run_name=f"stage_18_{args.mode.replace('-', '_')}") as active:
            mlflow.log_params(
                {
                    "mode": args.mode,
                    "model_run_id": FINAL_RUN_ID,
                    "threshold": FROZEN_THRESHOLD,
                    "automatic_retraining": False,
                    "test_partition_used": False,
                }
            )
            mlflow.set_tags(
                {"stage": "18_batch_monitoring", "evidence_scope": evidence_scope, "production_approved": "false"}
            )
            for source_file in tracked_source_files():
                mlflow.log_artifact(str(source_file), artifact_path="source_snapshot")
            save_json_atomically(versions(), output / "library_versions.json")
            if current is None:
                result = {
                    "status": "reference_created",
                    "rows": profile["rows"],
                    "reference_sha256": sha256_file(profile_path),
                    "automatic_retraining": False,
                }
                save_json_atomically(result, output / "monitoring_reference_receipt.json")
                mlflow.log_metric("reference_rows", profile["rows"])
                print(f"Monitoring reference created from {profile['rows']:,} explicit rows.")
            else:
                report, drift = monitor(current, profile)
                if args.mode != "smoke":
                    report["batch_source"] = source
                save_json_atomically(report, output / "monitoring_report.json")
                drift.to_csv(output / "feature_drift.csv", index=False, lineterminator="\n")
                mlflow.log_metrics(
                    {
                        "batch_rows": report["rows"],
                        "maximum_psi": float(drift.psi.max()),
                        "current_alert_rate": report["decision"]["current_alert_rate"],
                    }
                )
                print(f"Monitoring recommendation: {report['decision']['recommendation']}.")
                print(f"Maximum PSI: {drift.psi.max():.6f}.")
                print(f"Delayed-label performance: {report['performance']['status']}.")
                print("Automatic retraining was not triggered.")
            mlflow.log_artifacts(str(output))
            print(f"Stage 18 complete. MLflow run: {active.info.run_id}")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Experiment: {EXPERIMENT}")
            if args.mode == "smoke":
                print("This was a synthetic smoke run and is not thesis performance evidence.")


if __name__ == "__main__":
    main()
