"""Training-only descriptive diagnostics; no model fitting, validation analysis or test reuse."""

# %% Imports and fixed scope
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow

# %% Repository bootstrap for direct script and VS Code cell execution
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fraud_detection.common.modeling import PROJECT_DIR, configure_mlflow
from fraud_detection.common.training_data import load_training_data
from fraud_detection.diagnostics.predictive_quality import TARGET_COLUMN
from fraud_detection.diagnostics.predictive_signal import add_training_history, analyse_training, save_signal_plots
from fraud_detection.common.io import sha256_file

EXPERIMENT = "financial-fraud-stage-17-training-predictive-signal"
RAW_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
REFERENCE_PATH = PROJECT_DIR / "mlartifacts/5920c17849574c11b4854c3c5290bc41/artifacts/voting_full_data_summary.json"


# %% Evidence and provenance: unknown generator details remain unknown
def provenance_audit() -> dict:
    names = (
        "docs/experiments/01_initial_raw_data_eda.md",
        "initial_raw_data_eda_report_v1.md",
        "docs/reports/eda_report_v1.md",
    )
    return {
        "local_evidence": [
            {"path": name, "sha256": sha256_file(PROJECT_DIR / name)} for name in names if (PROJECT_DIR / name).exists()
        ],
        "dataset_generation": "Local documentation describes synthetic data; original generator/source specification not verified by this diagnostic.",
        "label_generation": "Unverified: no authoritative fraud-label generation rule is established here.",
        "upstream_score_generation": "Unverified derivation and prediction-time availability of spending_deviation_score, velocity_score and geo_anomaly_score.",
        "missingness_causality": "An association cannot establish why missing values occur or whether it transfers to real transactions.",
        "followup_needed": "Locate the original dataset source and generation/label documentation; request it if unavailable.",
        "no_external_source_inferred": True,
    }


def interpretation_markdown(summary: dict, tables: dict) -> str:
    quality = summary["quality"]
    missing = tables["group_rates"].loc[tables["group_rates"].feature == "time_since_last_transaction_missing"]
    lines = [
        "# Training-only predictive-signal diagnostic",
        "",
        f"Scope: {summary['evidence_scope']}. " "Smoke outputs are software checks, not thesis evidence.",
        "",
        f"Training rows: {quality['training_rows']:,}; fraud rows: {quality['training_fraud_rows']:,}; "
        f"fraud rate: {quality['training_fraud_rate']:.4%}.",
        "",
        "## Missingness evidence",
        "",
        "| Time-since-last status | Rows | Fraud rows | Fraud rate | Nominal 95% interval |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in missing.itertuples():
        lines.append(
            f"| {row.group} | {row.rows:,} | {row.fraud_rows:,} | {row.fraud_rate:.4%} | {row.ci_low:.4%}–{row.ci_high:.4%} |"
        )
    lines += [
        "",
        "This pattern is an association, not proof of leakage, a generation rule, or real-world validity. "
        "Inspect stability_group_rates.csv and missingness_stability.png for training-subset support.",
        "",
        "## Separation and practical size",
        "",
        "categorical_binned_associations.csv reports uncorrected Cramér's V, group support, and rate differences. "
        "Numerical effects include finite-only Pearson/point-biserial correlation and quantile-binned total variation; "
        "these are descriptive summaries, not a learned model or a complete measure of nonlinear information.",
        "",
        "Large samples can make small differences look precise. Do not rank features by significance alone. "
        "Intervals are nominal marginal intervals, not adjusted for repeated comparisons or correlated transactions. "
        "Sparse groups, constant features and missing values need explicit interpretation.",
        "",
        "## Boundaries",
        "",
        "No validation/test feature diagnostics, fitted classifiers, automatic feature selection, registration or serving changes. "
        "Training slice comparisons are not independent evaluation. Null/weak marginal associations cannot rule out all useful interactions.",
        "",
        "## Provenance gaps",
        "",
        "Original generation and fraud-label rules, and the provenance of supplied risk scores, remain unverified. "
        "No dataset replacement or predictive improvement is inferred from this report.",
    ]
    return "\n".join(lines) + "\n"


# %% Tracked workflow with no training/evaluation entry points
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--raw-data-path", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    scope = "full_data" if args.mode == "full" else "smoke_test"
    tracking_uri = configure_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"training_signal_{scope}") as run:
        mlflow.set_tags(
            {
                "stage": "17_training_predictive_signal",
                "evidence_scope": scope,
                "scope": "training_partition_only",
                "validation_analysed": "false",
                "test_analysed": "false",
                "model_trained": "false",
                "eligible_for_model_selection": "false",
            }
        )
        print(f"Running Stage 17 training-only predictive-signal {args.mode} diagnostics.", flush=True)
        if args.mode == "full":
            reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
            expected_source_hash = reference["raw_dataset_info"]["dataset_sha256"]
            source_hash = sha256_file(args.raw_data_path)
            if source_hash != expected_source_hash:
                raise ValueError("Raw source checksum differs from the frozen development source.")
        else:
            source_hash = "not_calculated_smoke_only"
        training, lineage = load_training_data(args.raw_data_path, args.mode)
        lineage["source_sha256"] = source_hash
        if args.mode == "full" and sha256_file(args.raw_data_path) != source_hash:
            raise ValueError("Raw source bytes changed during training extraction.")
        print("Building strictly earlier history from retained training events only.", flush=True)
        training = add_training_history(training)
        summary, tables, _ = analyse_training(training)
        summary.update(
            {
                "evidence_scope": scope,
                "run_id": run.info.run_id,
                "lineage": lineage,
                "provenance": provenance_audit(),
                "registered_model_changed": False,
                "api_changed": False,
            }
        )
        mlflow.log_dict(summary, "training_predictive_signal_summary.json")
        mlflow.log_dict(lineage, "training_diagnostic_lineage.json")
        mlflow.log_dict(summary["provenance"], "dataset_provenance_audit.json")
        mlflow.log_dict(
            {package: version(package) for package in ("pandas", "numpy", "matplotlib", "scikit-learn", "mlflow")},
            "library_versions.json",
        )
        mlflow.log_metrics(
            {
                "training_rows": len(training),
                "training_fraud_rows": int(training[TARGET_COLUMN].sum()),
                "training_fraud_rate": float(training[TARGET_COLUMN].mean()),
            }
        )
        mlflow.log_params(
            {
                "scope": "training_partition_only",
                "random_state": 42,
                "split_assignment_sha256": lineage["split_assignment_sha256"],
                "interval": "nominal_marginal_wilson_95pct",
                "quantile_bins": 10,
                "interaction_amount_bins": 5,
            }
        )
        for name, table in tables.items():
            mlflow.log_text(table.to_csv(index=False, lineterminator="\n"), f"{name}.csv")
        mlflow.log_text(interpretation_markdown(summary, tables), "training_predictive_signal_report.md")
        with TemporaryDirectory(prefix="fraud_signal_figures_") as temporary:
            for path in save_signal_plots(training, tables, Path(temporary)):
                mlflow.log_artifact(str(path), artifact_path="figures")
        for name in (
            "stages/17_training_predictive_signal_diagnostics_with_mlflow.py",
            "fraud_detection/common/training_data.py",
            "fraud_detection/diagnostics/predictive_signal.py",
            "fraud_detection/diagnostics/predictive_quality.py",
            "fraud_detection/features/sender_location_history.py",
            "fraud_detection/common/modeling.py",
            "fraud_detection/common/io.py",
            "requirements.txt",
            "docs/experiments/17_predictive_signal_diagnostics_plan.md",
        ):
            mlflow.log_artifact(str(PROJECT_DIR / name), artifact_path="source_snapshot")
        print(f"Stage 17 complete. MLflow run: {run.info.run_id}", flush=True)
        print(f"Tracking URI: {tracking_uri}\nExperiment: {EXPERIMENT}", flush=True)
        print(
            "Training-only diagnostics; no models fitted, no validation/test feature analysis, no registry/API changes.",
            flush=True,
        )
        if args.mode == "smoke":
            print("This was a smoke run and is not thesis evidence.", flush=True)


# %% Executable entry point
if __name__ == "__main__":
    main()
