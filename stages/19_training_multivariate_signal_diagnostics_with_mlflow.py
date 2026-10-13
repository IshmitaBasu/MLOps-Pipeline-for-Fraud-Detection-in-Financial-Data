"""Stage 19: training-only multivariate predictive-signal diagnostics."""

# %% Imports and immutable references
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

# %% Repository bootstrap for direct script and VS Code cell execution
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fraud_detection.common.modeling import PROJECT_DIR, configure_mlflow
from fraud_detection.common.training_data import load_training_data
from fraud_detection.diagnostics.multivariate_signal import (
    RANDOM_STATE,
    add_multivariate_features,
    attach_development_only_history,
    audit_slice_metrics,
    baseline_results,
    feature_importance_table,
    fit_and_score,
    frozen_feature_sets,
    internal_development_audit_split,
    interpretation_summary,
    validate_feature_sets,
)
from fraud_detection.common.io import sha256_file

EXPERIMENT = "financial-fraud-stage-19-training-multivariate-signal"
RAW_PATH = PROJECT_DIR / "financial_fraud_detection_dataset.csv"
REFERENCE_PATH = PROJECT_DIR / "mlartifacts/5920c17849574c11b4854c3c5290bc41/artifacts/voting_full_data_summary.json"


# %% Arguments and source verification
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--raw-data-path", type=Path, default=RAW_PATH)
    return parser.parse_args()


def verified_source_hash(path: Path, mode: str) -> str:
    if mode == "smoke":
        return "not_calculated_smoke_only"
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    expected = reference["raw_dataset_info"]["dataset_sha256"]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("Raw source checksum differs from the frozen development source.")
    return observed


# %% Aggregate reporting and figures
def save_figures(results: pd.DataFrame, slices: pd.DataFrame, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ordered = results.sort_values("average_precision")
    ax.barh(ordered["candidate"], ordered["average_precision"], color="#2f6f9f")
    prevalence = float(results.loc[results["candidate"] == "prevalence_baseline", "average_precision"].iloc[0])
    ax.axvline(prevalence, color="#a33a3a", linestyle="--", label=f"Prevalence AP = {prevalence:.4f}")
    ax.set_xlabel("Average Precision on the internal training-audit subset")
    ax.set_title("Stage 19: multivariate signal compared with baselines")
    ax.legend()
    fig.tight_layout()
    path = output / "multivariate_average_precision.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    selected = slices.loc[slices["candidate"].isin(["all_supplied_main_fields", "all_supplied_plus_interactions"])]
    pivot = selected.pivot(index="slice", columns="candidate", values="average_precision")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Average Precision")
    ax.set_xlabel("Internal training-audit stability slice")
    ax.set_title("Stage 19: main fields versus explicit interactions by slice")
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    path = output / "interaction_stability_slices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def report_markdown(summary: dict) -> str:
    interpretation = summary["interpretation"]
    return (
        "\n".join(
            [
                "# Stage 19 training-only multivariate-signal diagnostic",
                "",
                f"Evidence scope: `{summary['evidence_scope']}`; analysed rows came only from the frozen training partition.",
                "",
                "The comparison asks whether several fields used together outperform prevalence, the strongest Stage 17 "
                "marginal pattern, and a shuffled-label control. It also measures whether four predeclared interactions add "
                "anything beyond the supplied main fields.",
                "",
                f"Multivariate AP lift over the time-missingness pattern: "
                f"{interpretation['multivariate_ap_lift_over_time_missingness_pattern']:.6f}.",
                f"Explicit-interaction AP lift over all main fields: "
                f"{interpretation['interaction_ap_lift_over_all_main_fields']:.6f}.",
                f"True-label AP gap over the shuffled control: "
                f"{interpretation['true_label_ap_gap_over_shuffled_control']:.6f}.",
                f"Interaction reading: `{interpretation['interaction_reading']}`.",
                "",
                "These are exploratory training diagnostics, not new validation or test evidence. No model, threshold, "
                "registry version or API deployment is changed by this run.",
            ]
        )
        + "\n"
    )


# %% Tracked training-only workflow
def main() -> None:
    args = parse_args()
    evidence_scope = "smoke_test" if args.mode == "smoke" else "full_data"
    feature_sets = frozen_feature_sets()
    validate_feature_sets(feature_sets)
    tracking_uri = configure_mlflow(EXPERIMENT)
    print(f"Running Stage 19 training-only multivariate-signal {args.mode} diagnostics.", flush=True)
    source_hash = verified_source_hash(args.raw_data_path, args.mode)
    with mlflow.start_run(run_name=f"training_multivariate_signal_{evidence_scope}") as active:
        mlflow.set_tags(
            {
                "stage": "19_training_multivariate_signal",
                "evidence_scope": evidence_scope,
                "scope": "frozen_training_partition_only",
                "validation_analysed": "false",
                "test_analysed": "false",
                "eligible_for_model_selection": "false",
                "registry_changed": "false",
                "api_changed": "false",
            }
        )
        training, lineage = load_training_data(args.raw_data_path, args.mode)
        lineage["source_sha256"] = source_hash
        development, audit, internal_split = internal_development_audit_split(training)
        del training
        print(
            "Building history from earlier internal-development events and adding predeclared interactions.", flush=True
        )
        development, audit = attach_development_only_history(development, audit)
        development = add_multivariate_features(development)
        audit = add_multivariate_features(audit)
        print(
            f"Internal training-only split: development={len(development):,}, audit={len(audit):,}. "
            "Validation and test remain excluded.",
            flush=True,
        )

        result_rows = baseline_results(audit)
        slice_tables = []
        importance_tables = []
        for feature_set in feature_sets:
            pipeline, scores, result = fit_and_score(development, audit, feature_set)
            result_rows.append(result)
            slice_tables.append(audit_slice_metrics(audit, scores, feature_set.name))
            importance_tables.append(feature_importance_table(pipeline, feature_set.name))
            print(
                f"{feature_set.name}: internal-audit Average Precision={result['average_precision']:.6f}, "
                f"ROC-AUC={result['roc_auc']:.6f}",
                flush=True,
            )

        interaction_set = feature_sets[-1]
        shuffled_pipeline, shuffled_scores, shuffled_result = fit_and_score(
            development, audit, interaction_set, shuffle_development_labels=True
        )
        result_rows.append(shuffled_result)
        importance_tables.append(feature_importance_table(shuffled_pipeline, shuffled_result["candidate"]))
        print(
            f"{shuffled_result['candidate']}: internal-audit Average Precision="
            f"{shuffled_result['average_precision']:.6f}",
            flush=True,
        )

        results = pd.DataFrame(result_rows)
        slices = pd.concat(slice_tables, ignore_index=True)
        importances = pd.concat(importance_tables, ignore_index=True)
        interpretation = interpretation_summary(results, slices)
        summary = {
            "purpose": "Check whether multivariate and explicit interaction patterns exist inside frozen training data.",
            "evidence_scope": evidence_scope,
            "run_id": active.info.run_id,
            "lineage": lineage,
            "internal_split": internal_split,
            "feature_sets": [item.__dict__ for item in feature_sets],
            "model_role": "fixed_untuned_diagnostic_only",
            "model_configuration": "Stage 11 LightGBM reference configuration",
            "interpretation": interpretation,
            "validation_analysed": False,
            "test_analysed": False,
            "registered_model_changed": False,
            "api_changed": False,
        }
        mlflow.log_dict(summary, "multivariate_signal_summary.json")
        mlflow.log_dict(lineage, "training_lineage.json")
        mlflow.log_dict(internal_split, "internal_training_split.json")
        mlflow.log_text(results.to_csv(index=False, lineterminator="\n"), "candidate_metrics.csv")
        mlflow.log_text(slices.to_csv(index=False, lineterminator="\n"), "stability_slice_metrics.csv")
        mlflow.log_text(importances.to_csv(index=False, lineterminator="\n"), "transformed_feature_importances.csv")
        mlflow.log_text(report_markdown(summary), "multivariate_signal_report.md")
        mlflow.log_params(
            {
                "random_state": RANDOM_STATE,
                "scope": "frozen_training_partition_only",
                "audit_fraction": internal_split["audit_fraction"],
                "fixed_alert_rate": 0.05,
                "validation_rows_used": 0,
                "test_rows_used": 0,
            }
        )
        mlflow.log_metrics(
            {
                "training_partition_rows": len(development) + len(audit),
                "internal_development_rows": len(development),
                "internal_audit_rows": len(audit),
                "interaction_ap_lift": interpretation["interaction_ap_lift_over_all_main_fields"],
                "multivariate_ap_lift": interpretation["multivariate_ap_lift_over_time_missingness_pattern"],
                "true_label_ap_gap_over_shuffled": interpretation["true_label_ap_gap_over_shuffled_control"],
            }
        )
        mlflow.log_dict(
            {package: version(package) for package in ("pandas", "numpy", "scikit-learn", "lightgbm", "mlflow")},
            "library_versions.json",
        )
        with TemporaryDirectory(prefix="fraud_multivariate_figures_") as temporary:
            for path in save_figures(results, slices, Path(temporary)):
                mlflow.log_artifact(str(path), artifact_path="figures")
        for name in (
            "stages/19_training_multivariate_signal_diagnostics_with_mlflow.py",
            "fraud_detection/diagnostics/multivariate_signal.py",
            "fraud_detection/common/training_data.py",
            "fraud_detection/features/sender_location_history.py",
            "requirements.txt",
            "docs/experiments/19_multivariate_signal_diagnostics.md",
        ):
            mlflow.log_artifact(str(PROJECT_DIR / name), artifact_path="source_snapshot")

        print(f"Stage 19 complete. MLflow run: {active.info.run_id}", flush=True)
        print(f"Tracking URI: {tracking_uri}\nExperiment: {EXPERIMENT}", flush=True)
        print(f"Interpretation: {interpretation['interaction_reading']}.", flush=True)
        print("No validation/test analysis, model selection, registry update or API change occurred.", flush=True)
        if args.mode == "smoke":
            print("This was a smoke run and is not thesis evidence.", flush=True)


# %% Executable entry point
if __name__ == "__main__":
    main()
