"""Stage 20: chunked, read-only feasibility audit of IBM AML HI-Small."""

# %% Imports and immutable local source
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import mlflow

from fraud_modeling_utils import PROJECT_DIR, configure_mlflow
from ibm_aml_feasibility_utils import (
    DEFAULT_CHUNK_ROWS,
    EXPECTED_SHA256,
    SMOKE_ROWS,
    audit_ibm_aml_csv,
    audit_report_markdown,
)
from project_io_utils import sha256_file

EXPERIMENT = "financial-fraud-stage-20-ibm-aml-feasibility"
DEFAULT_DATA_PATH = PROJECT_DIR / "data_candidates" / "ibm_aml" / "HI-Small_Trans.csv"


# %% Command-line contract
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    return parser.parse_args()


# %% Tracked aggregate-only audit
def main() -> None:
    args = parse_args()
    evidence_scope = "smoke_test" if args.mode == "smoke" else "full_file"
    row_limit = SMOKE_ROWS if args.mode == "smoke" else None
    print(f"Running Stage 20 IBM AML {args.mode} feasibility audit.", flush=True)
    print("Calculating and verifying the local candidate-file checksum.", flush=True)
    source_sha256 = sha256_file(args.data_path).upper()
    if source_sha256 != EXPECTED_SHA256:
        raise ValueError(
            "The IBM AML candidate file does not match the verified download receipt. "
            f"Expected {EXPECTED_SHA256}, observed {source_sha256}."
        )
    tracking_uri = configure_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"ibm_aml_feasibility_{evidence_scope}") as active:
        mlflow.set_tags(
            {
                "stage": "20_ibm_aml_dataset_feasibility",
                "evidence_scope": evidence_scope,
                "read_only_data_audit": "true",
                "model_trained": "false",
                "existing_pipeline_changed": "false",
                "validation_or_test_reused": "false",
                "row_level_artifacts_logged": "false",
            }
        )
        summary, tables = audit_ibm_aml_csv(
            args.data_path, source_sha256=source_sha256, chunk_rows=args.chunk_rows, row_limit=row_limit
        )
        summary["run_id"] = active.info.run_id
        summary["evidence_scope"] = evidence_scope
        mlflow.log_dict(summary, "ibm_aml_feasibility_summary.json")
        mlflow.log_text(audit_report_markdown(summary), "ibm_aml_feasibility_report.md")
        for name, table in tables.items():
            mlflow.log_text(table.to_csv(index=False, lineterminator="\n"), f"{name}.csv")
        mlflow.log_dict(
            {package: version(package) for package in ("pandas", "numpy", "mlflow")}, "library_versions.json"
        )
        mlflow.log_params(
            {
                "mode": args.mode,
                "chunk_rows": args.chunk_rows,
                "row_limit": row_limit if row_limit is not None else "none",
                "primary_period_end_exclusive": summary["time"]["primary_period_end_exclusive"],
                "minimum_positive_rows": 1_000,
                "minimum_history_coverage": 0.20,
            }
        )
        mlflow.log_metrics(
            {
                "audited_rows": summary["quality"]["audited_valid_rows"],
                "laundering_rows": summary["labels"]["laundering_rows"],
                "laundering_rate": summary["labels"]["laundering_rate"],
                "sender_history_coverage": summary["history"]["sender_prior_history_coverage"],
                "receiver_history_coverage": summary["history"]["receiver_prior_history_coverage"],
                "pair_history_coverage": summary["history"]["pair_prior_history_coverage"],
                "malformed_rows": summary["quality"]["malformed_structure_rows"],
                "invalid_timestamp_rows": summary["quality"]["invalid_timestamp_rows"],
                "invalid_label_rows": summary["quality"]["invalid_label_rows"],
            }
        )
        for name in (
            "20_ibm_aml_dataset_feasibility_audit.py",
            "ibm_aml_feasibility_utils.py",
            "requirements.txt",
            "Markdown files/20_ibm_aml_dataset_feasibility_audit.md",
        ):
            mlflow.log_artifact(str(PROJECT_DIR / name), artifact_path="source_snapshot")

        print(f"Audited rows: {summary['quality']['audited_valid_rows']:,}.", flush=True)
        print(
            f"Laundering rows: {summary['labels']['laundering_rows']:,} "
            f"({summary['labels']['laundering_rate']:.6%}).",
            flush=True,
        )
        print(
            "Earlier-history coverage: "
            f"sender={summary['history']['sender_prior_history_coverage']:.2%}, "
            f"receiver={summary['history']['receiver_prior_history_coverage']:.2%}, "
            f"pair={summary['history']['pair_prior_history_coverage']:.2%}.",
            flush=True,
        )
        print(f"Stage 20 run: {active.info.run_id}", flush=True)
        print(f"Tracking URI: {tracking_uri}\nExperiment: {EXPERIMENT}", flush=True)
        print(f"Decision: {summary['decision']['outcome']}.", flush=True)
        print("No model was trained and the existing dataset pipeline was not changed.", flush=True)
        if args.mode == "smoke":
            print("This was a smoke run and cannot decide dataset suitability.", flush=True)


# %% Executable entry point
if __name__ == "__main__":
    main()
