"""Stage 21: one fixed validation-only LightGBM baseline on IBM AML HI-Small."""

# %% Imports and immutable local source
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# %% Repository bootstrap for direct script and VS Code cell execution
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fraud_detection.common.modeling import PROJECT_DIR, configure_mlflow, save_model_explanation
from fraud_detection.datasets.ibm_aml_baseline import (
    DEFAULT_CHUNK_ROWS,
    EXPECTED_SHA256,
    FEATURE_COLUMNS,
    MINIMUM_AP_LIFT,
    MINIMUM_STABILITY_AP_LIFT,
    MINIMUM_TOP_1PCT_RECALL,
    MODEL_SERIALIZATION_FORMAT,
    NUMERIC_FEATURES,
    REVIEW_RATES,
    SMOKE_ROWS,
    TARGET_COLUMN,
    baseline_progression_decision,
    build_baseline_pipeline,
    feature_contract,
    fit_and_evaluate_baseline,
    make_development_split,
    model_parameters,
    prepare_modeling_population,
    save_validation_figures,
)
from fraud_detection.common.io import sha256_file
from fraud_detection.evaluation.validation import positive_class_weight

EXPERIMENT = "financial-fraud-stage-21-ibm-aml-baseline"
DEFAULT_DATA_PATH = PROJECT_DIR / "data_candidates" / "ibm_aml" / "HI-Small_Trans.csv"


# %% Command-line contract
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    return parser.parse_args()


# %% Readable aggregate report
def report_markdown(summary: dict) -> str:
    metrics = summary["validation_metrics"]
    decision = summary["decision"]
    return (
        "\n".join(
            [
                "# IBM AML small-baseline report",
                "",
                f"Evidence scope: `{summary['evidence_scope']}`.",
                f"Training rows: {summary['split_manifest']['train_rows']:,}.",
                f"Validation rows: {summary['split_manifest']['validation_rows']:,}.",
                f"Held-out test rows: {summary['split_manifest']['held_out_test_rows']:,}; not evaluated.",
                "",
                f"Validation prevalence: {metrics['validation_prevalence']:.6%}.",
                f"Validation Average Precision: {metrics['validation_average_precision']:.6f}.",
                f"Average Precision lift over prevalence: {metrics['validation_average_precision_lift']:.2f}x.",
                f"Validation ROC-AUC: {metrics['validation_roc_auc']:.6f}.",
                f"Validation F1 at its validation-selected threshold: {metrics['validation_f1_at_max_f1']:.6f}.",
                "",
                f"Decision: `{decision['outcome']}`.",
                decision["reason"],
                "",
                "No test scores were produced. No model was registered, deployed or connected to the existing fraud pipeline.",
            ]
        )
        + "\n"
    )


def table_records(table) -> list[dict]:
    """Return JSON-safe aggregate rows, converting missing numeric cells to null."""

    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


# %% Tracked validation-only baseline
def main() -> None:
    args = parse_args()
    smoke = args.mode == "smoke"
    evidence_scope = "smoke_test" if smoke else "full_validation"
    row_limit = SMOKE_ROWS if smoke else None
    print(f"Running the Stage 21 IBM AML {args.mode} validation-only baseline.", flush=True)
    print("Verifying the candidate file and preparing the fixed primary period.", flush=True)
    source_sha256 = sha256_file(args.data_path).upper()
    if source_sha256 != EXPECTED_SHA256:
        raise ValueError(
            "The IBM AML file does not match the Stage 20 source receipt. "
            f"Expected {EXPECTED_SHA256}, observed {source_sha256}."
        )

    tracking_uri = configure_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"ibm_aml_lightgbm_{evidence_scope}") as active:
        mlflow.set_tags(
            {
                "stage": "21_ibm_aml_small_baseline",
                "evidence_scope": evidence_scope,
                "target_semantics": "synthetic_money_laundering",
                "validation_only": "true",
                "test_split_evaluated": "false",
                "model_registered": "false",
                "existing_pipeline_changed": "false",
            }
        )
        population, population_receipt = prepare_modeling_population(
            args.data_path,
            source_sha256=source_sha256,
            chunk_rows=args.chunk_rows,
            row_limit=row_limit,
        )
        split = make_development_split(population, source_sha256=source_sha256)
        del population
        print(
            "Frozen stratified random split: "
            f"train={split.manifest['train_rows']:,}, validation={split.manifest['validation_rows']:,}, "
            f"test={split.manifest['held_out_test_rows']:,}.",
            flush=True,
        )
        print("The held-out test split will not be materialised or evaluated.", flush=True)

        class_weight = positive_class_weight(split.train[TARGET_COLUMN])
        pipeline = build_baseline_pipeline(class_weight)
        result = fit_and_evaluate_baseline(pipeline, split.train, split.validation)
        decision = baseline_progression_decision(
            average_precision=result.metrics["validation_average_precision"],
            prevalence=result.metrics["validation_prevalence"],
            workloads=result.workloads,
            stability=result.stability,
            smoke=smoke,
        )
        summary = {
            "purpose": "Test one fixed transaction-level IBM AML baseline without changing the fraud pipeline.",
            "evidence_scope": evidence_scope,
            "source": {
                "filename": args.data_path.name,
                "sha256": source_sha256,
                "target_semantics": "synthetic_money_laundering_not_general_transaction_fraud",
            },
            "population_receipt": population_receipt,
            "split_manifest": split.manifest,
            "feature_contract": feature_contract(),
            "model_parameters": model_parameters(class_weight),
            "validation_metrics": result.metrics,
            "review_workloads": table_records(result.workloads),
            "stability_summary": table_records(result.stability),
            "decision": decision,
            "test_split_evaluated": False,
            "test_metrics_present": False,
            "model_registered": False,
            "existing_pipeline_changed": False,
            "run_id": active.info.run_id,
        }

        mlflow.log_params(
            {
                "mode": args.mode,
                "row_limit": row_limit if row_limit is not None else "none",
                "chunk_rows": args.chunk_rows,
                "model": "lightgbm_reference",
                "class_weight_source": "training_partition_only",
                "scale_pos_weight": class_weight,
                "review_rates": ",".join(str(value) for value in REVIEW_RATES),
                "minimum_average_precision_lift": MINIMUM_AP_LIFT,
                "minimum_stability_average_precision_lift": MINIMUM_STABILITY_AP_LIFT,
                "minimum_top_1pct_recall": MINIMUM_TOP_1PCT_RECALL,
                "test_split_evaluated": False,
            }
        )
        mlflow.log_metrics({key: value for key, value in result.metrics.items() if isinstance(value, (int, float))})
        mlflow.log_dict(summary, "ibm_aml_baseline_summary.json")
        mlflow.log_text(report_markdown(summary), "ibm_aml_baseline_report.md")
        mlflow.log_text(result.workloads.to_csv(index=False, lineterminator="\n"), "validation_review_workloads.csv")
        mlflow.log_text(
            result.currency_capture.to_csv(index=False, lineterminator="\n"),
            "validation_currency_amount_capture.csv",
        )
        mlflow.log_text(result.stability.to_csv(index=False, lineterminator="\n"), "validation_stability.csv")
        mlflow.log_dict(
            {package: version(package) for package in ("pandas", "numpy", "scikit-learn", "lightgbm", "mlflow")},
            "library_versions.json",
        )

        with TemporaryDirectory(prefix="ibm_aml_stage_21_") as temporary:
            artifact_dir = Path(temporary)
            save_validation_figures(
                split.validation[TARGET_COLUMN],
                result.scores,
                result.metrics["validation_max_f1_threshold"],
                result.workloads,
                artifact_dir,
            )
            save_model_explanation(pipeline, artifact_dir)
            mlflow.log_artifacts(str(artifact_dir), artifact_path="validation_evidence")

        if not smoke:
            signature_input = split.train[FEATURE_COLUMNS].head(100).copy()
            signature_input[NUMERIC_FEATURES] = signature_input[NUMERIC_FEATURES].astype("float64")
            signature = infer_signature(signature_input, pipeline.predict_proba(signature_input)[:, 1])
            mlflow.sklearn.log_model(
                pipeline,
                name="validation_model",
                signature=signature,
                serialization_format=MODEL_SERIALIZATION_FORMAT,
            )

        for name in (
            "stages/21_ibm_aml_baseline_with_mlflow.py",
            "fraud_detection/datasets/ibm_aml_baseline.py",
            "requirements.txt",
            "docs/experiments/20_ibm_aml_dataset_feasibility_audit.md",
            "docs/experiments/21_ibm_aml_baseline_plan.md",
        ):
            mlflow.log_artifact(str(PROJECT_DIR / name), artifact_path="source_snapshot")

        top_one = result.workloads.loc[result.workloads.requested_review_rate == 0.01].iloc[0]
        print(
            "lightgbm_reference: validation "
            f"Average Precision={result.metrics['validation_average_precision']:.6f}, "
            f"lift={result.metrics['validation_average_precision_lift']:.2f}x, "
            f"top-1% recall={top_one.recall:.2%}.",
            flush=True,
        )
        print(f"Stage 21 run: {active.info.run_id}", flush=True)
        print(f"Tracking URI: {tracking_uri}\nExperiment: {EXPERIMENT}", flush=True)
        print(f"Decision: {decision['outcome']}.", flush=True)
        print("The held-out test split was not evaluated and the existing fraud pipeline was not changed.", flush=True)
        if smoke:
            print("This was a smoke run and is not thesis evidence.", flush=True)


# %% Executable entry point
if __name__ == "__main__":
    main()
