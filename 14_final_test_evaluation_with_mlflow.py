"""Stage 14: one-time final test evaluation at validation-frozen rules."""

# %% Imports and runtime configuration
from __future__ import annotations

import argparse
import gc
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import mlflow
import numpy as np

from final_test_evaluation_utils import (
    STAGE_13_SUMMARY_RUN_ID,
    add_final_history_features,
    evaluate_frozen_test,
    freeze_final_protocol,
    reserve_final_evaluation,
)
from fraud_modeling_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURE_REPO_PATH,
    PROJECT_DIR,
    TARGET_COLUMN,
    configure_mlflow,
    dataset_metadata,
    load_modeling_table,
    make_optional_sample,
    predict_scores,
)
from lightgbm_tuning_utils import frozen_lightgbm_configs
from model_family_comparison_utils import build_model_pipeline
from oversampling_experiment_utils import MODEL_FEATURES
from predictive_quality_utils import TEST_CODE, TRAIN_CODE, build_split_manifest, make_stratified_random_partition
from project_io_utils import save_json_atomically
from sender_location_history_utils import attach_sender_account, restore_raw_source_order
from validation_evaluation_utils import positive_class_weight

# %% Frozen paths and CLI
DEFAULT_EXPERIMENT = "financial-fraud-stage-14-final-test-evaluation"
RECEIPT_DIR = PROJECT_DIR / "final_test_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the retained LightGBM at frozen rules; no test tuning.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-source", choices=("feast", "csv"), default="feast")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=PROJECT_DIR / "financial_fraud_detection_dataset.csv")
    parser.add_argument("--feature-repo-path", type=Path, default=DEFAULT_FEATURE_REPO_PATH)
    return parser.parse_args()


# %% Validation evidence is the only source of the frozen protocol
def load_protocol() -> tuple[dict, dict]:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(STAGE_13_SUMMARY_RUN_ID)
    if run.data.tags.get("run_role") != "full_data_summary":
        raise ValueError("The Stage 13 handoff is not a full-data summary.")
    artifact = client.download_artifacts(STAGE_13_SUMMARY_RUN_ID, "voting_full_data_summary.json")
    summary = json.loads(Path(artifact).read_text(encoding="utf-8"))
    protocol = freeze_final_protocol(summary, STAGE_13_SUMMARY_RUN_ID)
    definition_path = client.download_artifacts(protocol["validation_reference_run_id"], "candidate_definition.json")
    definition = json.loads(Path(definition_path).read_text(encoding="utf-8"))
    members = definition.get("members", [])
    if len(members) != 1 or members[0].get("parameters") != protocol["model_parameters"]:
        raise ValueError("The model parameters differ from the executed Stage 13 reference.")
    versions_path = client.download_artifacts(protocol["validation_reference_run_id"], "library_versions.json")
    recorded_versions = json.loads(Path(versions_path).read_text(encoding="utf-8"))
    for package in ("scikit-learn", "lightgbm", "pandas", "numpy"):
        if version(package) != recorded_versions.get(package):
            raise ValueError(f"The {package} version differs from the executed Stage 13 reference.")
    return protocol, summary


def tracked_source_files() -> list[Path]:
    names = (
        "14_final_test_evaluation_with_mlflow.py",
        "final_test_evaluation_utils.py",
        "model_family_comparison_utils.py",
        "lightgbm_tuning_utils.py",
        "validation_evaluation_utils.py",
        "sender_location_history_utils.py",
        "oversampling_experiment_utils.py",
        "fraud_modeling_utils.py",
        "predictive_quality_utils.py",
        "project_io_utils.py",
        "requirements.txt",
        "Markdown files/14_final_test_evaluation_plan.md",
        "Markdown files/14_final_test_evaluation_implementation_guide.md",
    )
    return [PROJECT_DIR / name for name in names]


# %% Final execution and one-time evaluation receipt
def main() -> None:
    args = parse_args()
    is_smoke = args.mode == "smoke"
    scope = "smoke_test" if is_smoke else "full_data"
    tracking_uri = configure_mlflow(DEFAULT_EXPERIMENT)
    protocol, stage_13 = load_protocol()
    receipt = RECEIPT_DIR / f"{protocol['split_assignment_sha256']}.json"
    if not is_smoke and receipt.exists():
        raise RuntimeError(f"This test split already has an evaluation receipt. Inspect it before any rerun: {receipt}")
    if not is_smoke:
        prior = mlflow.search_runs(filter_string="tags.canonical_test_evaluated = 'true'", search_all_experiments=True)
        if not prior.empty:
            raise RuntimeError("An existing canonical final-test run was found. Repeated final evaluation is blocked.")

    with mlflow.start_run(run_name=f"final_test_{scope}") as run:
        mlflow.set_tags(
            {
                "stage": "14_final_test_evaluation",
                "run_role": f"{scope}_final_evaluation",
                "evidence_scope": scope,
                "canonical_test_evaluated": "false",
                "test_split_evaluated": "false",
            }
        )
        mlflow.log_dict(protocol, "frozen_evaluation_protocol.json")
        for source in tracked_source_files():
            mlflow.log_artifact(str(source), artifact_path="source_snapshot")
        print(f"Frozen primary score threshold: {protocol['threshold']:.17g}", flush=True)
        table, dataset_info, lineage = load_modeling_table(
            data_source=args.data_source,
            data_path=args.data_path,
            feature_repo_path=args.feature_repo_path,
            calculate_hash=not is_smoke,
        )
        raw_info = dataset_metadata(args.raw_data_path, calculate_hash=not is_smoke)
        if is_smoke:
            table = make_optional_sample(table, 50_000, 42)
        else:
            if len(table) != 5_000_000:
                raise ValueError("The final full-data evaluation requires exactly 5,000,000 rows.")
            if dataset_info.get("dataset_sha256") != stage_13["dataset_info"]["dataset_sha256"]:
                raise ValueError("The modelling source changed since Stage 13.")
            if raw_info.get("dataset_sha256") != stage_13["raw_dataset_info"]["dataset_sha256"]:
                raise ValueError("The raw history source changed since Stage 13.")
        table = restore_raw_source_order(table, args.raw_data_path)
        partition = make_stratified_random_partition(
            table[TARGET_COLUMN], train_fraction=0.70, validation_fraction=0.15, seed=42
        )
        manifest = build_split_manifest(table, partition, seed=42, train_fraction=0.70, validation_fraction=0.15)
        if not is_smoke and manifest["split_assignment_sha256"] != protocol["split_assignment_sha256"]:
            raise ValueError("The final split differs from the frozen validation split.")
        print(
            f"Running Stage 14 {scope}: train={manifest['splits']['train']['rows']:,}, "
            f"test={manifest['splits']['test']['rows']:,}. Validation rows will not be used for fitting.",
            flush=True,
        )
        keep = (partition == TRAIN_CODE) | (partition == TEST_CODE)
        training_mask = np.asarray(partition[keep] == TRAIN_CODE)
        data = table.loc[keep].reset_index(drop=True)
        del table, partition, keep
        gc.collect()
        data = attach_sender_account(data, args.raw_data_path)
        data = add_final_history_features(data, training_mask)
        train_df = data.loc[training_mask].reset_index(drop=True)
        test_df = data.loc[~training_mask].reset_index(drop=True)
        del data, training_mask
        gc.collect()
        ratio = positive_class_weight(train_df[TARGET_COLUMN])
        model = build_model_pipeline(frozen_lightgbm_configs()[0].as_model_family_config(), positive_class_weight=ratio)
        print("Fitting the frozen LightGBM on the original training partition only.", flush=True)
        start = perf_counter()
        model.fit(train_df[MODEL_FEATURES], train_df[TARGET_COLUMN])
        training_seconds = perf_counter() - start
        del train_df
        gc.collect()
        reservation = {"status": "test_scoring_started", "run_id": run.info.run_id, "protocol": protocol}
        if not is_smoke:
            reserve_final_evaluation(receipt, reservation)
        print("Scoring test rows once at the frozen rules.", flush=True)
        start = perf_counter()
        scores, _, _ = predict_scores(model, test_df[MODEL_FEATURES])
        inference_seconds = perf_counter() - start
        metrics, workloads = evaluate_frozen_test(test_df[TARGET_COLUMN], scores, test_df["amount"], protocol)
        result = {
            "run_id": run.info.run_id,
            "evidence_scope": scope,
            "protocol": protocol,
            "split_manifest": manifest,
            "test_metrics": metrics,
            "test_workloads": workloads.to_dict("records"),
            "training_seconds": training_seconds,
            "test_inference_seconds": inference_seconds,
            "test_split_evaluated": True,
            "canonical_test_evaluated": not is_smoke,
            "model_registered": False,
        }
        if not is_smoke:
            save_json_atomically({"status": "test_evaluated", **result}, receipt)
        mlflow.set_tags({"test_split_evaluated": "true", "canonical_test_evaluated": str(not is_smoke).lower()})
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_metrics({"training_seconds": training_seconds, "test_inference_seconds": inference_seconds})
        mlflow.log_params(
            {
                "class_weight_ratio": ratio,
                "threshold_source": "Stage_13_validation_max_f1",
                "feature_set": protocol["feature_set"],
                "random_state": 42,
            }
        )
        mlflow.log_dict(result, "final_test_evaluation.json")
        mlflow.log_dict(dataset_info, "dataset_info.json")
        mlflow.log_dict(raw_info, "raw_dataset_info.json")
        mlflow.log_dict(lineage, "feature_lineage.json")
        mlflow.log_dict(
            {
                package: version(package)
                for package in ("scikit-learn", "lightgbm", "mlflow", "pandas", "numpy", "joblib")
            },
            "library_versions.json",
        )
        mlflow.log_text(workloads.to_csv(index=False, lineterminator="\n"), "test_workloads.csv")
        mlflow.log_dict(
            {
                key: metrics[key]
                for key in ("threshold", "true_negatives", "false_positives", "false_negatives", "true_positives")
            },
            "test_confusion_matrix.json",
        )
        with TemporaryDirectory(prefix="fraud_final_model_") as temporary:
            model_path = Path(temporary) / "evaluated_lightgbm_pipeline.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="evaluated_model")
        if not is_smoke:
            save_json_atomically({"status": "complete", **result}, receipt)
        print(json.dumps(metrics, indent=2), flush=True)
        print(f"Stage 14 complete. Run: {run.info.run_id}\nTracking URI: {tracking_uri}", flush=True)
        if is_smoke:
            print("This smoke evaluation is not canonical test evidence.", flush=True)
        else:
            print(f"Canonical test evaluation completed once. Receipt: {receipt}", flush=True)


# %% Cell-aware executable entry point
if __name__ == "__main__":
    main()
