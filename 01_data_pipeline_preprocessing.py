# %% [markdown]
# Purpose: overview of the preprocessing pipeline and the leakage-control boundary.
# # 01 - Data Pipeline Preprocessing
#
# This file prepares the selected public fraud-detection dataset for the later ML pipeline.
#
# The aim here is not to train a model. This creates a clean, documented, and leakage-aware feature table that can be handed to the feature repository and reused by the ML pipeline.
#
# The file covers:
#
# 1. loading the raw transaction dataset,
# 2. keeping an unchanged raw copy,
# 3. checking schema, missing values, duplicates, class imbalance, and numeric ranges,
# 4. applying only safe cleaning steps,
# 5. creating deterministic row-level features,
# 6. saving the processed feature table, schema, metadata, and data-quality report.
#
# Train-test splitting, scaling, one-hot encoding, SMOTE, threshold tuning, and model training are intentionally left for the ML pipeline notebook, where they can be applied without data leakage.

# %%
# This cell imports the standard library and data-analysis packages used throughout the pipeline.
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent
import json
import shutil

import numpy as np
import pandas as pd


# %%
# This cell defines project paths, dataset names, expected schema, and the final columns to save.
DEFAULT_BASE_DIR = Path(r"D:\Germany\Documents\Magdeburg\Semester Documents\Sem 5\Thesis\Code snippets") 
DATASET_FILE_NAME = "financial_fraud_detection_dataset.csv"
FEATURE_VERSION = "fraud_features_v1"

DTYPE_MAP = {
    "transaction_id": "string",
    "timestamp": "string",
    "sender_account": "string",
    "receiver_account": "string",
    "amount": "float64",
    "transaction_type": "category",
    "merchant_category": "category",
    "location": "category",
    "device_used": "category",
    "is_fraud": "bool",
    "fraud_type": "category",
    "time_since_last_transaction": "float64",
    "spending_deviation_score": "float64",
    "velocity_score": "int16",
    "geo_anomaly_score": "float64",
    "payment_channel": "category",
    "ip_address": "string",
    "device_hash": "string",
}

EXPECTED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "amount",
    "transaction_type",
    "merchant_category",
    "location",
    "device_used",
    "is_fraud",
    "fraud_type",
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
    "payment_channel",
    "ip_address",
    "device_hash",
}

NUMERIC_COLUMNS = [
    "amount",
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]

COLUMNS_TO_KEEP = [
    "transaction_id",
    "event_timestamp",
    "transaction_type",
    "merchant_category",
    "location",
    "device_used",
    "payment_channel",
    "amount",
    "amount_log1p",
    "time_since_last_transaction",
    "time_since_last_transaction_missing_flag",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_month",
    "is_fraud",
]


# %%
# This cell defines a typed container for all input, output, metadata, and report paths.
@dataclass(frozen=True)
class PipelinePaths:
    base_dir: Path
    raw_file: Path
    raw_copy: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    feature_data_dir: Path
    metadata_dir: Path
    reports_dir: Path


# %%
# This cell builds the folder structure and returns all file paths needed by the pipeline.
def build_paths(base_dir: Path = DEFAULT_BASE_DIR) -> PipelinePaths:
    raw_file = base_dir / DATASET_FILE_NAME

    # Useful when the script is run from the folder that contains the CSV.
    if not raw_file.exists():
        base_dir = Path.cwd()
        raw_file = base_dir / DATASET_FILE_NAME

    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    feature_repo_dir = base_dir / "feature_repo"
    feature_data_dir = feature_repo_dir / "data"
    metadata_dir = feature_repo_dir / "metadata"
    reports_dir = base_dir / "reports"

    for folder in [raw_dir, interim_dir, processed_dir, feature_data_dir, metadata_dir, reports_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    paths = PipelinePaths(
        base_dir=base_dir,
        raw_file=raw_file,
        raw_copy=raw_dir / raw_file.name,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        feature_data_dir=feature_data_dir,
        metadata_dir=metadata_dir,
        reports_dir=reports_dir,
    )

    print("Dataset file found:", paths.raw_file.exists())
    print("Dataset path:", paths.raw_file)
    print("Project base folder:", paths.base_dir)

    return paths


# %%
# This cell copies the raw dataset once, reads a small preview, and loads the full dataset.
def keep_raw_copy(paths: PipelinePaths) -> None:
    if not paths.raw_file.exists():
        raise FileNotFoundError(f"The dataset file was not found. Please check that {DATASET_FILE_NAME} is in the BASE_DIR folder or in the same folder as this script.")

    if not paths.raw_copy.exists():
        shutil.copy2(paths.raw_file, paths.raw_copy)
        print("Raw file copied to:", paths.raw_copy)
    else:
        print("Raw copy already exists:", paths.raw_copy)


def read_sample(raw_copy: Path) -> pd.DataFrame:
    sample = pd.read_csv(raw_copy, nrows=5)
    print("Sample columns:", sample.columns.tolist())
    return sample


def load_dataset(raw_copy: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_copy, dtype=DTYPE_MAP)
    print("Dataset shape:", df.shape)
    return df


# %%
# This cell standardises types, validates the expected schema, and profiles data quality.
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_fraud"] = df["is_fraud"].astype("int8")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    extra_columns = set(df.columns) - EXPECTED_COLUMNS

    print("Missing columns:", missing_columns)
    print("Extra columns:", extra_columns)

    if missing_columns:
        raise ValueError(f"Some expected columns are missing: {sorted(missing_columns)}")
    if extra_columns:
        raise ValueError(f"The dataset contains unexpected columns: {sorted(extra_columns)}")


def profile_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    row_count = len(df)
    column_count = df.shape[1]
    fraud_count = int(df["is_fraud"].sum())
    non_fraud_count = int(row_count - fraud_count)
    fraud_rate = fraud_count / row_count

    print("Rows:", row_count)
    print("Columns:", column_count)
    print("Fraud count:", fraud_count)
    print("Non-fraud count:", non_fraud_count)
    print("Fraud rate:", fraud_rate)
    print("Fraud rate percentage:", round(fraud_rate * 100, 4), "%")

    missing_summary = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing_summary["missing_percentage"] = missing_summary["missing_count"] / len(df) * 100

    duplicate_count = int(df.duplicated().sum())
    print("Exact duplicate rows:", duplicate_count)

    target_distribution = (
        df["is_fraud"]
        .value_counts()
        .rename_axis("is_fraud")
        .reset_index(name="row_count")
    )
    target_distribution["percentage"] = target_distribution["row_count"] / len(df) * 100

    type_summary = (
        df.groupby("transaction_type", observed=True)
        .agg(
            row_count=("is_fraud", "size"),
            fraud_count=("is_fraud", "sum"),
            amount_mean=("amount", "mean"),
            amount_median=("amount", "median"),
        )
        .reset_index()
    )
    type_summary["fraud_rate"] = type_summary["fraud_count"] / type_summary["row_count"]
    type_summary = type_summary.sort_values("fraud_rate", ascending=False)

    numeric_summary = df[NUMERIC_COLUMNS].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    ).T

    return {
        "missing_summary": missing_summary,
        "duplicate_count": duplicate_count,
        "target_distribution": target_distribution,
        "type_summary": type_summary,
        "numeric_summary": numeric_summary,
    }


# %%
# This cell removes only safe rows: exact duplicates, negative amounts, and invalid timestamps.
def apply_safe_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    initial_rows = len(df)
    df_clean = df.drop_duplicates().copy()

    duplicate_rows_removed = initial_rows - len(df_clean)

    invalid_amount_mask = df_clean["amount"] < 0
    invalid_amount_rows = int(invalid_amount_mask.sum())
    df_clean = df_clean.loc[~invalid_amount_mask].copy()

    cleaning_stats = {
        "initial_rows": initial_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "invalid_amount_rows": invalid_amount_rows,
    }

    print("Initial rows:", initial_rows)
    print("Duplicate rows removed:", duplicate_rows_removed)
    print("Invalid negative amount rows removed:", invalid_amount_rows)
    print("Final rows after basic cleaning:", len(df_clean))

    return df_clean, cleaning_stats


def parse_event_timestamp(df_clean: pd.DataFrame, cleaning_stats: dict[str, int]) -> tuple[pd.DataFrame, dict[str, int]]:
    df_clean = df_clean.copy()
    df_clean["event_timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")

    invalid_timestamp_mask = df_clean["event_timestamp"].isna()
    invalid_timestamp_rows = int(invalid_timestamp_mask.sum())
    df_clean = df_clean.loc[~invalid_timestamp_mask].copy()

    cleaning_stats = cleaning_stats.copy()
    cleaning_stats["invalid_timestamp_rows"] = invalid_timestamp_rows

    print("Invalid timestamp rows removed:", invalid_timestamp_rows)
    print("Rows after timestamp cleaning:", len(df_clean))

    return df_clean, cleaning_stats


# %%
# This cell creates deterministic row-level features that do not learn from the target column.
def create_leakage_safe_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_features = df_clean.copy()

    # Log-transform amount to reduce skew while keeping the original amount for cost analysis.
    df_features["amount_log1p"] = np.log1p(df_features["amount"])

    # Timestamp-derived features capture simple transaction timing patterns.
    df_features["transaction_hour"] = df_features["event_timestamp"].dt.hour.astype("int8")
    df_features["transaction_day_of_week"] = df_features["event_timestamp"].dt.dayofweek.astype("int8")
    df_features["transaction_month"] = df_features["event_timestamp"].dt.month.astype("int8")

    # Preserve missingness instead of doing split-dependent learned imputation here.
    df_features["time_since_last_transaction_missing_flag"] = df_features["time_since_last_transaction"].isna().astype("int8")
    df_features["time_since_last_transaction"] = df_features["time_since_last_transaction"].fillna(-1)

    return df_features


def select_final_columns(df_features: pd.DataFrame) -> pd.DataFrame:
    return df_features[COLUMNS_TO_KEEP].copy()


# %%
# This cell saves the processed feature table, feature schema, and preprocessing metadata.
def save_processed_datasets(df_features: pd.DataFrame, paths: PipelinePaths) -> tuple[Path, Path]:
    processed_file = paths.processed_dir / f"{FEATURE_VERSION}.parquet"
    feature_store_file = paths.feature_data_dir / f"{FEATURE_VERSION}.parquet"

    try:
        df_features.to_parquet(processed_file, index=False)
        df_features.to_parquet(feature_store_file, index=False)
        print("Saved processed file:", processed_file)
        print("Saved feature repository copy:", feature_store_file)
    except ImportError:
        processed_file = paths.processed_dir / f"{FEATURE_VERSION}.csv"
        feature_store_file = paths.feature_data_dir / f"{FEATURE_VERSION}.csv"

        df_features.to_csv(processed_file, index=False)
        df_features.to_csv(feature_store_file, index=False)

        print("Parquet support is missing, so CSV files were saved instead.")
        print("Saved processed file:", processed_file)
        print("Saved feature repository copy:", feature_store_file)
        print("To save parquet later, install pyarrow with: pip install pyarrow")

    return processed_file, feature_store_file


def save_feature_schema(df_features: pd.DataFrame, paths: PipelinePaths) -> Path:
    schema = {
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now().isoformat(),
        "columns": {
            col: {
                "dtype": str(df_features[col].dtype),
                "nullable_count": int(df_features[col].isna().sum()),
            }
            for col in df_features.columns
        },
        "target_column": "is_fraud",
        "event_timestamp_column": "event_timestamp",
        "entity_column": "transaction_id",
    }

    schema_file = paths.metadata_dir / "feature_schema_v1.json"

    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    print("Saved schema:", schema_file)
    return schema_file


def save_preprocessing_metadata(df_features: pd.DataFrame, paths: PipelinePaths, cleaning_stats: dict[str, int]) -> Path:
    metadata = {
        "dataset_name": "financial_fraud_detection_dataset",
        "source_file": str(paths.raw_copy),
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now().isoformat(),
        "row_count_before_cleaning": int(cleaning_stats["initial_rows"]),
        "row_count_after_cleaning": int(len(df_features)),
        "duplicate_rows_removed": int(cleaning_stats["duplicate_rows_removed"]),
        "invalid_negative_amount_rows_removed": int(cleaning_stats["invalid_amount_rows"]),
        "invalid_timestamp_rows_removed": int(cleaning_stats["invalid_timestamp_rows"]),
        "fraud_count": int(df_features["is_fraud"].sum()),
        "fraud_rate": float(df_features["is_fraud"].mean()),
        "target_column": "is_fraud",
        "kept_for_cost_analysis": ["amount"],
        "excluded_identifier_columns": [
            "sender_account",
            "receiver_account",
            "ip_address",
            "device_hash",
        ],
        "excluded_label_derived_columns": [
            "fraud_type",
        ],
        "notes": [
            "Only deterministic, row-level preprocessing was applied before the train-validation-test split.",
            "Scaling, one-hot encoding, learned imputation, SMOTE, and threshold tuning are postponed to the ML pipeline.",
            "event_timestamp is parsed from the dataset timestamp column.",
            "fraud_type is excluded because it is known only after fraud labeling or investigation.",
        ],
    }

    metadata_file = paths.metadata_dir / "feature_metadata_v1.json"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Saved metadata:", metadata_file)
    return metadata_file


# %%
# This cell writes the data-quality report and verifies that all expected outputs were created.
def save_quality_report(df_features: pd.DataFrame, paths: PipelinePaths, cleaning_stats: dict[str, int]) -> Path:
    quality_report_file = paths.reports_dir / "data_quality_report_v1.md"

    report = dedent(f"""
    # Data Quality Report - {FEATURE_VERSION}

    ## Dataset overview

    - Raw file: `{paths.raw_copy}`
    - Rows before cleaning: {cleaning_stats["initial_rows"]}
    - Rows after cleaning: {len(df_features)}
    - Columns after preprocessing: {df_features.shape[1]}
    - Fraud count: {int(df_features["is_fraud"].sum())}
    - Fraud rate: {df_features["is_fraud"].mean():.6f}

    ## Cleaning decisions

    - Exact duplicate rows removed: {cleaning_stats["duplicate_rows_removed"]}
    - Negative amount rows removed: {cleaning_stats["invalid_amount_rows"]}
    - Invalid timestamp rows removed: {cleaning_stats["invalid_timestamp_rows"]}
    - Direct identifier columns `sender_account`, `receiver_account`, `ip_address`, and `device_hash` were removed from the processed feature table.
    - `transaction_id` was kept as the entity key for feature-repository and lineage purposes.
    - `fraud_type` was excluded because it is label-derived information, not an input available at prediction time.
    - Missing `time_since_last_transaction` values were retained with a missing-value flag and filled with -1.

    ## Leakage-control decision

    Only deterministic row-level transformations were applied before the train-validation-test split.

    The following steps are postponed to the ML pipeline:

    - learned imputation
    - scaling or normalization
    - one-hot encoding
    - SMOTE or oversampling
    - feature selection using the target
    - threshold tuning

    ## Dataset limitation note

    The dataset is public and static. It is suitable for demonstrating a thesis prototype, but it should not be presented as evidence of production performance in a real banking environment.
    """).strip() + "\n"

    with open(quality_report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print("Saved report:", quality_report_file)
    return quality_report_file


def verify_outputs(df_features: pd.DataFrame, output_files: list[Path]) -> None:
    print("Processed dataset shape:", df_features.shape)
    print("Missing values total:", int(df_features.isna().sum().sum()))
    print("Fraud rate:", df_features["is_fraud"].mean())

    for file in output_files:
        print(file.exists(), file)


# %%
# This cell runs the full preprocessing pipeline in the correct order.
def main() -> None:
    paths = build_paths()
    keep_raw_copy(paths)
    read_sample(paths.raw_copy)

    df = load_dataset(paths.raw_copy)
    df = standardise_columns(df)
    validate_schema(df)
    profile_dataset(df)

    df_clean, cleaning_stats = apply_safe_cleaning(df)
    df_clean, cleaning_stats = parse_event_timestamp(df_clean, cleaning_stats)

    df_features = create_leakage_safe_features(df_clean)
    df_features = select_final_columns(df_features)

    processed_file, feature_store_file = save_processed_datasets(df_features, paths)
    schema_file = save_feature_schema(df_features, paths)
    metadata_file = save_preprocessing_metadata(df_features, paths, cleaning_stats)
    quality_report_file = save_quality_report(df_features, paths, cleaning_stats)

    verify_outputs(df_features, [processed_file, feature_store_file, schema_file, metadata_file, quality_report_file])


# %%
# This cell keeps the file executable as a normal Python script and avoids confusing
# NameError messages when the final cell is run before the earlier cells in an IDE.
REQUIRED_PIPELINE_NAMES = [
    "build_paths",
    "keep_raw_copy",
    "read_sample",
    "load_dataset",
    "standardise_columns",
    "validate_schema",
    "profile_dataset",
    "apply_safe_cleaning",
    "parse_event_timestamp",
    "create_leakage_safe_features",
    "select_final_columns",
    "save_processed_datasets",
    "save_feature_schema",
    "save_preprocessing_metadata",
    "save_quality_report",
    "verify_outputs",
    "main",
]

missing_pipeline_names = [
    name for name in REQUIRED_PIPELINE_NAMES
    if name not in globals()
]

if __name__ == "__main__":
    if missing_pipeline_names:
        print("Run the earlier cells first. Missing definitions:", ", ".join(missing_pipeline_names))
    else:
        main()
