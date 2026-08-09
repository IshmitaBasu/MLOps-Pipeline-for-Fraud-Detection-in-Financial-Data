# %% [markdown]
# Purpose: create one minimally cleaned table for post-cleaning EDA and the baseline.
# # 02 - Data Cleaning and Preparation
#
# This file prepares the selected public fraud-detection dataset for the later ML pipeline.
#
# The aim here is not to train a model. This creates one clean, documented, and leakage-aware table that can be reused by the next notebook or script.
#
# The file covers:
#
# 1. loading the raw transaction dataset,
# 2. checking schema, missing values, duplicates, class imbalance, and numeric ranges,
# 3. applying only safe cleaning steps,
# 4. removing prediction-time leakage and out-of-scope identifiers,
# 5. preserving missing values for train-fitted modeling pipelines,
# 6. saving the cleaned table and data-quality report in this same folder.
#
# Train-test splitting, scaling, one-hot encoding, SMOTE, threshold tuning, and model training are intentionally left for the ML pipeline notebook, where they can be applied without data leakage.

# %%
# This cell imports the standard library and data-analysis packages used throughout the pipeline.
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# %%
# This cell defines project paths, dataset names, expected schema, and the final columns to save.
DEFAULT_BASE_DIR = Path(r"D:\Germany\Documents\Magdeburg\Semester Documents\Sem 5\Thesis\Code snippets")
DATASET_FILE_NAME = "financial_fraud_detection_dataset.csv"
OUTPUT_FILE_NAME = "gold_financial_fraud_detection_table.csv"
DATA_QUALITY_REPORT_FILE_NAME = "data_quality_report_v1.md"
FEATURE_VERSION = "v1"
FEATURE_TABLE_FILE_NAME = f"fraud_features_{FEATURE_VERSION}.parquet"
LABEL_TABLE_FILE_NAME = f"fraud_labels_{FEATURE_VERSION}.parquet"
FEATURE_SCHEMA_FILE_NAME = f"feature_schema_{FEATURE_VERSION}.json"
FEATURE_METADATA_FILE_NAME = f"feature_metadata_{FEATURE_VERSION}.json"

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
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
    "is_fraud",
]

MODEL_FEATURE_COLUMNS = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_used",
    "payment_channel",
    "amount",
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]

FEATURE_HANDOFF_COLUMNS = [
    "transaction_id",
    "event_timestamp",
    *MODEL_FEATURE_COLUMNS,
]

LABEL_HANDOFF_COLUMNS = [
    "transaction_id",
    "event_timestamp",
    "is_fraud",
]


# %%
# This cell finds the dataset and defines the output files.
def build_paths(
    base_dir: Path = DEFAULT_BASE_DIR,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    raw_file = base_dir / DATASET_FILE_NAME

    # Useful when the script is run from the folder that contains the CSV.
    if not raw_file.exists():
        base_dir = Path.cwd()
        raw_file = base_dir / DATASET_FILE_NAME

    output_file = base_dir / OUTPUT_FILE_NAME
    report_file = base_dir / DATA_QUALITY_REPORT_FILE_NAME
    feature_data_dir = base_dir / "feature_repo" / "data"
    feature_metadata_dir = base_dir / "feature_repo" / "metadata"
    feature_table_file = feature_data_dir / FEATURE_TABLE_FILE_NAME
    label_table_file = feature_data_dir / LABEL_TABLE_FILE_NAME
    feature_schema_file = feature_metadata_dir / FEATURE_SCHEMA_FILE_NAME
    feature_metadata_file = feature_metadata_dir / FEATURE_METADATA_FILE_NAME

    print("Dataset file found:", raw_file.exists())
    print("Dataset path:", raw_file)
    print("Gold table path:", output_file)
    print("Data-quality report path:", report_file)
    print("Feast feature table path:", feature_table_file)
    print("Feast label table path:", label_table_file)
    print("Feature schema path:", feature_schema_file)
    print("Feature metadata path:", feature_metadata_file)

    return (
        raw_file,
        output_file,
        report_file,
        feature_table_file,
        label_table_file,
        feature_schema_file,
        feature_metadata_file,
    )


# %%
# This cell reads a small preview and loads the full dataset.
def read_sample(raw_file: Path) -> pd.DataFrame:
    if not raw_file.exists():
        raise FileNotFoundError(
            f"The dataset file was not found. Please check that {DATASET_FILE_NAME} "
            "is in the Code snippets folder or in the same folder as this script."
        )

    sample = pd.read_csv(raw_file, nrows=5)
    print("Sample columns:", sample.columns.tolist())
    return sample


def load_dataset(raw_file: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_file, dtype=DTYPE_MAP)
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

    missing_summary = df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing_count"})
    missing_summary["missing_percentage"] = missing_summary["missing_count"] / len(df) * 100

    duplicate_count = int(df.duplicated().sum())
    print("Exact duplicate rows:", duplicate_count)

    target_distribution = df["is_fraud"].value_counts().rename_axis("is_fraud").reset_index(name="row_count")
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

    numeric_summary = df[NUMERIC_COLUMNS].describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]).T

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


def parse_event_timestamp(
    df_clean: pd.DataFrame, cleaning_stats: dict[str, int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    df_clean = df_clean.copy()
    # Explicit ISO-8601 parsing accepts timestamps both with and without fractional seconds.
    df_clean["event_timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce", format="ISO8601")

    invalid_timestamp_mask = df_clean["event_timestamp"].isna()
    invalid_timestamp_rows = int(invalid_timestamp_mask.sum())
    df_clean = df_clean.loc[~invalid_timestamp_mask].copy()

    cleaning_stats = cleaning_stats.copy()
    cleaning_stats["invalid_timestamp_rows"] = invalid_timestamp_rows

    print("Invalid timestamp rows removed:", invalid_timestamp_rows)
    print("Rows after timestamp cleaning:", len(df_clean))

    return df_clean, cleaning_stats


# %%
# This cell creates the minimal cleaned handoff table. It deliberately does not
# create model features or fill missing values.
def select_clean_columns(df_clean: pd.DataFrame) -> pd.DataFrame:
    return df_clean[COLUMNS_TO_KEEP].copy()


# %%
# This cell saves the final preprocessed table and data-quality report.
def save_gold_table(df_clean_table: pd.DataFrame, output_file: Path) -> Path:
    temporary_file = output_file.with_name(f"{output_file.stem}_temporary{output_file.suffix}")

    try:
        df_clean_table.to_csv(temporary_file, index=False)
        temporary_file.replace(output_file)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not save {output_file}. Close the file if it is open in Excel, "
            "VS Code, or another program, then run this script again. "
            f"The completed temporary table is at {temporary_file}."
        ) from exc

    print("Saved gold table:", output_file)
    return output_file


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_parquet_atomically(table: pd.DataFrame, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(f"{output_file.stem}_temporary{output_file.suffix}")
    table.to_parquet(temporary_file, index=False)
    temporary_file.replace(output_file)
    return output_file


def save_json_atomically(payload: dict, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(f"{output_file.stem}_temporary{output_file.suffix}")
    temporary_file.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return output_file


def build_feature_schema(feature_table: pd.DataFrame, label_table: pd.DataFrame) -> dict:
    feast_types = {
        "transaction_id": "String",
        "event_timestamp": "UnixTimestamp",
        "transaction_type": "String",
        "merchant_category": "String",
        "location": "String",
        "device_used": "String",
        "payment_channel": "String",
        "amount": "Float64",
        "time_since_last_transaction": "Float64",
        "spending_deviation_score": "Float64",
        "velocity_score": "Int64",
        "geo_anomaly_score": "Float64",
        "is_fraud": "Int32",
    }

    def table_schema(table: pd.DataFrame) -> list[dict]:
        return [
            {
                "name": column,
                "pandas_dtype": str(table[column].dtype),
                "feast_dtype": feast_types[column],
                "nullable": bool(table[column].isna().any()),
            }
            for column in table.columns
        ]

    return {
        "feature_version": FEATURE_VERSION,
        "entity": {
            "name": "transaction",
            "join_key": "transaction_id",
        },
        "event_timestamp": "event_timestamp",
        "feature_view": "fraud_transaction_features",
        "feature_service": f"fraud_model_features_{FEATURE_VERSION}",
        "feature_table": table_schema(feature_table),
        "label_table": table_schema(label_table),
        "target_registered_as_feature": False,
    }


def save_feature_store_handoff(
    df_clean_table: pd.DataFrame,
    raw_file: Path,
    feature_table_file: Path,
    label_table_file: Path,
    feature_schema_file: Path,
    feature_metadata_file: Path,
    cleaning_stats: dict[str, int],
) -> list[Path]:
    """Create the versioned, leakage-safe offline-store handoff for Feast."""
    if df_clean_table["transaction_id"].duplicated().any():
        raise ValueError("The Feast transaction entity requires unique transaction_id values.")
    if df_clean_table["is_fraud"].isna().any():
        raise ValueError("The label handoff cannot contain missing is_fraud values.")

    feature_table = df_clean_table[FEATURE_HANDOFF_COLUMNS].copy()
    label_table = df_clean_table[LABEL_HANDOFF_COLUMNS].copy()

    feature_table["transaction_id"] = feature_table["transaction_id"].astype("string")
    label_table["transaction_id"] = label_table["transaction_id"].astype("string")
    for column in [
        "transaction_type",
        "merchant_category",
        "location",
        "device_used",
        "payment_channel",
    ]:
        feature_table[column] = feature_table[column].astype("string")
    feature_table["velocity_score"] = feature_table["velocity_score"].astype("int64")
    label_table["is_fraud"] = label_table["is_fraud"].astype("int8")

    save_parquet_atomically(feature_table, feature_table_file)
    save_parquet_atomically(label_table, label_table_file)

    schema_payload = build_feature_schema(feature_table, label_table)
    save_json_atomically(schema_payload, feature_schema_file)

    pipeline_source = Path(__file__).resolve()
    metadata_payload = {
        "feature_version": FEATURE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_store": "feast_local_file_offline_store",
        "feature_view": "fraud_transaction_features",
        "feature_service": f"fraud_model_features_{FEATURE_VERSION}",
        "entity": "transaction",
        "entity_join_key": "transaction_id",
        "event_timestamp": "event_timestamp",
        "row_count": int(len(feature_table)),
        "fraud_count": int(label_table["is_fraud"].sum()),
        "fraud_rate": float(label_table["is_fraud"].mean()),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "label_column": "is_fraud",
        "target_registered_as_feature": False,
        "cleaning_stats": cleaning_stats,
        "artifacts": {
            "raw_source": {
                "path": str(raw_file.resolve()),
                "size_bytes": int(raw_file.stat().st_size),
                "modified_time_ns": int(raw_file.stat().st_mtime_ns),
                "sha256": sha256_file(raw_file),
            },
            "feature_table": {
                "path": str(feature_table_file.resolve()),
                "size_bytes": int(feature_table_file.stat().st_size),
                "sha256": sha256_file(feature_table_file),
            },
            "label_table": {
                "path": str(label_table_file.resolve()),
                "size_bytes": int(label_table_file.stat().st_size),
                "sha256": sha256_file(label_table_file),
            },
            "feature_schema": {
                "path": str(feature_schema_file.resolve()),
                "sha256": sha256_file(feature_schema_file),
            },
            "data_pipeline_source": {
                "path": str(pipeline_source),
                "sha256": sha256_file(pipeline_source),
            },
        },
        "leakage_boundary": {
            "allowed_in_handoff": (
                "Record-level values available at prediction time; the target is "
                "stored separately and is not registered as a Feast feature."
            ),
            "kept_in_ml_pipeline": [
                "learned imputation",
                "scaling",
                "categorical encoding",
                "resampling or SMOTE",
                "target-based feature selection",
                "decision-threshold tuning",
            ],
        },
        "limitations": [
            "The source is a static public/synthetic transaction dataset.",
            "transaction_id is used as an offline prototype entity, not as a reusable customer entity.",
            "The local Feast repository is a prototype and not a production feature platform.",
        ],
    }
    save_json_atomically(metadata_payload, feature_metadata_file)

    print("Saved Feast feature table:", feature_table_file)
    print("Saved Feast label table:", label_table_file)
    print("Saved feature schema:", feature_schema_file)
    print("Saved feature metadata:", feature_metadata_file)
    return [
        feature_table_file,
        label_table_file,
        feature_schema_file,
        feature_metadata_file,
    ]


def save_quality_report(
    df_clean_table: pd.DataFrame,
    raw_file: Path,
    report_file: Path,
    cleaning_stats: dict[str, int],
    quality_profile: dict[str, pd.DataFrame | int],
) -> Path:
    missing_summary = quality_profile["missing_summary"]
    missing_summary = missing_summary.loc[missing_summary["missing_count"] > 0]

    if missing_summary.empty:
        missing_text = "No missing values were found in the raw dataset."
    else:
        missing_text = "\n".join(
            f"- {row.column}: {int(row.missing_count)} missing values ({row.missing_percentage:.4f}%)"
            for row in missing_summary.itertuples(index=False)
        )

    target_distribution = quality_profile["target_distribution"]
    target_text = "\n".join(
        f"- is_fraud={int(row.is_fraud)}: {int(row.row_count)} rows ({row.percentage:.4f}%)"
        for row in target_distribution.itertuples(index=False)
    )

    transaction_type_summary = quality_profile["type_summary"].head(10)
    transaction_type_text = "\n".join(
        f"- {row.transaction_type}: {int(row.row_count)} rows, fraud rate {row.fraud_rate:.6f}"
        for row in transaction_type_summary.itertuples(index=False)
    )

    numeric_summary = quality_profile["numeric_summary"]
    numeric_text = numeric_summary[["mean", "std", "min", "50%", "max"]].round(4).to_string()

    report = f"""# Data Quality Report

## Dataset Overview

- Source file: `{raw_file}`
- Rows before cleaning: {cleaning_stats["initial_rows"]}
- Rows after cleaning: {len(df_clean_table)}
- Columns after cleaning: {df_clean_table.shape[1]}
- Fraud count after cleaning: {int(df_clean_table["is_fraud"].sum())}
- Fraud rate after cleaning: {df_clean_table["is_fraud"].mean():.6f}
- Final table: `{OUTPUT_FILE_NAME}`

## Missing Values Before Cleaning

{missing_text}

## Target Distribution Before Cleaning

{target_text}

## Cleaning Decisions

- Exact duplicate rows removed: {cleaning_stats["duplicate_rows_removed"]}
- Negative amount rows removed: {cleaning_stats["invalid_amount_rows"]}
- Invalid timestamp rows removed: {cleaning_stats["invalid_timestamp_rows"]}
- Direct identifier columns `sender_account`, `receiver_account`, `ip_address`, and `device_hash` were removed from the final table.
- `fraud_type` was excluded because it is label-derived information, not an input available at prediction time.
- Missing `time_since_last_transaction` values were preserved for train-fitted imputation.
- No engineered features were added. Log amount, temporal fields, and a missingness flag remain separate feature experiments after the baseline.
- A versioned Feast offline-store handoff is produced as separate Parquet feature and label tables.
- `is_fraud` is stored in the label table and is not registered as a Feast feature.

## Transaction Type Summary Before Cleaning

{transaction_type_text}

## Numeric Summary Before Cleaning

```text
{numeric_text}
```

## Leakage-Control Decision

Only schema validation, conservative row checks, ISO-8601 timestamp parsing, and column selection were applied before the train-validation-test split.

The following steps are postponed to the ML pipeline:

- learned imputation
- scaling or normalization
- one-hot encoding
- SMOTE or oversampling
- log-amount feature creation
- hour, weekday, or month feature creation
- missingness-indicator feature creation
- feature selection using the target
- threshold tuning
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print("Saved data-quality report:", report_file)
    return report_file


def verify_outputs(df_clean_table: pd.DataFrame, output_files: list[Path]) -> None:
    print("Cleaned dataset shape:", df_clean_table.shape)
    print("Preserved missing values:", int(df_clean_table.isna().sum().sum()))
    print("Fraud rate:", df_clean_table["is_fraud"].mean())

    for output_file in output_files:
        print("Output file created:", output_file.exists(), output_file)


# %%
# This cell runs the full preprocessing pipeline in the correct order.
def main() -> None:
    (
        raw_file,
        output_file,
        report_file,
        feature_table_file,
        label_table_file,
        feature_schema_file,
        feature_metadata_file,
    ) = build_paths()
    read_sample(raw_file)

    df = load_dataset(raw_file)
    df = standardise_columns(df)
    validate_schema(df)
    quality_profile = profile_dataset(df)

    df_clean, cleaning_stats = apply_safe_cleaning(df)
    df_clean, cleaning_stats = parse_event_timestamp(df_clean, cleaning_stats)

    df_clean_table = select_clean_columns(df_clean)

    quality_report_file = save_quality_report(
        df_clean_table,
        raw_file,
        report_file,
        cleaning_stats,
        quality_profile,
    )
    gold_table_file = save_gold_table(df_clean_table, output_file)
    feature_handoff_files = save_feature_store_handoff(
        df_clean_table,
        raw_file,
        feature_table_file,
        label_table_file,
        feature_schema_file,
        feature_metadata_file,
        cleaning_stats,
    )
    verify_outputs(
        df_clean_table,
        [gold_table_file, quality_report_file, *feature_handoff_files],
    )


# %%
# This cell keeps the file executable as a normal Python script and avoids confusing
# NameError messages when the final cell is run before the earlier cells in an IDE.
REQUIRED_PIPELINE_NAMES = [
    "build_paths",
    "read_sample",
    "load_dataset",
    "standardise_columns",
    "validate_schema",
    "profile_dataset",
    "apply_safe_cleaning",
    "parse_event_timestamp",
    "select_clean_columns",
    "save_gold_table",
    "save_feature_store_handoff",
    "save_quality_report",
    "verify_outputs",
    "main",
]

missing_pipeline_names = [name for name in REQUIRED_PIPELINE_NAMES if name not in globals()]

if __name__ == "__main__":
    if missing_pipeline_names:
        print("Run the earlier cells first. Missing definitions:", ", ".join(missing_pipeline_names))
    else:
        main()
