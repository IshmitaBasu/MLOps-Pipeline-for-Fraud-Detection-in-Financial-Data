"""Feast definitions for the versioned fraud-detection feature handoff."""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String

from sender_location_feature_store import build_behaviour_feast_objects

PROJECT_DIR = Path(__file__).resolve().parent
FEATURE_VERSION = "v1"
FEATURE_DATA_PATH = PROJECT_DIR / "feature_repo" / "data" / f"fraud_features_{FEATURE_VERSION}.parquet"

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    value_type=ValueType.STRING,
    description="Unique transaction identifier used for the offline prototype handoff.",
    tags={"scope": "offline_prototype"},
)

fraud_transaction_source = FileSource(
    name=f"fraud_transaction_source_{FEATURE_VERSION}",
    path=str(FEATURE_DATA_PATH),
    timestamp_field="event_timestamp",
    description="Versioned leakage-safe transaction features produced by the data pipeline.",
)

fraud_transaction_features = FeatureView(
    name="fraud_transaction_features",
    entities=[transaction],
    source=fraud_transaction_source,
    ttl=timedelta(days=3650),
    online=False,
    offline=True,
    schema=[
        Field(name="transaction_type", dtype=String),
        Field(name="merchant_category", dtype=String),
        Field(name="location", dtype=String),
        Field(name="device_used", dtype=String),
        Field(name="payment_channel", dtype=String),
        Field(name="amount", dtype=Float64),
        Field(name="time_since_last_transaction", dtype=Float64),
        Field(name="spending_deviation_score", dtype=Float64),
        Field(name="velocity_score", dtype=Int64),
        Field(name="geo_anomaly_score", dtype=Float64),
    ],
    description=(
        "Original cleaned transaction predictors. Learned preprocessing remains "
        "inside the train-fitted scikit-learn pipeline."
    ),
    tags={
        "feature_version": FEATURE_VERSION,
        "target_excluded": "true",
        "environment": "local_prototype",
    },
)

fraud_model_features_v1 = FeatureService(
    name=f"fraud_model_features_{FEATURE_VERSION}",
    features=[fraud_transaction_features],
    description="Feature contract used by the baseline comparison and later fraud models.",
    tags={"feature_version": FEATURE_VERSION},
)

# Version 2 behavioural state is additive: the original v1 transaction contract
# remains unchanged and usable by existing experiments.
BEHAVIOUR_FEATURE_VERSION = "v2"
SENDER_HISTORY_DATA_PATH = (
    PROJECT_DIR
    / "feature_repo"
    / "data"
    / f"sender_history_features_{BEHAVIOUR_FEATURE_VERSION}.parquet"
)
SENDER_LOCATION_HISTORY_DATA_PATH = (
    PROJECT_DIR
    / "feature_repo"
    / "data"
    / f"sender_location_history_features_{BEHAVIOUR_FEATURE_VERSION}.parquet"
)

(
    sender,
    sender_location,
    sender_history_features_v2,
    sender_location_history_features_v2,
    fraud_behaviour_features_v2,
) = build_behaviour_feast_objects(
    SENDER_HISTORY_DATA_PATH,
    SENDER_LOCATION_HISTORY_DATA_PATH,
)
