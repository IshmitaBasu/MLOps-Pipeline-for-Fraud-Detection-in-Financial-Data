# Fraud Feature Definitions — v1

This document defines the first versioned feature contract used by the local Feast offline-store prototype. The executable Feast definitions are maintained in `fraud_feature_definitions.py`.

## Entity and time

- Entity: `transaction`
- Join key: `transaction_id`
- Event timestamp: `event_timestamp`
- Scope: offline, transaction-level thesis prototype

`transaction_id` is unique per transaction. It is suitable for reconstructing the historical training table, but it is not presented as a reusable customer or account entity for a production feature platform.

## Registered features

| Feature                       | Type              | Definition                                                   |
| ----------------------------- | ----------------- | ------------------------------------------------------------ |
| `transaction_type`            | String            | Transaction category supplied in the source record.          |
| `merchant_category`           | String            | Merchant category supplied in the source record.             |
| `location`                    | String            | Transaction location category supplied in the source record. |
| `device_used`                 | String            | Device category supplied in the source record.               |
| `payment_channel`             | String            | Payment-channel category supplied in the source record.      |
| `amount`                      | Float64           | Original transaction amount.                                 |
| `time_since_last_transaction` | Float64, nullable | Original time-since-last-transaction value.                  |
| `spending_deviation_score`    | Float64           | Original spending-deviation score.                           |
| `velocity_score`              | Int64             | Original transaction-velocity score.                         |
| `geo_anomaly_score`           | Float64           | Original geographic-anomaly score.                           |

## Label boundary

`is_fraud` is stored in a separate versioned label table. It is deliberately not registered as a Feast feature. `fraud_type`, direct account identifiers, IP addresses, and device hashes remain excluded.

## ML-pipeline boundary

The Feast handoff contains record-level values only. Median imputation, scaling, categorical encoding, resampling, target-based feature selection, and decision-threshold tuning remain inside the leakage-aware ML pipeline and are fitted using the appropriate training or validation partition.
