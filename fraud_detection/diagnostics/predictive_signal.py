"""Descriptive training-only signal diagnostics; no estimator fitting or held-out evaluation."""

# %% Imports and predeclared scope
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_detection.common.modeling import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from fraud_detection.diagnostics.predictive_quality import ID_COLUMN, TARGET_COLUMN, TIME_COLUMN, TRAIN_CODE
from fraud_detection.features.sender_location_history import (
    HISTORY_FEATURES,
    _strictly_earlier_training_count,
    validate_history_features,
)

NUMERICAL = [*NUMERIC_FEATURES, *HISTORY_FEATURES]
MINIMUM_GROUP_ROWS = 1_000
MINIMUM_FRAUD_ROWS = 30
MISSING = "<MISSING_OR_NONFINITE>"


# %% Partition isolation and label-free strictly earlier history
def extract_training_rows(metadata: pd.DataFrame, partition: np.ndarray) -> pd.DataFrame:
    if partition.ndim != 1 or len(metadata) != len(partition) or not np.isin(partition, [0, 1, 2]).all():
        raise ValueError("A matching frozen partition is required.")
    training = metadata.loc[partition == TRAIN_CODE].copy()
    if training.empty:
        raise ValueError("No training rows were selected.")
    return training


def add_training_history(training: pd.DataFrame) -> pd.DataFrame:
    if training[TIME_COLUMN].isna().any() or training["sender_account"].isna().any():
        raise ValueError("History requires valid training timestamps and sender keys.")
    ordered = training[[ID_COLUMN, TIME_COLUMN, "sender_account", "location"]].copy()
    ordered["_is_training_event"] = np.ones(len(ordered), dtype="int8")
    ordered = ordered.sort_values([TIME_COLUMN, ID_COLUMN], kind="mergesort")
    sender = _strictly_earlier_training_count(ordered, ["sender_account"])
    pair = _strictly_earlier_training_count(ordered, ["sender_account", "location"])
    result = training.copy()
    result[HISTORY_FEATURES[0]] = sender.reindex(result.index).astype("int32")
    result[HISTORY_FEATURES[1]] = pair.reindex(result.index).astype("int32")
    result[HISTORY_FEATURES[2]] = ((result[HISTORY_FEATURES[0]] > 0) & (result[HISTORY_FEATURES[1]] == 0)).astype(
        "int8"
    )
    validate_history_features(result)
    return result


# %% Rates, nominal uncertainty and practical association sizes
def wilson_interval(successes, totals):
    successes, totals = np.asarray(successes, dtype=float), np.asarray(totals, dtype=float)
    if (totals <= 0).any() or (successes < 0).any() or (successes > totals).any():
        raise ValueError("Wilson intervals require valid positive group sizes.")
    z = 1.959963984540054
    rate = successes / totals
    denominator = 1 + z * z / totals
    centre = (rate + z * z / (2 * totals)) / denominator
    width = z * np.sqrt(rate * (1 - rate) / totals + z * z / (4 * totals * totals)) / denominator
    return np.minimum(np.maximum(centre - width, 0), rate), np.maximum(np.minimum(centre + width, 1), rate)


def group_rates(labels: pd.Series, groups: pd.Series, feature: str, slice_name="all_training") -> pd.DataFrame:
    if (
        len(labels) != len(groups)
        or not labels.index.equals(groups.index)
        or labels.isna().any()
        or not labels.isin([0, 1]).all()
    ):
        raise ValueError("Group rates require aligned binary labels and groups.")
    frame = pd.DataFrame({"group": groups.astype("string").fillna(MISSING).to_numpy(), "fraud": labels.to_numpy()})
    table = (
        frame.groupby("group", observed=True, dropna=False, sort=True)
        .fraud.agg(rows="size", fraud_rows="sum")
        .reset_index()
    )
    table["fraud_rate"] = table.fraud_rows / table.rows
    table["ci_low"], table["ci_high"] = wilson_interval(table.fraud_rows, table.rows)
    baseline = float(labels.mean())
    table["slice_fraud_rate"] = baseline
    table["fraud_lift"] = table.fraud_rate / baseline if baseline else np.nan
    table["rate_difference_pp"] = (table.fraud_rate - baseline) * 100
    table["coverage"] = table.rows / len(labels)
    table["small_group"] = table.rows < MINIMUM_GROUP_ROWS
    table["few_fraud_cases"] = table.fraud_rows < MINIMUM_FRAUD_ROWS
    table.insert(0, "slice", slice_name)
    table.insert(1, "feature", feature)
    return table


def cramers_v(table: pd.DataFrame) -> float:
    observed = np.column_stack([table.rows - table.fraud_rows, table.fraud_rows]).astype(float)
    observed = observed[observed.sum(axis=1) > 0]
    if len(observed) < 2 or (observed.sum(axis=0) == 0).any():
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    return float(np.sqrt((((observed - expected) ** 2 / expected).sum()) / observed.sum()))


def quantile_groups(values: pd.Series, bins=10) -> tuple[pd.Series, list[float]]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    edges = np.unique(finite.dropna().quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    edges = edges[np.isfinite(edges)]
    if len(edges) < 2:
        groups = pd.Series("constant", index=values.index, dtype="string").mask(finite.isna(), MISSING)
    elif finite.nunique() <= bins:
        # Preserve separate 0/1 values and low-cardinality counts rather than merging them into one interval.
        groups = finite.astype("string").fillna(MISSING)
    else:
        groups = pd.cut(finite, bins=edges, include_lowest=True, duplicates="drop").astype("string").fillna(MISSING)
    return groups, edges.tolist()


# %% Quality audit without asserting that differing labels are necessarily errors
def quality_audit(training: pd.DataFrame) -> dict:
    content = [TIME_COLUMN, "sender_account", *FEATURE_COLUMNS]
    repeated = training.duplicated(subset=content, keep=False)
    repeats = training.loc[repeated, [*content, TARGET_COLUMN]]
    conflicts = (
        int((repeats.groupby(content, dropna=False, observed=True)[TARGET_COLUMN].nunique() > 1).sum())
        if len(repeats)
        else 0
    )
    numeric = training[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    return {
        "training_rows": len(training),
        "training_fraud_rows": int(training[TARGET_COLUMN].sum()),
        "training_fraud_rate": float(training[TARGET_COLUMN].mean()),
        "missing_by_column": training.isna().sum().astype(int).to_dict(),
        "duplicate_transaction_ids": int(training[ID_COLUMN].duplicated().sum()),
        "exact_repeat_rows_without_id": int(
            training.duplicated(subset=[column for column in training if column != ID_COLUMN]).sum()
        ),
        "rows_sharing_selected_transaction_content": int(repeated.sum()),
        "selected_content_groups_with_conflicting_labels": conflicts,
        "conflict_interpretation": "Same selected observables with different labels is a screen, not proof of mislabelling.",
        "invalid_numeric_text": {
            column: int((training[column].notna() & numeric[column].isna()).sum()) for column in NUMERIC_FEATURES
        },
        "infinite_numeric_values": {column: int(np.isinf(numeric[column]).sum()) for column in NUMERIC_FEATURES},
        "negative_amount_rows": int((numeric.amount < 0).sum()),
        "invalid_timestamps": int(training[TIME_COLUMN].isna().sum()),
        "fraud_type_audit": (
            pd.crosstab(training[TARGET_COLUMN], training.fraud_type.notna()).to_dict()
            if "fraud_type" in training
            else "column unavailable"
        ),
        "excluded_predictors": [ID_COLUMN, "fraud_type", "sender_account", TIME_COLUMN],
    }


# %% Bounded feature, interaction and stability diagnostics
def analyse_training(training: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    if training[TARGET_COLUMN].nunique() != 2 or not training[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Diagnostics require both valid binary target classes.")
    quality = quality_audit(training)
    labels = training[TARGET_COLUMN]
    group_definitions = {column: training[column].astype("string").fillna(MISSING) for column in CATEGORICAL_FEATURES}
    quantiles, effects, edges = [], [], {}
    for column in NUMERICAL:
        values = pd.to_numeric(training[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        groups, boundaries = quantile_groups(values)
        group_definitions[column] = groups
        edges[column] = boundaries
        rates = group_rates(labels, groups, column)
        observed = rates.loc[rates.group != MISSING]
        totals = np.column_stack([observed.rows - observed.fraud_rows, observed.fraud_rows]).astype(float)
        distance = (
            float(np.abs(totals[:, 0] / totals[:, 0].sum() - totals[:, 1] / totals[:, 1].sum()).sum() / 2)
            if len(totals) and (totals.sum(axis=0) > 0).all()
            else None
        )
        valid = values.notna()
        correlation = (
            values[valid].corr(labels[valid])
            if values[valid].nunique() > 1 and labels[valid].nunique() == 2
            else np.nan
        )
        effects.append(
            {
                "feature": column,
                "finite_rows": int(valid.sum()),
                "pearson_point_biserial": float(correlation) if pd.notna(correlation) else None,
                "finite_quantile_binned_total_variation": distance,
            }
        )
        for target in (0, 1):
            subset = values.loc[labels == target].dropna()
            row = {
                "feature": column,
                "class": target,
                "finite_rows": len(subset),
                "mean": float(subset.mean()) if len(subset) else None,
                "std": float(subset.std()) if len(subset) > 1 else None,
            }
            row.update(
                {
                    f"q{int(q * 100):02d}": float(subset.quantile(q)) if len(subset) else None
                    for q in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
                }
            )
            quantiles.append(row)
    missing_time = training.time_since_last_transaction.isna()
    group_definitions["time_since_last_transaction_missing"] = missing_time.map({True: "missing", False: "observed"})
    amount_groups, amount_edges = quantile_groups(training.amount, bins=5)
    for category in ("transaction_type", "payment_channel"):
        group_definitions[f"{category}_x_amount_quintile"] = (
            training[category].astype("string").fillna(MISSING) + " | " + amount_groups
        )
    groups = pd.concat(
        [group_rates(labels, values, feature) for feature, values in group_definitions.items()], ignore_index=True
    )
    associations = []
    for feature in group_definitions:
        table = groups.loc[groups.feature == feature]
        associations.append(
            {
                "feature": feature,
                "cramers_v": cramers_v(table),
                "groups": len(table),
                "max_absolute_rate_difference_pp": float(table.rate_difference_pp.abs().max()),
            }
        )
    times = training[TIME_COLUMN]
    if times.isna().any():
        raise ValueError("Stability diagnostics require valid training timestamps.")
    if times.nunique() > 1:
        cuts = np.linspace(times.astype("int64").min(), times.astype("int64").max(), 5)
        periods = pd.cut(
            times.astype("int64"), cuts, labels=[f"time_{i}" for i in range(1, 5)], include_lowest=True
        ).astype("string")
    else:
        cuts = [int(times.astype("int64").iloc[0])]
        periods = pd.Series("time_1", index=training.index, dtype="string")
    hashes = pd.util.hash_pandas_object(training[ID_COLUMN].astype("string"), index=False).to_numpy()
    halves = pd.Series(np.where(hashes % 2 == 0, "hash_half_A", "hash_half_B"), index=training.index)
    stability = []
    for slice_column in (periods, halves):
        for name in sorted(slice_column.dropna().unique()):
            mask = slice_column == name
            for feature, values in group_definitions.items():
                stability.append(group_rates(labels.loc[mask], values.loc[mask], feature, name))
    tables = {
        "group_rates": groups,
        "numerical_class_quantiles": pd.DataFrame(quantiles),
        "numerical_effect_sizes": pd.DataFrame(effects),
        "categorical_binned_associations": pd.DataFrame(associations),
        "stability_group_rates": pd.concat(stability, ignore_index=True),
    }
    summary = {
        "scope": "training_partition_only",
        "quality": quality,
        "quantile_edges_training_only": edges,
        "interaction_amount_edges": amount_edges,
        "time_slice_boundaries_utc": [pd.Timestamp(int(cut), tz="UTC").isoformat() for cut in cuts],
        "group_flags": {"small_group_rows_below": MINIMUM_GROUP_ROWS, "few_fraud_cases_below": MINIMUM_FRAUD_ROWS},
        "uncertainty": "Nominal 95% marginal Wilson intervals; no independence or multiple-testing guarantee.",
        "interpretation": "Descriptive associations, not causal explanations or independently validated predictive performance.",
        "validation_analysed": False,
        "test_analysed": False,
        "model_trained": False,
    }
    return summary, tables, group_definitions


# %% Aggregate-only evidence plots; never export transaction-level records
def save_signal_plots(training: pd.DataFrame, tables: dict, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    rates = tables["group_rates"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feature in zip(axes.flat, [*CATEGORICAL_FEATURES, "time_since_last_transaction_missing"]):
        subset = rates.loc[rates.feature == feature]
        y = subset.fraud_rate.to_numpy() * 100
        ax.errorbar(
            np.arange(len(subset)),
            y,
            yerr=[y - subset.ci_low.to_numpy() * 100, subset.ci_high.to_numpy() * 100 - y],
            fmt="o",
            capsize=3,
        )
        ax.set_xticks(np.arange(len(subset)), subset.group, rotation=35, ha="right")
        ax.axhline(float(training[TARGET_COLUMN].mean()) * 100, color="grey", linestyle="--")
        ax.set_title(feature.replace("_", " "))
        ax.set_ylabel("Fraud rate (%)")
        ax.set_ylim(bottom=0)
    fig.suptitle("Training-only category and missingness associations | nominal 95% Wilson intervals")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "categorical_missingness_rates.png")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, feature in zip(axes.flat, NUMERICAL):
        for label, colour in ((0, "#3777a8"), (1, "#cf6a32")):
            values = (
                pd.to_numeric(training.loc[training[TARGET_COLUMN] == label, feature], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(values):
                probabilities = np.linspace(0.001, 0.999, 300)
                ax.plot(
                    values.quantile(probabilities),
                    probabilities,
                    label="fraud" if label else "legitimate",
                    color=colour,
                )
        ax.set_title(feature.replace("_", " "))
        ax.set_ylabel("Cumulative share within class")
        ax.legend(fontsize=8)
        if feature in ("amount", *HISTORY_FEATURES[:2]):
            ax.set_xscale("symlog", linthresh=1)
    for ax in list(axes.flat)[len(NUMERICAL) :]:
        ax.set_visible(False)
    fig.suptitle("Training-only numerical distributions | finite values, 0.1–99.9% quantile range")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "numerical_class_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, feature in zip(axes, ("transaction_type_x_amount_quintile", "payment_channel_x_amount_quintile")):
        subset = rates.loc[rates.feature == feature].copy()
        parts = subset.group.str.split(" | ", n=1, regex=False, expand=True)
        subset["category"], subset["amount_band"] = parts[0].to_numpy(), parts[1].to_numpy()
        matrix = subset.pivot(index="category", columns="amount_band", values="fraud_rate") * 100
        heat = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0)
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
        ax.set_title(feature.replace("_", " "))
        for row in range(len(matrix)):
            for column in range(len(matrix.columns)):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(heat, ax=ax, label="Fraud rate (%)")
    fig.suptitle("Training-only predeclared interactions | consult CSV support flags before interpreting cells")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "bounded_interaction_rates.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stability = tables["stability_group_rates"]
    missing = stability.loc[stability.feature == "time_since_last_transaction_missing"]
    for ax, prefix in zip(axes, ("time_", "hash_half_")):
        selected = missing.loc[missing.slice.str.startswith(prefix)]
        for group in ("observed", "missing"):
            subset = selected.loc[selected.group == group].sort_values("slice")
            if len(subset):
                ax.errorbar(
                    subset.slice,
                    subset.fraud_rate * 100,
                    yerr=[(subset.fraud_rate - subset.ci_low) * 100, (subset.ci_high - subset.fraud_rate) * 100],
                    marker="o",
                    capsize=3,
                    label=group,
                )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Fraud rate (%)")
        ax.set_title("Training time slices" if prefix == "time_" else "Fixed training hash halves")
        ax.legend()
    fig.suptitle("Training-only stability of time-since-last-transaction missingness")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "missingness_stability.png")
    return paths
