"""Render documentation figures from saved results only; never fit or score a model."""

# %% Imports, immutable source references and appearance
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from fraud_detection.paths import DOCUMENTATION_IMAGES_DIR, REPOSITORY_ROOT

PROJECT_DIR = REPOSITORY_ROOT
OUTPUT_DIR = DOCUMENTATION_IMAGES_DIR / "experiment_figures"
RUNS = {
    "07": ("95a914f6b68f4faba00dc2132e6ec790", "feature_feasibility_decision.json"),
    "08": ("72fcb6b0a5394bcea26ddd2b75a79c61", "sender_location_feature_comparison.json"),
    "10": ("7d1e57bd9c0a49a59e0898bb3c7bad21", "oversampling_full_data_summary.json"),
    "11": ("ba890331d5a149f0bb17ee1a7f63c589", "model_family_full_data_summary.json"),
    "12": ("cceb57941a914ba2b3e015573b3e34b1", "lightgbm_tuning_full_data_summary.json"),
    "13": ("5920c17849574c11b4854c3c5290bc41", "voting_full_data_summary.json"),
    "13_hard": ("c7aadb13580746b4a58018aa8e506deb", "hard_voting_full_data_summary.json"),
    "14": ("37b41018f28646598344fbcfe4ea4db6", "final_test_evaluation.json"),
}
BLUE, GREEN, GREY = "#3777a8", "#26836b", "#a7b3bf"
FIGURES: list[dict] = []
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_result(stage: str) -> dict:
    run, filename = RUNS[stage]
    data = json.loads((PROJECT_DIR / "mlartifacts" / run / "artifacts" / filename).read_text(encoding="utf-8"))
    if stage != "07" and data["evidence_scope"] != "full_data":
        raise ValueError("Documentation must use the recorded full-data run, not smoke evidence.")
    return data


def save_figure(fig, filename: str, stages: list[str], description: str, extra_sources: list[Path] | None = None):
    sources = [PROJECT_DIR / "mlartifacts" / RUNS[stage][0] / "artifacts" / RUNS[stage][1] for stage in stages]
    sources.extend(extra_sources or [])
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(
        {
            "image": filename,
            "description": description,
            "sources": [
                {
                    "path": str(path.relative_to(PROJECT_DIR)).replace("\\", "/"),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sources
            ],
        }
    )


# %% Figure helpers: scales, operational comparisons and confusion counts
def bar_panel(
    ax,
    labels: list[str],
    values: list[float],
    title: str,
    retained: int | None = None,
    percentage: bool = False,
    reference: float | None = None,
):
    plotted = np.asarray(values) * (100 if percentage else 1)
    positions = np.arange(len(labels))
    ax.barh(positions, plotted, color=[GREEN if i == retained else BLUE for i in positions])
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(plotted.max()) * 1.32, 1e-6))
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_xlabel("Recall (%)" if percentage else "Average Precision (AP)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for pos, value in zip(positions, plotted):
        label = f"{value:.3f}%" if percentage else f"{value:.6f}"
        ax.text(value + ax.get_xlim()[1] * 0.015, pos, label, va="center", fontsize=9)
    if reference is not None:
        ax.axvline(reference, color="#555555", linestyle="--", linewidth=1)


def comparison_figure(stage: str, rows: list[dict], key: str, ap_key: str, retained_name: str, labels=None):
    labels = labels or [row[key].replace("_", " ") for row in rows]
    retained = next(i for i, row in enumerate(rows) if row[key] == retained_name)
    fig, axes = plt.subplots(1, 3, figsize=(17, max(4.2, len(rows) * 0.55 + 1.5)))
    panels = [
        (ap_key, "Validation ranking", False),
        ("workload_5pct_fraud_count_recall", "Fraud-count recall at 5% review", True),
        ("workload_5pct_fraud_value_recall", "Fraud-value recall at 5% review", True),
    ]
    prevalence = read_result(stage)["split_manifest"]["splits"]["validation"]["fraud_rate"]
    for ax, (metric, title, percent) in zip(axes, panels):
        bar_panel(
            ax, labels, [row[metric] for row in rows], title, retained, percent, prevalence if not percent else None
        )
    fig.suptitle(f"Stage {stage}: full-data validation comparison", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.015,
        "Green = retained/selected candidate; all axes start at zero. Dashed AP line = validation fraud prevalence. "
        "Point estimates only; no uncertainty intervals were recorded.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), w_pad=2)
    save_figure(
        fig, f"{stage}_validation_comparison.png", [stage], "Validation AP and equal-5%-workload count/value recall."
    )


def confusion_from_saved_metrics(metrics: dict, fraud_rows: int, rows: int) -> np.ndarray:
    """Reconstruct historical integer counts only when they reproduce saved full-precision metrics."""
    recall, precision = metrics["validation_recall_at_max_f1"], metrics["validation_precision_at_max_f1"]
    tp = round(recall * fraud_rows)
    fp = round(tp / precision) - tp
    fn, tn = fraud_rows - tp, rows - fraud_rows - fp
    matrix = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    if (matrix < 0).any() or not np.isclose(tp / (tp + fp), precision, rtol=0, atol=1e-12):
        raise ValueError("Historical confusion reconstruction does not reproduce precision.")
    if not np.isclose(tp / (tp + fn), recall, rtol=0, atol=1e-12):
        raise ValueError("Historical confusion reconstruction does not reproduce recall.")
    return matrix


def draw_confusion(ax, matrix: np.ndarray, title: str, row_normalised: bool = True):
    if matrix.shape != (2, 2) or (matrix < 0).any() or (matrix.sum(axis=1) == 0).any():
        raise ValueError("A nonnegative two-class confusion matrix with both classes is required.")
    values = matrix / matrix.sum(axis=1, keepdims=True) * 100 if row_normalised else matrix
    maximum = 100 if row_normalised else float(matrix.max())
    heatmap = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraudulent"])
    ax.set_xlabel("Predicted decision")
    ax.set_ylabel("Actual class")
    ax.set_title(title, pad=12)
    names = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for column in range(2):
            annotation = f"{names[row][column]}: {matrix[row, column]:,}"
            if row_normalised:
                annotation += f"\n{values[row, column]:.2f}% of actual class"
            ax.text(
                column,
                row,
                annotation,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if values[row, column] > maximum * 0.52 else "#182733",
            )
    return heatmap


# %% Full-data evidence figures, without accessing datasets or model artifacts
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.clear()
    seven = read_result("07")
    entities = seven["entity_decisions"]
    labels = [row["entity"].replace("_", " ") for row in entities]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, metric, threshold, title in zip(
        axes,
        ["history_coverage", "relative_fraud_rate_difference"],
        [
            seven["screening_thresholds"]["minimum_prior_history_coverage"],
            seven["screening_thresholds"]["minimum_relative_fraud_rate_difference"],
        ],
        ["Earlier-history coverage", "Relative fraud-rate difference"],
    ):
        for i, row in enumerate(entities):
            value = row[metric]
            valid = value is not None and np.isfinite(value)
            color = (
                GREEN if row["passes_initial_screening_gate"] else BLUE if row["sufficient_fraud_evidence"] else GREY
            )
            ax.barh(i, value * 100 if valid else 0, color=color)
            ax.text(
                (value * 100 if valid else 0) + 1,
                i,
                f"{value * 100:.3f}%" if valid else "Unavailable",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 118)
        ax.axvline(threshold * 100, color="#555555", linestyle="--")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Percentage (%)")
    fig.suptitle("Stage 07: training-only feature screening", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.01,
        "Green = all gates passed; grey = insufficient fraud evidence. A large difference with almost no history is not reliable evidence.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(
        fig,
        "07_history_screening.png",
        ["07"],
        "Training-only coverage and relative fraud-rate separation, with screening gates.",
    )

    eight = read_result("08")
    groups = list(eight["group_results"].values())
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bar_panel(
        ax,
        ["Original features", "Original + behavioural history"],
        [row["validation_pr_auc"] for row in groups],
        "Stage 08: controlled feature comparison",
        retained=1,
        reference=eight["split_manifest"]["splits"]["validation"]["fraud_rate"],
    )
    fig.text(
        0.02,
        0.015,
        "Same Random Forest configuration and split; full-data validation only. Dashed line = fraud prevalence.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(
        fig, "08_history_feature_comparison.png", ["08"], "Validation AP before and after adding behavioural history."
    )

    comparison_figure("10", read_result("10")["strategies"], "strategy", "validation_pr_auc", "class_weight_reference")
    eleven = read_result("11")
    # Sampled KNN has a different evaluation population: do not place it among full-data bars.
    full_models = [row for row in eleven["model_results"] if row["full_data_candidate"]]
    comparison_figure("11", full_models, "model", "validation_average_precision", "lightgbm")
    split = eleven["split_manifest"]["splits"]["validation"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, row in zip(axes.flat, full_models):
        matrix = confusion_from_saved_metrics(row, split["fraud_rows"], split["rows"])
        draw_confusion(ax, matrix, row["model"].replace("_", " "))
    fig.suptitle("Stage 11: validation confusion matrices at each model's own maximum-F1 rule", fontsize=15)
    fig.text(
        0.02,
        0.01,
        "Historical counts reconstructed from saved precision/recall and class totals. Colour = percentage of each actual class. Different alert volumes; not equal-workload comparisons.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=3)
    save_figure(
        fig,
        "11_validation_confusion_matrices.png",
        ["11"],
        "Reconstructed validation confusion counts and row-normalised rates for full-data candidates.",
    )

    twelve = read_result("12")
    comparison_figure(
        "12", twelve["configuration_results"], "configuration", "validation_average_precision", "lgbm_reference"
    )
    split = twelve["split_manifest"]["splits"]["validation"]
    reference = next(row for row in twelve["configuration_results"] if row["configuration"] == "lgbm_reference")
    fig, ax = plt.subplots(figsize=(7, 6))
    heat = draw_confusion(
        ax,
        confusion_from_saved_metrics(reference, split["fraud_rows"], split["rows"]),
        "Stage 12: retained LightGBM validation confusion",
    )
    fig.colorbar(heat, ax=ax, label="Percentage of each actual class")
    fig.text(
        0.03,
        0.01,
        "Historical counts reconstructed from saved scalar metrics; validation maximum-F1 rule.\nThe original run did not save its numerical threshold.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save_figure(
        fig,
        "12_validation_confusion_matrix.png",
        ["12"],
        "Reconstructed retained-reference validation counts, not final-test evidence.",
    )

    thirteen = read_result("13")
    rows = thirteen["candidate_results"]
    vote_labels = ["LightGBM", "LightGBM + CatBoost", "LightGBM + CatBoost + RF"]
    comparison_figure("13", rows, "configuration", "validation_average_precision", "lgbm_reference", vote_labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, row, label in zip(axes, rows, vote_labels):
        matrix = np.array(
            [
                [row["validation_true_negatives_at_max_f1"], row["validation_false_positives_at_max_f1"]],
                [row["validation_false_negatives_at_max_f1"], row["validation_true_positives_at_max_f1"]],
            ]
        )
        draw_confusion(ax, matrix, f"{label}\nThreshold {row['validation_threshold_at_max_f1']:.6f}")
    fig.suptitle("Stage 13: directly recorded validation confusion matrices", fontsize=16)
    fig.text(
        0.02,
        0.01,
        "Colour = percentage of each actual class. Each candidate uses its own maximum-F1 threshold; alert volumes differ.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(
        fig,
        "13_validation_confusion_matrices.png",
        ["13"],
        "Recorded validation counts and rates for reference and soft votes.",
    )

    threshold_path = PROJECT_DIR / "mlartifacts" / rows[0]["run_id"] / "artifacts" / "validation_thresholds.csv"
    thresholds = pd.read_csv(threshold_path).sort_values("threshold")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for metric, label in [("alert_rate", "Transactions flagged"), ("recall", "Fraud recall")]:
        axes[0].plot(thresholds["threshold"], thresholds[metric] * 100, "o--", label=label)
    axes[1].plot(thresholds["threshold"], thresholds["precision"] * 100, "o--", color=GREEN, label="Precision")
    for ax in axes:
        ax.axvline(
            rows[0]["validation_threshold_at_max_f1"],
            color="#333333",
            linestyle=":",
            label="Validation max-F1 threshold",
        )
        ax.set_xlabel("Score threshold")
        ax.set_ylim(0, 100 if ax is axes[0] else 6.5)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("Stage 13: LightGBM threshold diagnostics (validation)", fontsize=15)
    fig.text(
        0.02,
        0.01,
        "Markers = recorded diagnostic thresholds; dashed segments only connect them. Not a full precision-recall curve. Zero-alert precision is reported as 0 by the metric function.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(
        fig,
        "13_threshold_tradeoff.png",
        ["13"],
        "Saved threshold grid: recall, alert volume and precision.",
        [threshold_path],
    )

    fourteen = read_result("14")
    if fourteen["canonical_test_evaluated"] is not True:
        raise ValueError("The canonical completed test result is required.")
    metrics = fourteen["test_metrics"]
    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )
    if matrix.sum() != metrics["rows"] or matrix[1].sum() != metrics["fraud_rows"]:
        raise ValueError("Final confusion counts do not match the saved population.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    heat = draw_confusion(axes[0], matrix, "Actual transaction counts", False)
    fig.colorbar(heat, ax=axes[0], label="Transactions")
    heat = draw_confusion(axes[1], matrix, "Percentage within each actual class")
    fig.colorbar(heat, ax=axes[1], label="Percentage of actual class")
    fig.suptitle(
        f"Stage 14: final LightGBM test confusion matrix\nFrozen threshold {metrics['threshold']:.6f} | {metrics['rows']:,} test transactions",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(
        fig,
        "14_test_confusion_matrix.png",
        ["14"],
        "Directly recorded final confusion counts and row-normalised percentages.",
    )

    workloads = fourteen["test_workloads"]
    labels = [f"Top {row['requested_alert_rate'] * 100:g}% batch" for row in workloads] + ["Frozen score threshold"]
    series = [
        ([row["actual_alert_rate"] for row in workloads] + [metrics["alert_rate"]], "Transactions flagged (%)"),
        ([row["precision"] for row in workloads] + [metrics["precision"]], "Precision (%)"),
        ([row["fraud_count_recall"] for row in workloads] + [metrics["recall"]], "Fraud-count recall (%)"),
        ([row["fraud_value_recall"] for row in workloads] + [metrics["fraud_value_recall"]], "Fraud-value recall (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (values, title) in zip(axes.flat, series):
        bar_panel(ax, labels, values, title, percentage=True)
        ax.set_xlabel("Percentage (%)")
        if "Precision" not in title:
            ax.set_xlim(0, 115)
    fig.suptitle("Stage 14: test review workload versus fraud capture", fontsize=16)
    fig.text(
        0.02,
        0.01,
        "Batch top-k rules and the frozen score threshold are distinct predeclared scenarios, not interchangeable API rules. No test rule was selected from this figure.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), h_pad=2)
    save_figure(
        fig,
        "14_test_review_tradeoff.png",
        ["14"],
        "Saved final-test scenarios: alert share, precision and count/value recall.",
    )

    hard = read_result("13_hard")
    candidates = hard["candidate_results"]
    names = ["LightGBM native", "Hard LightGBM + CatBoost", "Hard LightGBM + CatBoost + RF"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, row, label in zip(axes, candidates, names):
        metrics = row["classification"]
        matrix = np.array(
            [
                [metrics["true_negatives"], metrics["false_positives"]],
                [metrics["false_negatives"], metrics["true_positives"]],
            ]
        )
        if matrix.sum() != metrics["rows"] or matrix[1].sum() != metrics["fraud_rows"]:
            raise ValueError("Hard-voting confusion counts do not match the saved population.")
        draw_confusion(ax, matrix, label)
    fig.suptitle("Stage 13 follow-up: native hard-voting validation decisions", fontsize=16)
    fig.text(
        0.02,
        0.01,
        "Recorded full-data validation counts. Colour = percentage of actual class. "
        "Two-member ties go to legitimate; three-member votes require two fraud decisions. No test evaluation.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save_figure(
        fig,
        "13_hard_validation_confusion_matrices.png",
        ["13_hard"],
        "Directly recorded native hard-voting validation confusion matrices.",
    )

    policies = [policy for row in candidates for policy in row["policy_results"]]
    labels = [
        f"{name}: {rule}"
        for name in ("LightGBM", "LGBM + CatBoost", "LGBM + CatBoost + RF")
        for rule in ("native/hard", "soft argmax 0.5", "soft validation max-F1")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, key, title, percentage in zip(
        axes,
        ("f1", "recall", "alert_rate"),
        ("Validation F1", "Fraud-count recall (%)", "Transactions flagged (%)"),
        (False, True, True),
    ):
        bar_panel(ax, labels, [policy[key] for policy in policies], title, percentage=percentage)
        ax.set_xlabel("Percentage (%)" if percentage else "F1")
    fig.suptitle("Stage 13 follow-up: paired classification policies from the same fitted members", fontsize=16)
    fig.text(
        0.02,
        0.01,
        "Soft max-F1 is selected on validation; alert shares differ. These are development diagnostics, "
        "not an equal-workload or independent test comparison. No candidate was promoted.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2)
    save_figure(
        fig,
        "13_hard_soft_policy_comparison.png",
        ["13_hard"],
        "Paired native/hard, soft argmax and validation-selected soft rules, with alert burden.",
    )

    manifest = {
        "purpose": "Documentation-only figures from recorded artifacts; no modelling or scoring.",
        "figure_count": len(FIGURES),
        "figures": FIGURES,
    }
    (OUTPUT_DIR / "figure_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(FIGURES)} evidence figures and their source manifest to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
