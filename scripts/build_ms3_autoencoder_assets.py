#!/usr/bin/env python3
"""Build clean presentation assets for the MS3 autoencoder section."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUT_ROOT = PROJECT_ROOT / "deliverables" / "ms3_autoencoder" / "assets"

COLORS = {
    "bg": "#FFFFFF",
    "grid": "#E7ECF2",
    "text": "#122033",
    "muted": "#5B6575",
    "pca": "#8A93A5",
    "dae": "#4C78A8",
    "sup": "#159A9C",
    "accent": "#F28E2B",
    "good": "#1B9E77",
    "bad": "#D55E00",
    "line_train": "#4C78A8",
    "line_val": "#159A9C",
}

DISPLAY_LABELS = {
    "PCA-50": "PCA-50",
    "DAE-32-LR-noise0.10": "DAE-32 + LR",
    "SupDAE-32-head-noise0.10": "SupDAE w=1.0",
    "SupDAE-32-head-w0.5-noise0.10": "SupDAE w=0.5",
    "SupDAE-32-head-w2.0-noise0.10": "SupDAE w=2.0",
}

DISPLAY_ORDER = [
    "PCA-50",
    "DAE-32-LR-noise0.10",
    "SupDAE-32-head-noise0.10",
    "SupDAE-32-head-w0.5-noise0.10",
    "SupDAE-32-head-w2.0-noise0.10",
]

BEST_RUN = "SupDAE-32-head-w0.5-noise0.10"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": COLORS["bg"],
        }
    )


def ensure_out_dir() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for legacy_name in [
        "07_unsupervised_vs_supervised.png",
        "07_unsupervised_vs_supervised.svg",
    ]:
        legacy_path = OUT_ROOT / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


def read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["split"] == "test"].copy()


def build_variant_summary() -> pd.DataFrame:
    parts = [
        read_metrics(RESULTS_ROOT / "gse96583_dae_lr_noise010_e10_metrics.csv"),
        read_metrics(RESULTS_ROOT / "gse96583_supdae_head_noise010_e10_metrics.csv"),
        read_metrics(RESULTS_ROOT / "gse96583_supdae_head_w05_noise010_e10_metrics.csv"),
        read_metrics(RESULTS_ROOT / "gse96583_supdae_head_w20_noise010_e10_metrics.csv"),
    ]
    summary = pd.concat(parts, ignore_index=True)
    summary = summary.drop_duplicates(subset=["representation"], keep="last")
    summary["display_label"] = summary["representation"].map(DISPLAY_LABELS)
    order_map = {name: idx for idx, name in enumerate(DISPLAY_ORDER)}
    summary["display_order"] = summary["representation"].map(order_map)
    summary = summary.sort_values("display_order").reset_index(drop=True)
    pca_row = summary.loc[summary["representation"] == "PCA-50"].iloc[0]
    summary["delta_macro_f1_vs_pca"] = summary["macro_f1"] - pca_row["macro_f1"]
    summary["delta_accuracy_vs_pca"] = summary["accuracy"] - pca_row["accuracy"]
    summary["delta_bal_acc_vs_pca"] = summary["balanced_accuracy"] - pca_row["balanced_accuracy"]
    return summary[
        [
            "representation",
            "display_label",
            "classifier",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "delta_accuracy_vs_pca",
            "delta_bal_acc_vs_pca",
            "delta_macro_f1_vs_pca",
        ]
    ]


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_ROOT / name, index=False)


def color_for_representation(rep: str) -> str:
    if rep == "PCA-50":
        return COLORS["pca"]
    if rep == "DAE-32-LR-noise0.10":
        return COLORS["dae"]
    return COLORS["sup"]


def draw_table_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    columns: list[str],
    out_stem: str,
    highlight_row: int | None = None,
) -> None:
    n_rows = len(df)
    n_cols = len(columns)
    fig_h = 1.2 + 0.75 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_axis_off()

    ax.text(0.0, 1.08, title, fontsize=18, fontweight="bold", ha="left", va="bottom", transform=ax.transAxes)
    ax.text(0.0, 1.02, subtitle, fontsize=11, color=COLORS["muted"], ha="left", va="bottom", transform=ax.transAxes)

    cell_text = df[columns].values.tolist()
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="upper left",
        bbox=[0, 0.02, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        if row == 0:
            cell.set_facecolor("#F5F8FB")
            cell.set_text_props(weight="bold", color=COLORS["text"])
            cell.set_height(0.12)
        else:
            rep = df.iloc[row - 1]["representation"]
            base_color = "#FFFFFF"
            if highlight_row is not None and row - 1 == highlight_row:
                base_color = "#E9F8F5"
            elif rep == "PCA-50":
                base_color = "#F8F9FB"
            cell.set_facecolor(base_color)
            cell.set_height(0.11)
            if col == 0:
                cell.get_text().set_color(color_for_representation(rep))
                cell.get_text().set_fontweight("bold")

    fig.savefig(OUT_ROOT / f"{out_stem}.png", dpi=220)
    fig.savefig(OUT_ROOT / f"{out_stem}.svg")
    plt.close(fig)


def format_signed(x: float) -> str:
    return f"{x:+.3f}"


def build_best_run_table(variant_summary: pd.DataFrame) -> pd.DataFrame:
    subset = variant_summary[variant_summary["representation"].isin(["PCA-50", BEST_RUN])].copy()
    subset = subset.set_index("representation")
    pca = subset.loc["PCA-50"]
    best = subset.loc[BEST_RUN]
    return pd.DataFrame(
        [
            ["Accuracy", f"{pca['accuracy']:.3f}", f"{best['accuracy']:.3f}", format_signed(best["accuracy"] - pca["accuracy"])],
            [
                "Macro-F1",
                f"{pca['macro_f1']:.3f}",
                f"{best['macro_f1']:.3f}",
                format_signed(best["macro_f1"] - pca["macro_f1"]),
            ],
            [
                "Balanced Acc.",
                f"{pca['balanced_accuracy']:.3f}",
                f"{best['balanced_accuracy']:.3f}",
                format_signed(best["balanced_accuracy"] - pca["balanced_accuracy"]),
            ],
            [
                "Weighted-F1",
                f"{pca['weighted_f1']:.3f}",
                f"{best['weighted_f1']:.3f}",
                format_signed(best["weighted_f1"] - pca["weighted_f1"]),
            ],
        ],
        columns=["Metric", "PCA-50", "SupDAE w=0.5", "Delta"],
    )


def draw_best_run_metric_plot(best_run_table: pd.DataFrame, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    rows = best_run_table.copy()
    metrics = rows["Metric"].tolist()
    y = np.arange(len(metrics))
    pca_vals = rows["PCA-50"].astype(float).to_numpy()
    best_vals = rows["SupDAE w=0.5"].astype(float).to_numpy()

    ax.barh(y + 0.17, pca_vals, height=0.3, color=COLORS["pca"], label="PCA-50")
    ax.barh(y - 0.17, best_vals, height=0.3, color=COLORS["sup"], label="SupDAE w=0.5")

    for idx, (pca_val, best_val) in enumerate(zip(pca_vals, best_vals)):
        ax.text(pca_val + 0.008, y[idx] + 0.17, f"{pca_val:.3f}", va="center", ha="left", fontsize=10)
        ax.text(best_val + 0.008, y[idx] - 0.17, f"{best_val:.3f}", va="center", ha="left", fontsize=10)

    ax.set_yticks(y, metrics)
    ax.invert_yaxis()
    ax.set_xlim(0.78, 0.985)
    ax.set_xlabel("Test score")
    ax.set_title("Best Run vs PCA", pad=14, fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    fig.subplots_adjust(bottom=0.20)

    fig.savefig(OUT_ROOT / f"{out_stem}.png", dpi=220)
    fig.savefig(OUT_ROOT / f"{out_stem}.svg")
    plt.close(fig)


def build_per_class_delta() -> pd.DataFrame:
    per_class = pd.read_csv(RESULTS_ROOT / "gse96583_supdae_head_w05_noise010_e10_per_class.csv")
    test = per_class[per_class["split"] == "test"].copy()
    pivot = test.pivot(index="cell_type", columns="representation", values="f1_score")
    pivot = pivot[["PCA-50", BEST_RUN]].reset_index()
    pivot["delta_f1"] = pivot[BEST_RUN] - pivot["PCA-50"]
    pivot = pivot.sort_values("delta_f1", ascending=True).reset_index(drop=True)
    return pivot


def draw_per_class_delta(delta_df: pd.DataFrame, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = [COLORS["good"] if x >= 0 else COLORS["bad"] for x in delta_df["delta_f1"]]
    ax.barh(delta_df["cell_type"], delta_df["delta_f1"], color=colors, edgecolor="none")
    ax.axvline(0, color=COLORS["muted"], linewidth=1)
    ax.set_title("Per-Class F1 Change: SupDAE w=0.5 minus PCA", pad=12, fontweight="bold")
    ax.set_xlabel("Delta F1 on test split")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)

    for idx, value in enumerate(delta_df["delta_f1"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.004 if value >= 0 else -0.004
        ax.text(value + offset, idx, f"{value:+.3f}", va="center", ha=ha, fontsize=10, color=COLORS["text"])

    max_delta = float(delta_df["delta_f1"].max())
    min_delta = float(delta_df["delta_f1"].min())
    left_pad = 0.01 if min_delta >= 0 else 0.02
    right_pad = 0.02
    ax.set_xlim(min(0.0, min_delta - left_pad), max_delta + right_pad)

    fig.savefig(OUT_ROOT / f"{out_stem}.png", dpi=220)
    fig.savefig(OUT_ROOT / f"{out_stem}.svg")
    plt.close(fig)


def draw_training_curve(out_stem: str) -> None:
    history = pd.read_csv(RESULTS_ROOT / "gse96583_supdae_head_w05_noise010_e10_training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", color=COLORS["line_train"], label="Train total loss")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", color=COLORS["line_val"], label="Val total loss")
    axes[0].set_title("Best Run Training Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(color=COLORS["grid"], linewidth=0.8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2)

    axes[1].plot(history["epoch"], history["val_recon_loss"], marker="o", color=COLORS["dae"], label="Val recon")
    axes[1].plot(history["epoch"], history["val_class_loss"], marker="o", color=COLORS["accent"], label="Val class")
    axes[1].set_title("Validation Loss Components", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(color=COLORS["grid"], linewidth=0.8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2)

    fig.subplots_adjust(top=0.88, wspace=0.22)

    fig.savefig(OUT_ROOT / f"{out_stem}.png", dpi=220)
    fig.savefig(OUT_ROOT / f"{out_stem}.svg")
    plt.close(fig)


def add_box(ax, x: float, y: float, w: float, h: float, label: str, facecolor: str, fontsize: int = 12) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=facecolor,
        facecolor=facecolor,
        alpha=0.14,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, fontweight="bold")


def add_light_box(ax, x: float, y: float, w: float, h: float, label: str, edgecolor: str, fontsize: int = 11) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor="#FFFFFF",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=COLORS["text"])


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = COLORS["muted"]) -> None:
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color)
    ax.add_patch(arrow)


def draw_pipeline_diagram(out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.95, "Supervised DAE Pipeline", fontsize=18, fontweight="bold", ha="left")
    ax.text(
        0.02,
        0.89,
        "Single-cell input is denoised, encoded into latent z, then used for reconstruction and classification.",
        fontsize=11,
        color=COLORS["muted"],
        ha="left",
    )

    add_box(ax, 0.03, 0.40, 0.12, 0.18, "Gene\nexpression", "#C9D7E8")
    add_box(ax, 0.20, 0.40, 0.12, 0.18, "Dropout\ncorruption", "#D6EDEB")
    add_box(ax, 0.38, 0.40, 0.13, 0.18, "Encoder", "#B7D1F0")
    add_box(ax, 0.57, 0.40, 0.10, 0.18, "Latent z", "#A7E1DB")
    add_box(ax, 0.75, 0.60, 0.12, 0.14, "Classifier\nhead", "#BEE8D8")
    add_box(ax, 0.75, 0.24, 0.12, 0.14, "Decoder", "#B7D1F0")
    add_light_box(ax, 0.90, 0.60, 0.08, 0.14, "Cell type\nprediction", COLORS["sup"])
    add_light_box(ax, 0.90, 0.24, 0.08, 0.14, "Input\nreconstruction", COLORS["dae"])

    add_arrow(ax, 0.15, 0.49, 0.20, 0.49)
    add_arrow(ax, 0.32, 0.49, 0.38, 0.49)
    add_arrow(ax, 0.51, 0.49, 0.57, 0.49)
    add_arrow(ax, 0.67, 0.53, 0.75, 0.67, color=COLORS["sup"])
    add_arrow(ax, 0.67, 0.45, 0.75, 0.31, color=COLORS["dae"])
    add_arrow(ax, 0.87, 0.67, 0.90, 0.67, color=COLORS["sup"])
    add_arrow(ax, 0.87, 0.31, 0.90, 0.31, color=COLORS["dae"])

    ax.text(0.04, 0.13, "Training objective:", fontsize=13, fontweight="bold")
    ax.text(0.25, 0.13, "reconstruction loss + lambda * classification loss", fontsize=13)
    ax.text(0.25, 0.07, "Best setting: lambda = 0.5", fontsize=12, color=COLORS["sup"], fontweight="bold")

    fig.savefig(OUT_ROOT / f"{out_stem}.png", dpi=220)
    fig.savefig(OUT_ROOT / f"{out_stem}.svg")
    plt.close(fig)


def draw_unsupervised_pipeline_diagram(out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.95, "Unsupervised DAE Pipeline", fontsize=18, fontweight="bold", ha="left")
    ax.text(
        0.02,
        0.89,
        "The autoencoder is trained without labels, then latent z is used by a downstream logistic-regression classifier.",
        fontsize=11,
        color=COLORS["muted"],
        ha="left",
    )

    add_box(ax, 0.03, 0.42, 0.12, 0.18, "Gene\nexpression", "#C9D7E8")
    add_box(ax, 0.20, 0.42, 0.12, 0.18, "Dropout\ncorruption", "#D6EDEB")
    add_box(ax, 0.38, 0.42, 0.13, 0.18, "Encoder", "#B7D1F0")
    add_box(ax, 0.57, 0.42, 0.10, 0.18, "Latent z", "#A7E1DB")
    add_box(ax, 0.75, 0.24, 0.12, 0.14, "Decoder", "#B7D1F0")
    add_light_box(ax, 0.90, 0.24, 0.08, 0.14, "Input\nreconstruction", COLORS["dae"])

    add_box(ax, 0.75, 0.61, 0.12, 0.14, "Logistic\nregression", "#C9D7E8")
    add_light_box(ax, 0.90, 0.61, 0.08, 0.14, "Cell type\nprediction", COLORS["pca"])

    add_arrow(ax, 0.15, 0.51, 0.20, 0.51)
    add_arrow(ax, 0.32, 0.51, 0.38, 0.51)
    add_arrow(ax, 0.51, 0.51, 0.57, 0.51)
    add_arrow(ax, 0.67, 0.47, 0.75, 0.31, color=COLORS["dae"])
    add_arrow(ax, 0.67, 0.55, 0.75, 0.68, color=COLORS["pca"])
    add_arrow(ax, 0.87, 0.31, 0.90, 0.31, color=COLORS["dae"])
    add_arrow(ax, 0.87, 0.68, 0.90, 0.68, color=COLORS["pca"])

    ax.text(0.72, 0.80, "Downstream evaluation after AE pretraining", fontsize=10.5, color=COLORS["muted"], ha="left")
    ax.plot([0.71, 0.985], [0.78, 0.78], color=COLORS["grid"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.plot([0.71, 0.71], [0.18, 0.78], color=COLORS["grid"], linewidth=1.2, linestyle=(0, (4, 3)))

    ax.text(0.04, 0.13, "Training objective:", fontsize=13, fontweight="bold")
    ax.text(0.25, 0.13, "reconstruction loss only", fontsize=13)
    ax.text(0.25, 0.07, "Classifier is fitted afterward on latent z", fontsize=12, color=COLORS["pca"], fontweight="bold")

    fig.savefig(OUT_ROOT / f"{out_stem}.png", dpi=220)
    fig.savefig(OUT_ROOT / f"{out_stem}.svg")
    plt.close(fig)


def draw_learning_modes_diagram(out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.95, "Unsupervised vs Supervised DAE", fontsize=18, fontweight="bold", ha="left")
    ax.text(
        0.02,
        0.89,
        "Two training modes were tested on the same fixed split before selecting the best variant.",
        fontsize=11,
        color=COLORS["muted"],
        ha="left",
    )

    add_box(ax, 0.05, 0.55, 0.24, 0.22, "Unsupervised DAE\nReconstruct noisy input\nthen fit logistic regression on latent z", "#D9E5F2")
    add_box(ax, 0.38, 0.55, 0.24, 0.22, "Supervised DAE\nReconstruct input and\npredict cell type jointly", "#D5F0EA")
    add_box(ax, 0.71, 0.55, 0.20, 0.22, "Best result\nSupDAE w=0.5\nMacro-F1 = 0.898", "#E7F5EE")

    add_box(ax, 0.08, 0.18, 0.18, 0.16, "Train on\nreconstruction only", "#EEF3F8", fontsize=11)
    add_box(ax, 0.41, 0.18, 0.18, 0.16, "Train on\nreconstruction + labels", "#EEF9F6", fontsize=11)
    add_box(ax, 0.71, 0.18, 0.20, 0.16, "Improved\naccuracy and macro-F1\nvs PCA", "#F7FBF9", fontsize=11)

    add_arrow(ax, 0.17, 0.55, 0.17, 0.34, color=COLORS["dae"])
    add_arrow(ax, 0.50, 0.55, 0.50, 0.34, color=COLORS["sup"])
    add_arrow(ax, 0.62, 0.66, 0.71, 0.66, color=COLORS["accent"])

    fig.savefig(OUT_ROOT / f"{out_stem}.png", dpi=220)
    fig.savefig(OUT_ROOT / f"{out_stem}.svg")
    plt.close(fig)


def build_readme() -> None:
    lines = [
        "# MS3 Autoencoder Assets",
        "",
        "This folder contains clean reusable assets for the autoencoder PPT section.",
        "",
        "## Files",
        "",
        "- `01_variant_summary_table.png/.svg/.csv`: test-set comparison across PCA, unsupervised DAE, and supervised DAE variants",
        "- `02_best_run_vs_pca_table.png/.svg/.csv`: compact best-run vs PCA table",
        "- `03_best_run_metric_comparison.png/.svg`: grouped metric comparison for the best run vs PCA",
        "- `04_best_run_per_class_delta.png/.svg/.csv`: per-class F1 change for the best supervised run minus PCA",
        "- `05_best_run_training_curve.png/.svg`: training curve and validation loss breakdown",
        "- `06_supervised_dae_pipeline.png/.svg`: clean method diagram",
        "- `07_unsupervised_dae_pipeline.png/.svg`: unsupervised method diagram with downstream classifier branch",
        "- `08_unsupervised_vs_supervised.png/.svg`: comparison diagram for backup / explanation slides",
        "",
        "## Source results",
        "",
        "- `results/gse96583_dae_lr_noise010_e10_metrics.csv`",
        "- `results/gse96583_supdae_head_noise010_e10_metrics.csv`",
        "- `results/gse96583_supdae_head_w05_noise010_e10_metrics.csv`",
        "- `results/gse96583_supdae_head_w20_noise010_e10_metrics.csv`",
        "- `results/gse96583_supdae_head_w05_noise010_e10_per_class.csv`",
        "- `results/gse96583_supdae_head_w05_noise010_e10_training_history.csv`",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    configure_style()
    ensure_out_dir()

    variant_summary = build_variant_summary()
    save_csv(variant_summary, "01_variant_summary_table.csv")
    display_variant = variant_summary.copy()
    display_variant["Variant"] = display_variant["display_label"]
    display_variant["Head"] = display_variant["classifier"].replace(
        {
            "Logistic Regression": "LogReg",
            "Supervised Head": "Sup. head",
        }
    )
    display_variant["Acc."] = display_variant["accuracy"].map(lambda x: f"{x:.3f}")
    display_variant["Bal. Acc."] = display_variant["balanced_accuracy"].map(lambda x: f"{x:.3f}")
    display_variant["Macro-F1"] = display_variant["macro_f1"].map(lambda x: f"{x:.3f}")
    display_variant["Weighted-F1"] = display_variant["weighted_f1"].map(lambda x: f"{x:.3f}")
    display_variant["dMacro-F1"] = display_variant["delta_macro_f1_vs_pca"].map(format_signed)
    draw_table_figure(
        df=display_variant,
        title="Variant Summary",
        subtitle="Test-set comparison on the notebook fixed split",
        columns=["Variant", "Head", "Acc.", "Bal. Acc.", "Macro-F1", "Weighted-F1", "dMacro-F1"],
        out_stem="01_variant_summary_table",
        highlight_row=int(np.where(variant_summary["representation"].to_numpy() == BEST_RUN)[0][0]),
    )

    best_run_table = build_best_run_table(variant_summary)
    save_csv(best_run_table, "02_best_run_vs_pca_table.csv")
    draw_table_figure(
        df=best_run_table.assign(representation=["PCA-50", "PCA-50", "PCA-50", "PCA-50"]),
        title="Best Run vs PCA",
        subtitle="SupDAE w=0.5 is the strongest current setting on the fixed split",
        columns=["Metric", "PCA-50", "SupDAE w=0.5", "Delta"],
        out_stem="02_best_run_vs_pca_table",
    )
    draw_best_run_metric_plot(best_run_table, "03_best_run_metric_comparison")

    per_class_delta = build_per_class_delta()
    save_csv(per_class_delta, "04_best_run_per_class_delta.csv")
    draw_per_class_delta(per_class_delta, "04_best_run_per_class_delta")

    draw_training_curve("05_best_run_training_curve")
    draw_pipeline_diagram("06_supervised_dae_pipeline")
    draw_unsupervised_pipeline_diagram("07_unsupervised_dae_pipeline")
    draw_learning_modes_diagram("08_unsupervised_vs_supervised")
    build_readme()


if __name__ == "__main__":
    main()
