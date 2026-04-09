#!/usr/bin/env python3
"""Build milestone 2 notebook, figures, and slide materials."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from prepare_gse96583_ms2 import prepare_gse96583_dataset, raw_file_inventory, sample_manifest


ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "deliverables" / "ms2_ab"
FIG = DELIV / "figures"

BG = "#F7F4EE"
NAVY = "#133C55"
TEAL = "#2E7D6F"
GOLD = "#C68B31"
CORAL = "#D05D4E"
INK = "#1E2933"
MUTED = "#5E6C76"
GRID = "#D8D2C8"
LEGEND_FONT = 11.5
LEGEND_TITLE = 12.5
LEGEND_MARKER = 8
CELL_TYPE_PLOT_CAP = 2500

CELL_TYPE_PALETTE = {
    "CD4 T cells": "#1b9e77",
    "CD14+ Monocytes": "#d95f02",
    "B cells": "#7570b3",
    "CD8 T cells": "#e7298a",
    "NK cells": "#66a61e",
    "FCGR3A+ Monocytes": "#e6ab02",
    "Dendritic cells": "#a6761d",
    "Megakaryocytes": "#666666",
}

BATCH_PALETTE = {
    "batch1": NAVY,
    "batch2": CORAL,
}

CONDITION_PALETTE = {
    "batch1": NAVY,
    "ctrl": TEAL,
    "stim": GOLD,
}

STAGE_ORDER = ["raw", "qc_filtered", "metadata_singlets", "strict_qc_final"]


def ensure_dirs() -> None:
    DELIV.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def build_tables(
    adata_b1,
    adata_b2,
    combined,
    stage_counts: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    files_df = raw_file_inventory()
    manifest_df = sample_manifest()

    sample_counts = (
        combined.obs.groupby(["batch_label", "sample_accession", "sample_label", "condition"], observed=True)
        .size()
        .rename("cells")
        .reset_index()
        .sort_values(["batch_label", "sample_label"])
    )

    batch_counts = (
        combined.obs["batch_label"].value_counts().rename_axis("batch_label").reset_index(name="cells")
    )

    condition_counts = (
        combined.obs["condition"].value_counts().rename_axis("condition").reset_index(name="cells")
    )

    cell_type_counts = (
        combined.obs["cell_type"].value_counts().rename_axis("cell_type").reset_index(name="cells")
    )
    cell_type_counts["percent"] = (cell_type_counts["cells"] / cell_type_counts["cells"].sum()) * 100
    cell_type_counts["is_rare_under_2pct"] = cell_type_counts["percent"] < 2.0

    condition_composition = pd.crosstab(
        combined.obs["cell_type"], combined.obs["condition"], normalize="columns"
    ).mul(100)
    condition_composition = condition_composition.loc[cell_type_counts["cell_type"]]

    qc_long = combined.obs[
        ["batch_label", "condition", "n_genes_by_counts", "total_counts", "pct_counts_mt", "sparsity"]
    ].copy()

    return {
        "raw_file_inventory": files_df,
        "sample_manifest": manifest_df,
        "stage_counts": stage_counts,
        "batch_effect_metrics": metrics_df,
        "sample_counts": sample_counts,
        "batch_counts": batch_counts,
        "condition_counts": condition_counts,
        "cell_type_counts": cell_type_counts,
        "condition_composition": condition_composition,
        "qc_long": qc_long,
    }


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    tables["raw_file_inventory"].to_csv(DELIV / "raw_file_inventory.csv", index=False)
    tables["sample_manifest"].to_csv(DELIV / "sample_manifest.csv", index=False)
    tables["stage_counts"].to_csv(DELIV / "wrangling_stage_counts.csv", index=False)
    tables["batch_effect_metrics"].to_csv(DELIV / "batch_effect_metrics.csv", index=False)
    tables["sample_counts"].to_csv(DELIV / "sample_counts.csv", index=False)
    tables["batch_counts"].to_csv(DELIV / "batch_counts.csv", index=False)
    tables["condition_counts"].to_csv(DELIV / "condition_counts.csv", index=False)
    tables["cell_type_counts"].to_csv(DELIV / "cell_type_counts.csv", index=False)
    tables["condition_composition"].to_csv(DELIV / "condition_composition.csv")


def save_raw_file_sizes(files_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    df = files_df.sort_values("size_mb", ascending=True)
    ax.barh(df["file"], df["size_mb"], color="#5C88C4")
    ax.set_title("GSE96583 Supplementary Files")
    ax.set_xlabel("Size (MB)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    fig.savefig(FIG / "raw_file_sizes.png", dpi=220)
    plt.close(fig)


def save_wrangling_summary(stage_counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    stage_counts = stage_counts.copy()
    stage_counts["stage"] = pd.Categorical(stage_counts["stage"], categories=STAGE_ORDER, ordered=True)

    sns.barplot(
        data=stage_counts,
        x="stage",
        y="cells",
        hue="batch_label",
        palette=BATCH_PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("Cells Kept at Each Wrangling Stage")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Cells")
    axes[0].tick_params(axis="x", rotation=10)

    sns.barplot(
        data=stage_counts,
        x="stage",
        y="genes",
        hue="batch_label",
        palette=BATCH_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Genes Retained After QC")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Genes")
    axes[1].tick_params(axis="x", rotation=10)

    for ax in axes:
        ax.grid(axis="y", color=GRID)
        if ax is axes[1]:
            ax.legend_.remove()
        else:
            ax.legend(title="Batch", fontsize=LEGEND_FONT, title_fontsize=LEGEND_TITLE)

    fig.suptitle("Wrangling Summary: QC, Metadata Singlets, and Strict Doublet Filtering", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "wrangling_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution(cell_type_counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    df = cell_type_counts.sort_values("cells", ascending=True)
    colors = [CELL_TYPE_PALETTE.get(cell_type, "#7A7A7A") for cell_type in df["cell_type"]]
    ax.barh(df["cell_type"], df["cells"], color=colors)
    ax.set_title("Class Distribution After Strict QC")
    ax.set_xlabel("Cells")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID)

    xmax = df["cells"].max()
    for _, row in df.iterrows():
        ax.text(
            row["cells"] + xmax * 0.01,
            row["cell_type"],
            f"{row['percent']:.1f}%",
            va="center",
            fontsize=9.5,
            color=INK,
        )

    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=220)
    plt.close(fig)


def save_batch_condition_distribution(
    sample_counts: pd.DataFrame,
    condition_counts: pd.DataFrame,
    condition_composition: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(14.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.55], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    sample_order = ["A", "B", "C", "ctrl", "stim"]
    sample_counts = sample_counts.copy()
    sample_counts["sample_label"] = pd.Categorical(sample_counts["sample_label"], categories=sample_order, ordered=True)
    sample_counts = sample_counts.sort_values("sample_label")
    sns.barplot(
        data=sample_counts,
        x="sample_label",
        y="cells",
        hue="batch_label",
        palette=BATCH_PALETTE,
        ax=ax1,
    )
    ax1.set_title("Cells Per Source Matrix")
    ax1.set_xlabel("Sample label")
    ax1.set_ylabel("Cells")
    ax1.legend(title="Batch", fontsize=LEGEND_FONT, title_fontsize=LEGEND_TITLE)
    ax1.grid(axis="y", color=GRID)

    ax2 = fig.add_subplot(gs[0, 1])
    condition_order = ["batch1", "ctrl", "stim"]
    condition_counts = condition_counts.copy()
    condition_counts["condition"] = pd.Categorical(
        condition_counts["condition"], categories=condition_order, ordered=True
    )
    condition_counts = condition_counts.sort_values("condition")
    sns.barplot(
        data=condition_counts,
        x="condition",
        y="cells",
        hue="condition",
        palette=CONDITION_PALETTE,
        dodge=False,
        legend=False,
        ax=ax2,
    )
    ax2.set_title("Batch / Condition Totals")
    ax2.set_xlabel("")
    ax2.set_ylabel("Cells")
    ax2.grid(axis="y", color=GRID)

    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(
        condition_composition.loc[:, condition_order],
        cmap="YlGnBu",
        linewidths=0.4,
        cbar_kws={"label": "Percent of condition"},
        ax=ax3,
    )
    ax3.set_title("Cell-Type Composition by Condition")
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    fig.tight_layout()
    fig.savefig(FIG / "batch_condition_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_qc_metrics(qc_long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    metrics = [
        ("n_genes_by_counts", "Detected genes per cell"),
        ("total_counts", "Total counts per cell"),
        ("sparsity", "Per-cell sparsity"),
    ]
    for ax, (column, title) in zip(axes, metrics):
        sns.violinplot(
            data=qc_long,
            x="batch_label",
            y=column,
            hue="batch_label",
            palette=BATCH_PALETTE,
            cut=0,
            inner="quartile",
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", color=GRID)

    fig.suptitle("Quality / Missingness Proxies Differ Across Batches", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "qc_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_batch_effect_metrics(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df.copy()
    metric_order = [
        "silhouette_batch",
        "silhouette_condition",
        "silhouette_cell_type",
        "neighbor_same_batch",
        "neighbor_same_condition",
        "neighbor_same_cell_type",
    ]
    label_map = {
        "silhouette_batch": "Silhouette: batch",
        "silhouette_condition": "Silhouette: condition",
        "silhouette_cell_type": "Silhouette: cell type",
        "neighbor_same_batch": "15-NN same batch",
        "neighbor_same_condition": "15-NN same condition",
        "neighbor_same_cell_type": "15-NN same cell type",
    }
    plot_df["metric"] = pd.Categorical(plot_df["metric"], categories=metric_order, ordered=True)
    plot_df["embedding"] = plot_df["embedding"].map(
        {"before_harmony": "Before Harmony", "after_harmony": "After Harmony"}
    )
    plot_df = plot_df.sort_values("metric")

    fig, ax = plt.subplots(figsize=(11.8, 5.3))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="embedding",
        palette={"Before Harmony": CORAL, "After Harmony": TEAL},
        ax=ax,
    )
    ax.set_title("Batch Effect Metrics Before and After Harmony")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels([label_map[m] for m in metric_order], rotation=18, ha="right")
    ax.grid(axis="y", color=GRID)
    ax.legend(title="", fontsize=LEGEND_FONT, title_fontsize=LEGEND_TITLE)
    fig.tight_layout()
    fig.savefig(FIG / "batch_effect_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def legend_handles(order, palette, labels, alpha: float = 0.5) -> list[Line2D]:
    label_values = set(pd.Series(labels).astype(str).tolist())
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=LEGEND_MARKER,
            markerfacecolor=palette.get(value, "#7A7A7A"),
            markeredgewidth=0,
            alpha=min(alpha + 0.25, 1.0),
            label=value,
        )
        for value in order
        if value in label_values
    ]


def scatter_panel(
    ax,
    coords,
    labels,
    palette,
    title,
    xlabel,
    ylabel,
    order=None,
    legend=False,
    alpha: float = 0.5,
    shuffle: bool = False,
    random_state: int = 0,
) -> None:
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    order = order or list(pd.Index(labels).value_counts().index)
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
    plot_df["color"] = plot_df["label"].map(lambda value: palette.get(value, "#7A7A7A"))
    if shuffle:
        plot_df = plot_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=5,
        alpha=alpha,
        linewidths=0,
        c=plot_df["color"],
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(
            handles=legend_handles(order, palette, labels, alpha=alpha),
            frameon=False,
            fontsize=LEGEND_FONT,
            loc="best",
        )


def balanced_batch_indices(labels: pd.Series, random_state: int = 0) -> np.ndarray:
    labels = labels.astype(str)
    counts = labels.value_counts()
    n = counts.min()
    rng = np.random.default_rng(random_state)
    indices: list[int] = []
    values = labels.to_numpy()
    for label in counts.index:
        choices = np.where(values == label)[0]
        indices.extend(rng.choice(choices, size=n, replace=False).tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return indices


def capped_label_indices(labels: pd.Series, max_per_label: int, random_state: int = 0) -> np.ndarray:
    labels = labels.astype(str)
    rng = np.random.default_rng(random_state)
    indices: list[int] = []
    values = labels.to_numpy()
    for label in labels.value_counts().index:
        choices = np.where(values == label)[0]
        n = min(max_per_label, len(choices))
        if n < len(choices):
            chosen = rng.choice(choices, size=n, replace=False)
        else:
            chosen = choices
        indices.extend(np.asarray(chosen).tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return indices


def save_before_after_harmony(combined) -> None:
    umap_before = combined.obsm["X_umap"]
    umap_after = combined.obsm["X_umap_harmony"]
    batch_order = ["batch1", "batch2"]
    cell_type_order = list(combined.obs["cell_type"].value_counts().index)
    batch_idx = balanced_batch_indices(combined.obs["batch_label"], random_state=0)
    cell_idx = capped_label_indices(combined.obs["cell_type"], max_per_label=CELL_TYPE_PLOT_CAP, random_state=1)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.8))
    scatter_panel(
        axes[0, 0],
        umap_before[batch_idx],
        combined.obs["batch_label"].iloc[batch_idx],
        BATCH_PALETTE,
        "Before Harmony: UMAP by Batch (balanced subsample)",
        "UMAP1",
        "UMAP2",
        order=batch_order,
        legend=True,
        alpha=0.35,
        shuffle=True,
        random_state=0,
    )
    scatter_panel(
        axes[0, 1],
        umap_before[cell_idx],
        combined.obs["cell_type"].iloc[cell_idx],
        CELL_TYPE_PALETTE,
        f"Before Harmony: UMAP by Cell Type (capped at {CELL_TYPE_PLOT_CAP:,}/class)",
        "UMAP1",
        "UMAP2",
        order=cell_type_order,
        legend=False,
        alpha=0.28,
        shuffle=False,
        random_state=1,
    )
    scatter_panel(
        axes[1, 0],
        umap_after[batch_idx],
        combined.obs["batch_label"].iloc[batch_idx],
        BATCH_PALETTE,
        "After Harmony: UMAP by Batch (balanced subsample)",
        "UMAP1",
        "UMAP2",
        order=batch_order,
        legend=True,
        alpha=0.35,
        shuffle=True,
        random_state=2,
    )
    scatter_panel(
        axes[1, 1],
        umap_after[cell_idx],
        combined.obs["cell_type"].iloc[cell_idx],
        CELL_TYPE_PALETTE,
        f"After Harmony: UMAP by Cell Type (capped at {CELL_TYPE_PLOT_CAP:,}/class)",
        "UMAP1",
        "UMAP2",
        order=cell_type_order,
        legend=False,
        alpha=0.28,
        shuffle=False,
        random_state=3,
    )

    handles = legend_handles(cell_type_order, CELL_TYPE_PALETTE, combined.obs["cell_type"].iloc[cell_idx], alpha=0.28)
    if handles:
        fig.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="center right",
            bbox_to_anchor=(1.11, 0.5),
            frameon=False,
            fontsize=LEGEND_FONT,
        )
    fig.suptitle("Harmony Removes Batch Separation While Preserving Cell-Type Structure", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "harmony_before_after.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_notebook(tables: dict[str, pd.DataFrame], artifacts: dict[str, Path | int], combined) -> None:
    shared_genes = int(artifacts["shared_genes"])
    batch1_cells = int(tables["batch_counts"].set_index("batch_label").loc["batch1", "cells"])
    batch2_cells = int(tables["batch_counts"].set_index("batch_label").loc["batch2", "cells"])
    rare_classes = tables["cell_type_counts"].query("is_rare_under_2pct")["cell_type"].tolist()
    rare_classes_text = ", ".join(rare_classes)
    metrics_df = tables["batch_effect_metrics"].copy()
    metrics_summary = metrics_df.pivot(index="metric", columns="embedding", values="value").round(4)

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"""# MS2 AB: GSE96583 Data Wrangling and EDA

This notebook documents the milestone 2 data workflow for our PBMC benchmark:

1. Describe the GEO source and raw file layout.
2. Explain what `batch1`, `batch2`, `ctrl`, and `stim` mean.
3. Load the raw data, attach metadata, run strict QC, and align shared genes.
4. Visualize the batch effect before correction.
5. Apply Harmony and verify that batch separation is reduced afterwards.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Data Description

`GSE96583` is the GEO series **"Multiplexing droplet-based single cell RNA-sequencing using genetic barcodes"**. The series contains five PBMC supplementary matrices:

- `batch1`: three matrices labeled `A`, `B`, and `C`
- `batch2`: one control matrix (`ctrl`) and one IFN-beta stimulation matrix (`stim`)

This dataset is a good milestone benchmark because it already includes cell-type labels, a clear batch split, and a realistic condition shift inside `batch2`.

For batch correction we use **Harmony**. Harmony is best known from the R / Seurat ecosystem, but here we use the Python wrapper `harmonypy`, which makes it easy to keep the entire milestone workflow in one notebook. If only an R environment were available, the same step could be inserted as an R cell.

Source links:
- GEO accession: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583>
- GEO supplementary directory: <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/>
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.lines import Line2D
from IPython.display import display

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "scripts" / "prepare_gse96583_ms2.py").exists():
    ROOT = ROOT.parent

if not (ROOT / "scripts" / "prepare_gse96583_ms2.py").exists():
    raise FileNotFoundError("Could not locate scripts/prepare_gse96583_ms2.py from the notebook working directory.")

sys.path.append(str(ROOT / "scripts"))

from prepare_gse96583_ms2 import prepare_gse96583_dataset, raw_file_inventory, sample_manifest

sns.set_theme(style="whitegrid", context="talk")
pd.set_option("display.max_colwidth", 120)

artifacts = prepare_gse96583_dataset(force=False)
artifacts"""
        ),
        nbf.v4.new_markdown_cell(
            """## Access

The preprocessing helper fetches the complete set of files required for milestone 2:

- `GSE96583_RAW.tar` for the raw count matrices and barcode tables
- `GSE96583_batch1.genes.tsv.gz` and `GSE96583_batch2.genes.tsv.gz` for gene symbols
- `GSE96583_batch1.total.tsne.df.tsv.gz` and `GSE96583_batch2.total.tsne.df.tsv.gz` for cell metadata

The file inventory below confirms that the local download contains both the archive members and the extra supplementary metadata tables.
"""
        ),
        nbf.v4.new_code_cell(
            """inventory = raw_file_inventory()
manifest = sample_manifest()

display(inventory)
display(manifest)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Load

The notebook uses three processed artifacts:

- `batch1` strict-QC singlets
- `batch2` strict-QC singlets
- one combined object restricted to the shared post-QC gene space

`batch1` has no explicit `ctrl / stim` annotation, so we keep its condition label as `batch1`. Inside `batch2`, the `stim` field distinguishes `ctrl` from `stim`.
"""
        ),
        nbf.v4.new_code_cell(
            """adata_b1 = sc.read_h5ad(artifacts["batch1_path"])
adata_b2 = sc.read_h5ad(artifacts["batch2_path"])
adata = sc.read_h5ad(artifacts["combined_path"])
stage_counts = pd.read_csv(artifacts["stage_counts_path"])
metrics_df = pd.read_csv(artifacts["metrics_path"])

print("batch1 shape:", adata_b1.shape)
print("batch2 shape:", adata_b2.shape)
print("combined shared-gene shape:", adata.shape)
print("shared genes used for cross-batch EDA:", artifacts["shared_genes"])

display(stage_counts)"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Preprocess

The wrangling pipeline is:

1. Load each GEO matrix and its matching barcode table.
2. Attach GEO metadata (`cell.type` / `cell`, `cluster`, `multiplets`, and condition labels).
3. Apply basic QC: keep cells with at least 200 detected genes and genes seen in at least 3 cells.
4. Keep metadata singlets via `multiplets == "singlet"`.
5. Run an extra strict doublet filter using Scrublet plus conservative high-count / high-gene tail removal.
6. Restrict cross-batch analyses to the {shared_genes} genes shared by both strict-QC batches.
7. Normalize each batch separately, select per-batch HVGs, take their union, then run PCA and Harmony in that shared feature space.

After preprocessing, `batch1` keeps {batch1_cells:,} cells and `batch2` keeps {batch2_cells:,} cells.
"""
        ),
        nbf.v4.new_code_cell(
            """sample_counts = (
    adata.obs.groupby(["batch_label", "sample_accession", "sample_label", "condition"], observed=True)
    .size()
    .rename("cells")
    .reset_index()
    .sort_values(["batch_label", "sample_label"])
)

cell_type_counts = (
    adata.obs["cell_type"].value_counts().rename_axis("cell_type").reset_index(name="cells")
)
cell_type_counts["percent"] = 100 * cell_type_counts["cells"] / cell_type_counts["cells"].sum()
cell_type_counts["is_rare_under_2pct"] = cell_type_counts["percent"] < 2.0

display(sample_counts)
display(cell_type_counts)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Analyze: Class Distribution and Rare Classes

The final label distribution is strongly imbalanced. Broad T-cell and monocyte classes dominate, while rare classes remain small enough that later evaluation should not rely on accuracy alone.
"""
        ),
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(8.5, 5.6))
plot_df = cell_type_counts.sort_values("cells", ascending=True)
palette = {
    "CD4 T cells": "#1b9e77",
    "CD14+ Monocytes": "#d95f02",
    "B cells": "#7570b3",
    "CD8 T cells": "#e7298a",
    "NK cells": "#66a61e",
    "FCGR3A+ Monocytes": "#e6ab02",
    "Dendritic cells": "#a6761d",
    "Megakaryocytes": "#666666",
}

ax.barh(plot_df["cell_type"], plot_df["cells"], color=[palette.get(x, "#777777") for x in plot_df["cell_type"]])
ax.set_title("Class Distribution After Strict QC")
ax.set_xlabel("Cells")
ax.set_ylabel("")

for _, row in plot_df.iterrows():
    ax.text(row["cells"] + plot_df["cells"].max() * 0.01, row["cell_type"], f"{row['percent']:.1f}%", va="center", fontsize=9)

plt.tight_layout()
plt.show()

cell_type_counts.query("is_rare_under_2pct")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Visualize: Batch / Condition Composition and QC

`batch1` and `batch2` are not perfectly balanced, and the condition split only exists inside `batch2`. The QC summary below also shows that `batch2` tends to have higher counts and more detected genes per cell.
"""
        ),
        nbf.v4.new_code_cell(
            """condition_counts = adata.obs["condition"].value_counts().rename_axis("condition").reset_index(name="cells")
composition = pd.crosstab(adata.obs["cell_type"], adata.obs["condition"], normalize="columns").mul(100)
qc_long = adata.obs[["batch_label", "n_genes_by_counts", "total_counts", "sparsity"]].copy()

fig = plt.figure(figsize=(14.5, 9.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], wspace=0.4, hspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
sample_plot = sample_counts.copy()
sample_plot["sample_label"] = pd.Categorical(sample_plot["sample_label"], categories=["A", "B", "C", "ctrl", "stim"], ordered=True)
sample_plot = sample_plot.sort_values("sample_label")
sns.barplot(data=sample_plot, x="sample_label", y="cells", hue="batch_label", ax=ax1)
ax1.set_title("Cells Per Matrix")
ax1.set_xlabel("Sample label")
ax1.set_ylabel("Cells")
ax1.legend(title="Batch", fontsize=11.5, title_fontsize=12.5)

ax2 = fig.add_subplot(gs[0, 1])
sns.barplot(
    data=condition_counts.assign(condition=pd.Categorical(condition_counts["condition"], categories=["batch1", "ctrl", "stim"], ordered=True)).sort_values("condition"),
    x="condition",
    y="cells",
    ax=ax2,
)
ax2.set_title("Batch / Condition Totals")
ax2.set_xlabel("")
ax2.set_ylabel("Cells")

ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(composition.loc[cell_type_counts["cell_type"], ["batch1", "ctrl", "stim"]], cmap="YlGnBu", linewidths=0.4, ax=ax3)
ax3.set_title("Cell-Type Composition by Condition")
ax3.set_xlabel("")
ax3.set_ylabel("")

for i, metric in enumerate(["n_genes_by_counts", "total_counts", "sparsity"]):
    ax = fig.add_subplot(gs[1, i])
    sns.violinplot(data=qc_long, x="batch_label", y=metric, hue="batch_label", cut=0, inner="quartile", legend=False, ax=ax)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Analyze: Batch Effect Before Correction

Before correction, the shared-gene embedding is dominated by batch. We quantify this with silhouette scores and local-neighbor purity:

- high `silhouette_batch` means cells cluster by batch
- high `neighbor_same_batch` means each cell's local neighborhood is mostly the same batch
- `silhouette_cell_type` and `neighbor_same_cell_type` tell us whether biological structure is preserved

For fair visualization, the batch-colored UMAP panels below use an equal-size per-batch subsample plus randomized point order. This avoids the larger `batch2` cloud visually covering the smaller `batch1` cloud.

For the cell-type panels, each abundant class is capped for plotting so that major lymphoid populations do not visually swamp nearby smaller classes.
"""
        ),
        nbf.v4.new_code_cell(
            """metrics_summary = metrics_df.pivot(index="metric", columns="embedding", values="value").round(4)
metrics_summary"""
        ),
        nbf.v4.new_code_cell(
            """batch_palette = {"batch1": "#133C55", "batch2": "#D05D4E"}
cell_palette = {
    "CD4 T cells": "#1b9e77",
    "CD14+ Monocytes": "#d95f02",
    "B cells": "#7570b3",
    "CD8 T cells": "#e7298a",
    "NK cells": "#66a61e",
    "FCGR3A+ Monocytes": "#e6ab02",
    "Dendritic cells": "#a6761d",
    "Megakaryocytes": "#666666",
}
cell_type_order = list(adata.obs["cell_type"].value_counts().index)

def legend_handles(order, palette, labels, alpha=0.5):
    label_values = set(pd.Series(labels).astype(str).tolist())
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=palette.get(value, "#777777"),
            markeredgewidth=0,
            alpha=min(alpha + 0.25, 1.0),
            label=value,
        )
        for value in order
        if value in label_values
    ]

def balanced_batch_indices(labels, random_state=0):
    labels = pd.Series(labels).astype(str)
    counts = labels.value_counts()
    n = counts.min()
    rng = np.random.default_rng(random_state)
    indices = []
    values = labels.to_numpy()
    for label in counts.index:
        choices = np.where(values == label)[0]
        indices.extend(rng.choice(choices, size=n, replace=False).tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return indices

def capped_label_indices(labels, max_per_label, random_state=0):
    labels = pd.Series(labels).astype(str)
    rng = np.random.default_rng(random_state)
    indices = []
    values = labels.to_numpy()
    for label in labels.value_counts().index:
        choices = np.where(values == label)[0]
        n = min(max_per_label, len(choices))
        if n < len(choices):
            choices = rng.choice(choices, size=n, replace=False)
        indices.extend(np.asarray(choices).tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return indices

def draw_embedding(ax, coords, labels, palette, title, xlabel, ylabel, order, alpha=0.5, shuffle=False, random_state=0):
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
    plot_df["color"] = plot_df["label"].map(lambda value: palette.get(value, "#777777"))
    if shuffle:
        plot_df = plot_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    ax.scatter(plot_df["x"], plot_df["y"], s=5, alpha=alpha, linewidths=0, c=plot_df["color"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

batch_plot_idx = balanced_batch_indices(adata.obs["batch_label"], random_state=0)
cell_plot_idx = capped_label_indices(adata.obs["cell_type"], max_per_label=2500, random_state=1)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
draw_embedding(
    axes[0],
    adata.obsm["X_umap"][batch_plot_idx],
    adata.obs["batch_label"].iloc[batch_plot_idx],
    batch_palette,
    "Before Harmony: UMAP by Batch (balanced subsample)",
    "UMAP1",
    "UMAP2",
    ["batch1", "batch2"],
    alpha=0.35,
    shuffle=True,
    random_state=0,
)
draw_embedding(
    axes[1],
    adata.obsm["X_umap"][cell_plot_idx],
    adata.obs["cell_type"].iloc[cell_plot_idx],
    cell_palette,
    "Before Harmony: UMAP by Cell Type (capped at 2,500/class)",
    "UMAP1",
    "UMAP2",
    cell_type_order,
    alpha=0.28,
    shuffle=False,
    random_state=1,
)
axes[0].legend(handles=legend_handles(["batch1", "batch2"], batch_palette, adata.obs["batch_label"].iloc[batch_plot_idx], alpha=0.35), frameon=False, fontsize=11.5, loc="best")
cell_handles = legend_handles(cell_type_order, cell_palette, adata.obs["cell_type"].iloc[cell_plot_idx], alpha=0.28)
fig.legend(cell_handles, [handle.get_label() for handle in cell_handles], frameon=False, fontsize=11.5, bbox_to_anchor=(1.11, 0.5), loc="center right")
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Batch Correction: Harmony

Harmony correction has already been computed and stored in the processed combined AnnData object as:

- `adata.obsm["X_pca_harmony"]`
- `adata.obsm["X_umap_harmony"]`

For milestone 2, this is the right place to correct batch effect: we first demonstrate the problem, then show that Harmony reduces batch-driven structure while keeping cell-type neighborhoods largely intact.
"""
        ),
        nbf.v4.new_code_cell(
            """metric_plot = metrics_df.copy()
metric_plot["embedding"] = metric_plot["embedding"].map({"before_harmony": "Before Harmony", "after_harmony": "After Harmony"})
metric_order = [
    "silhouette_batch",
    "silhouette_condition",
    "silhouette_cell_type",
    "neighbor_same_batch",
    "neighbor_same_condition",
    "neighbor_same_cell_type",
]

fig, ax = plt.subplots(figsize=(11.5, 5.2))
sns.barplot(data=metric_plot, x="metric", y="value", hue="embedding", order=metric_order, ax=ax)
ax.set_title("Batch-Effect Metrics Before and After Harmony")
ax.set_xlabel("")
ax.set_ylabel("Metric value")
ax.tick_params(axis="x", rotation=18)
ax.legend(title="", fontsize=11.5)
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Visualize: After Harmony

After Harmony, the embedding is no longer dominated by batch. The batch silhouette becomes slightly negative, while cell-type separation remains strong. The batch-colored panel again uses a balanced subsample and randomized draw order so the visual is not biased by the larger batch.
"""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
draw_embedding(
    axes[0],
    adata.obsm["X_umap_harmony"][batch_plot_idx],
    adata.obs["batch_label"].iloc[batch_plot_idx],
    batch_palette,
    "After Harmony: UMAP by Batch (balanced subsample)",
    "UMAP1",
    "UMAP2",
    ["batch1", "batch2"],
    alpha=0.35,
    shuffle=True,
    random_state=2,
)
draw_embedding(
    axes[1],
    adata.obsm["X_umap_harmony"][cell_plot_idx],
    adata.obs["cell_type"].iloc[cell_plot_idx],
    cell_palette,
    "After Harmony: UMAP by Cell Type (capped at 2,500/class)",
    "UMAP1",
    "UMAP2",
    cell_type_order,
    alpha=0.28,
    shuffle=False,
    random_state=3,
)
axes[0].legend(handles=legend_handles(["batch1", "batch2"], batch_palette, adata.obs["batch_label"].iloc[batch_plot_idx], alpha=0.35), frameon=False, fontsize=11.5, loc="best")
cell_handles = legend_handles(cell_type_order, cell_palette, adata.obs["cell_type"].iloc[cell_plot_idx], alpha=0.28)
fig.legend(cell_handles, [handle.get_label() for handle in cell_handles], frameon=False, fontsize=11.5, bbox_to_anchor=(1.11, 0.5), loc="center right")
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Insights

- The wrangling pipeline keeps **{batch1_cells:,} batch1 cells** and **{batch2_cells:,} batch2 cells**, then aligns them on **{shared_genes:,} shared genes**.
- `batch2` is almost balanced between `ctrl` and `stim`, but the full benchmark still has batch imbalance because `batch1` contributes fewer cells.
- The label space is usable but imbalanced; the rare classes are **{rare_classes_text}**.
- Before correction, batch neighborhoods are almost perfectly separated.
- After Harmony, batch silhouette drops from **{metrics_summary.loc['silhouette_batch', 'before_harmony']:.4f}** to **{metrics_summary.loc['silhouette_batch', 'after_harmony']:.4f}**, while cell-type silhouette increases from **{metrics_summary.loc['silhouette_cell_type', 'before_harmony']:.4f}** to **{metrics_summary.loc['silhouette_cell_type', 'after_harmony']:.4f}**.
- For milestone 2, the main story is now complete: identify the batch effect, correct it with Harmony, and verify that the corrected embedding is more appropriate for downstream modeling.
"""
        ),
    ]

    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with (DELIV / "MS2_AB_Data_Wrangling.ipynb").open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)


def build_slides_markdown(tables: dict[str, pd.DataFrame], artifacts: dict[str, Path | int]) -> None:
    stage_counts = tables["stage_counts"]
    metrics_df = tables["batch_effect_metrics"]
    metric_lookup = metrics_df.set_index(["embedding", "metric"])["value"]
    batch1_final = int(stage_counts.query("batch_label == 'batch1' and stage == 'strict_qc_final'")["cells"].iloc[0])
    batch2_final = int(stage_counts.query("batch_label == 'batch2' and stage == 'strict_qc_final'")["cells"].iloc[0])
    shared_genes = int(artifacts["shared_genes"])
    rare_classes = tables["cell_type_counts"].query("is_rare_under_2pct")["cell_type"].tolist()
    before_batch = metric_lookup[("before_harmony", "silhouette_batch")]
    after_batch = metric_lookup[("after_harmony", "silhouette_batch")]
    before_cell = metric_lookup[("before_harmony", "silhouette_cell_type")]
    after_cell = metric_lookup[("after_harmony", "silhouette_cell_type")]

    slides = f"""# MS2 Sections 1-2 Slides

## Slide 1: Data Source and Why GSE96583
- Source: GEO accession `GSE96583`, *Multiplexing droplet-based single cell RNA-sequencing using genetic barcodes*.
- The study gives us five PBMC supplementary matrices, usable cell-type labels, and both batch and condition shift inside one public benchmark.
- `batch1` consists of samples `A`, `B`, and `C`; `batch2` consists of `ctrl` and `stim`.
- This lets us tell a clean wrangling story before moving on to baseline modeling and SSL.

## Slide 2: GEO File Structure
- The raw GEO release is fragmented across count matrices, barcode files, gene tables, and cell metadata tables.
- `GSE96583_RAW.tar` only contains matrices + barcodes, so the metadata TSVs must be downloaded separately from the supplementary directory.
- That fragmented file structure is why explicit data wrangling is necessary.
![Raw file sizes](figures/raw_file_sizes.png)

## Slide 3: Data Wrangling Pipeline
- Step 1: download the archive plus the batch-specific gene and metadata tables.
- Step 2: load each matrix, align barcodes, and attach GEO metadata.
- Step 3: apply QC (`min_genes >= 200`, `min_cells >= 3`) and keep metadata singlets.
- Step 4: run strict residual doublet filtering with Scrublet plus conservative tail trimming.
- Step 5: align `batch1` and `batch2` on {shared_genes} post-QC shared genes for cross-batch EDA.
- Final sizes: `batch1 = {batch1_final:,}` cells, `batch2 = {batch2_final:,}` cells.
![Wrangling summary](figures/wrangling_summary.png)

## Slide 4: Class Distribution and Rare Classes
- The processed dataset preserves the main immune populations needed for downstream classification.
- The class distribution is clearly imbalanced, so later evaluation should emphasize macro-F1 and per-class behavior.
- Rare classes under 2% of the final benchmark: {", ".join(rare_classes)}.
![Class distribution](figures/class_distribution.png)

## Slide 5: Batch Effect Before Correction
- Before correction, the shared-gene UMAP is strongly separated by batch.
- Quantitatively, `silhouette_batch = {before_batch:.4f}` before Harmony.
- This is exactly the problem we want milestone 2 to surface and fix.
![Before and after Harmony](figures/harmony_before_after.png)

## Slide 6: Harmony Batch Correction
- Harmony is a strong batch-correction method that is most commonly introduced in the R / Seurat ecosystem.
- We use the Python wrapper `harmonypy` so the notebook can stay end-to-end reproducible in Python.
- After Harmony, `silhouette_batch` drops to `{after_batch:.4f}`, while `silhouette_cell_type` improves from `{before_cell:.4f}` to `{after_cell:.4f}`.
![Batch effect metrics](figures/batch_effect_metrics.png)

## Slide 7: Composition and QC Context
- `batch2` is nearly balanced between `ctrl` and `stim`, while `batch1` contributes three separate source matrices.
- QC metrics also differ by batch, so the benchmark includes both technical and biological variation.
![Batch and condition distribution](figures/batch_condition_distribution.png)
![QC metrics](figures/qc_metrics.png)
"""
    (DELIV / "MS2_AB_Sections_1_2_Slides.md").write_text(slides, encoding="utf-8")


def build_notes(tables: dict[str, pd.DataFrame], artifacts: dict[str, Path | int]) -> None:
    shared_genes = int(artifacts["shared_genes"])
    notes = f"""# MS2 Sections 1-2 Speaker Notes

## Slide 1: Data Source and Why GSE96583
- Say explicitly that we scoped MS2 to one dataset with defensible labels and real shift.
- Mention both the batch split and the condition split.

## Slide 2: GEO File Structure
- Explain that the archive alone is not enough because gene tables and metadata live outside the tarball.
- Point out that wrangling begins before modeling because the raw release is fragmented.

## Slide 3: Data Wrangling Pipeline
- Walk through download, metadata attachment, basic QC, metadata singlets, extra Scrublet filtering, and shared-gene alignment.
- Emphasize the final shared-gene count of {shared_genes}.

## Slide 4: Class Distribution and Rare Classes
- Stress that the benchmark is usable but imbalanced.
- Set up later evaluation choices such as macro-F1.

## Slide 5: Batch Effect Before Correction
- Explicitly say that the uncorrected embedding is too batch-driven.
- Use this slide to motivate Harmony rather than jumping straight to the fix.

## Slide 6: Harmony Batch Correction
- Mention that Harmony is a standard batch-correction tool and often appears in the R / Seurat workflow.
- The main result is that batch separation drops while cell-type structure is preserved.

## Slide 7: Composition and QC Context
- End by reminding the audience that this dataset still contains real condition structure and QC differences.
- That is why we treat Harmony as correction for technical batch effect, not as a way to erase all variation.
"""
    (DELIV / "MS2_AB_Speaker_Notes.md").write_text(notes, encoding="utf-8")


def wrap_lines(text: str, width: int = 70) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def new_slide() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(13.33, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.89),
            0.18,
            0.055,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=NAVY,
            edgecolor="none",
        )
    )
    return fig, ax


def add_header(ax: plt.Axes, section: str, title: str, subtitle: str) -> None:
    ax.text(0.052, 0.916, section.upper(), color="white", fontsize=11, fontweight="bold", va="center")
    ax.text(0.05, 0.84, title, color=NAVY, fontsize=25, fontweight="bold", va="top")
    ax.text(0.05, 0.79, subtitle, color=MUTED, fontsize=12.5, va="top")


def add_bullets(ax: plt.Axes, bullets: list[str], x: float, y: float, width: int = 52, gap: float = 0.085) -> None:
    cursor = y
    for bullet in bullets:
        wrapped = wrap_lines(bullet, width=width)
        ax.text(x, cursor, f"• {wrapped}", fontsize=14.5, color=INK, va="top")
        cursor -= gap + 0.02 * wrapped.count("\n")


def add_image(ax: plt.Axes, image_path: Path, left: float, bottom: float, width: float, height: float) -> None:
    image = plt.imread(image_path)
    img_ax = ax.figure.add_axes([left, bottom, width, height])
    img_ax.imshow(image)
    img_ax.axis("off")


def add_footer(ax: plt.Axes, text: str, page: int) -> None:
    ax.text(0.05, 0.035, text, fontsize=10.5, color=MUTED, va="center")
    ax.text(0.95, 0.035, str(page), fontsize=10.5, color=MUTED, va="center", ha="right")


def build_pdf_slides(tables: dict[str, pd.DataFrame], artifacts: dict[str, Path | int]) -> None:
    stage_counts = tables["stage_counts"]
    metrics_df = tables["batch_effect_metrics"]
    metric_lookup = metrics_df.set_index(["embedding", "metric"])["value"]
    batch1_final = int(stage_counts.query("batch_label == 'batch1' and stage == 'strict_qc_final'")["cells"].iloc[0])
    batch2_final = int(stage_counts.query("batch_label == 'batch2' and stage == 'strict_qc_final'")["cells"].iloc[0])
    shared_genes = int(artifacts["shared_genes"])
    rare_classes = tables["cell_type_counts"].query("is_rare_under_2pct")["cell_type"].tolist()
    before_batch = metric_lookup[("before_harmony", "silhouette_batch")]
    after_batch = metric_lookup[("after_harmony", "silhouette_batch")]
    before_cell = metric_lookup[("before_harmony", "silhouette_cell_type")]
    after_cell = metric_lookup[("after_harmony", "silhouette_cell_type")]

    pdf_path = DELIV / "MS2_AB_Sections_1_2_Slides.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = new_slide()
        add_header(
            ax,
            "Section 1",
            "Data Source: GSE96583",
            "A single PBMC benchmark with labels, batch shift, and condition shift is enough for a defensible milestone 2 story.",
        )
        add_bullets(
            ax,
            [
                "GSE96583 is public on GEO and directly downloadable from the accession page plus the supplementary FTP directory.",
                "batch1 contains samples A, B, and C, while batch2 contains ctrl and stim.",
                "This gives us both technical and biological shift without depending on a second dataset.",
            ],
            x=0.06,
            y=0.68,
        )
        add_image(ax, FIG / "raw_file_sizes.png", 0.57, 0.18, 0.33, 0.52)
        add_footer(ax, "Dataset source and file access", 1)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            "Section 2",
            "Raw File Structure",
            "The GEO release is fragmented: matrices and barcodes live in the tarball, but the metadata needed for labels and strict doublet filtering lives outside it.",
        )
        add_bullets(
            ax,
            [
                "batch1 matrices are GSM2560245_A, GSM2560246_B, and GSM2560247_C.",
                "batch2 matrices are GSM2560248_2.1 (ctrl) and GSM2560249_2.2 (stim).",
                "Gene tables and tsne.df metadata must be fetched separately before any valid wrangling can start.",
            ],
            x=0.06,
            y=0.67,
        )
        add_image(ax, FIG / "raw_file_sizes.png", 0.56, 0.16, 0.36, 0.56)
        add_footer(ax, "Raw GEO structure", 2)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            "Section 2",
            "Data Wrangling Pipeline",
            f"After metadata alignment, basic QC, metadata singlets, and strict residual doublet filtering, batch1 keeps {batch1_final:,} cells and batch2 keeps {batch2_final:,} cells.",
        )
        add_bullets(
            ax,
            [
                "QC keeps cells with at least 200 detected genes and genes seen in at least 3 cells.",
                "We first keep GEO metadata singlets, then run Scrublet to catch residual doublets.",
                "We also trim conservative extreme count / gene tails that look doublet-like.",
                f"Cross-batch EDA then aligns both batches on {shared_genes} shared genes.",
            ],
            x=0.06,
            y=0.69,
        )
        add_image(ax, FIG / "wrangling_summary.png", 0.52, 0.14, 0.40, 0.60)
        add_footer(ax, "Wrangling decisions and outcomes", 3)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            "Section 2",
            "Class Imbalance",
            "The processed benchmark preserves the main immune classes, but rare classes are still scarce enough to matter for evaluation.",
        )
        add_bullets(
            ax,
            [
                "CD4 T cells and monocytes dominate the benchmark.",
                f"Rare classes under 2% are {', '.join(rare_classes)}.",
                "This is why later modeling should emphasize macro-F1 and class-wise behavior.",
            ],
            x=0.06,
            y=0.68,
        )
        add_image(ax, FIG / "class_distribution.png", 0.55, 0.14, 0.36, 0.60)
        add_footer(ax, "Class distribution and rare classes", 4)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            "Section 2",
            "Batch Effect Before Correction",
            "The uncorrected shared-gene embedding is too batch-driven, so batch correction is justified inside milestone 2.",
        )
        add_bullets(
            ax,
            [
                f"Before Harmony, silhouette_batch is {before_batch:.4f}.",
                "The pre-correction UMAP shows large batch-driven separation.",
                "We want the corrected representation to mix batches without destroying cell-type structure.",
            ],
            x=0.06,
            y=0.69,
        )
        add_image(ax, FIG / "harmony_before_after.png", 0.46, 0.10, 0.47, 0.64)
        add_footer(ax, "Uncorrected vs corrected embedding", 5)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            "Section 2",
            "Harmony Correction",
            "Harmony strongly reduces batch structure while preserving biological neighborhoods.",
        )
        add_bullets(
            ax,
            [
                "Harmony is widely used for batch correction and is especially well known from the R / Seurat workflow.",
                f"silhouette_batch drops from {before_batch:.4f} to {after_batch:.4f}.",
                f"silhouette_cell_type improves from {before_cell:.4f} to {after_cell:.4f}.",
            ],
            x=0.06,
            y=0.69,
        )
        add_image(ax, FIG / "batch_effect_metrics.png", 0.46, 0.14, 0.47, 0.56)
        add_footer(ax, "Harmony metrics", 6)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            "Section 2",
            "Composition and QC Context",
            "Even after batch correction, the benchmark still contains real condition structure and QC differences.",
        )
        add_bullets(
            ax,
            [
                "batch2 is nearly balanced between ctrl and stim.",
                "Cell-type composition differs across batch1, ctrl, and stim.",
                "QC differences are part of the realism of the benchmark and should be described rather than hidden.",
            ],
            x=0.06,
            y=0.69,
        )
        add_image(ax, FIG / "batch_condition_distribution.png", 0.52, 0.42, 0.40, 0.28)
        add_image(ax, FIG / "qc_metrics.png", 0.52, 0.10, 0.40, 0.24)
        add_footer(ax, "Composition and QC context", 7)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    artifacts = prepare_gse96583_dataset(force=False)
    adata_b1 = sc.read_h5ad(artifacts["batch1_path"])
    adata_b2 = sc.read_h5ad(artifacts["batch2_path"])
    combined = sc.read_h5ad(artifacts["combined_path"])
    stage_counts = pd.read_csv(artifacts["stage_counts_path"])
    metrics_df = pd.read_csv(artifacts["metrics_path"])

    tables = build_tables(adata_b1, adata_b2, combined, stage_counts, metrics_df)
    save_tables(tables)
    save_raw_file_sizes(tables["raw_file_inventory"])
    save_wrangling_summary(tables["stage_counts"])
    save_class_distribution(tables["cell_type_counts"])
    save_batch_condition_distribution(
        tables["sample_counts"],
        tables["condition_counts"],
        tables["condition_composition"],
    )
    save_qc_metrics(tables["qc_long"])
    save_batch_effect_metrics(tables["batch_effect_metrics"])
    save_before_after_harmony(combined)
    build_notebook(tables, artifacts, combined)
    build_slides_markdown(tables, artifacts)
    build_notes(tables, artifacts)
    build_pdf_slides(tables, artifacts)
    print(f"Built MS2 milestone assets in {DELIV}")


if __name__ == "__main__":
    main()
