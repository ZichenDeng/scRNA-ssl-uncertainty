#!/usr/bin/env python3
"""Download, preprocess, and summarize the GSE96583 PBMC benchmark for MS2."""

from __future__ import annotations

import argparse
import gzip
import tarfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy import io
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "GSE96583"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

RAW_ARCHIVE_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96583&format=file"
SUPPL_URL_ROOT = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl"

EXPECTED_RAW_ARCHIVE_MEMBERS = [
    "GSM2560245_A.mat.gz",
    "GSM2560245_barcodes.tsv.gz",
    "GSM2560246_B.mat.gz",
    "GSM2560246_barcodes.tsv.gz",
    "GSM2560247_C.mat.gz",
    "GSM2560247_barcodes.tsv.gz",
    "GSM2560248_2.1.mtx.gz",
    "GSM2560248_barcodes.tsv.gz",
    "GSM2560249_2.2.mtx.gz",
    "GSM2560249_barcodes.tsv.gz",
]

SUPPLEMENTARY_FILES = [
    "GSE96583_batch1.genes.tsv.gz",
    "GSE96583_batch1.total.tsne.df.tsv.gz",
    "GSE96583_batch2.genes.tsv.gz",
    "GSE96583_batch2.total.tsne.df.tsv.gz",
    "GSE96583_genes.txt.gz",
    "filelist.txt",
]

BATCH1_SAMPLE_MAP = {
    "GSM2560245_A": "A",
    "GSM2560246_B": "B",
    "GSM2560247_C": "C",
}

BATCH2_SAMPLE_MAP = {
    "GSM2560248_2.1": "ctrl",
    "GSM2560249_2.2": "stim",
}

BATCH1_FILES = list(BATCH1_SAMPLE_MAP)
BATCH2_FILES = list(BATCH2_SAMPLE_MAP)

MIN_GENES = 200
MIN_CELLS = 3
SCRUBLET_EXPECTED_DOUBLET_RATE = 0.08
TAIL_QUANTILE = 0.995
HVG_PER_BATCH = 2000
UMAP_MIN_DIST = 0.2


def ensure_dirs() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url} -> {destination}")
    urlretrieve(url, destination)


def extract_archive_members(archive_path: Path, out_dir: Path, expected_members: Iterable[str]) -> None:
    missing = [member for member in expected_members if not (out_dir / member).exists()]
    if not missing:
        return

    print(f"Extracting {archive_path} -> {out_dir}")
    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(out_dir)


def ensure_gse96583_raw_data() -> Path:
    ensure_dirs()
    archive_path = RAW_ROOT.parent / "GSE96583.tar"

    if not archive_path.exists():
        download_file(RAW_ARCHIVE_URL, archive_path)

    extract_archive_members(archive_path, RAW_ROOT, EXPECTED_RAW_ARCHIVE_MEMBERS)

    for filename in SUPPLEMENTARY_FILES:
        download_file(f"{SUPPL_URL_ROOT}/{filename}", RAW_ROOT / filename)

    return RAW_ROOT


def raw_file_inventory(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or RAW_ROOT
    rows = []
    for path in sorted(raw_dir.glob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "file": path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "kind": infer_raw_kind(path.name),
            }
        )
    return pd.DataFrame(rows)


def infer_raw_kind(name: str) -> str:
    if name.endswith(".mat.gz") or name.endswith(".mtx.gz"):
        return "count matrix"
    if "barcodes" in name:
        return "barcode table"
    if "genes" in name:
        return "gene metadata"
    if "tsne" in name:
        return "cell metadata"
    if name.endswith(".txt"):
        return "inventory"
    return "other"


def sample_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "batch_label": "batch1",
                "sample_accession": "GSM2560245_A",
                "sample_label": "A",
                "condition": "batch1",
                "matrix_format": "MAT",
                "description": "batch 1 sample A",
            },
            {
                "batch_label": "batch1",
                "sample_accession": "GSM2560246_B",
                "sample_label": "B",
                "condition": "batch1",
                "matrix_format": "MAT",
                "description": "batch 1 sample B",
            },
            {
                "batch_label": "batch1",
                "sample_accession": "GSM2560247_C",
                "sample_label": "C",
                "condition": "batch1",
                "matrix_format": "MAT",
                "description": "batch 1 sample C",
            },
            {
                "batch_label": "batch2",
                "sample_accession": "GSM2560248_2.1",
                "sample_label": "ctrl",
                "condition": "ctrl",
                "matrix_format": "MTX",
                "description": "batch 2 control",
            },
            {
                "batch_label": "batch2",
                "sample_accession": "GSM2560249_2.2",
                "sample_label": "stim",
                "condition": "stim",
                "matrix_format": "MTX",
                "description": "batch 2 IFN-beta stimulation",
            },
        ]
    )


def read_gene_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["ensembl_id", "gene_symbol"])


def read_barcodes(path: Path) -> pd.Series:
    return pd.read_csv(path, sep="\t", header=None).iloc[:, 0].astype(str)


def read_matrix_market(path: Path):
    with gzip.open(path, "rt") as handle:
        return io.mmread(handle).tocsr()


def load_subset(file_stems: list[str], genes_file: str, batch_label: str) -> ad.AnnData:
    genes = read_gene_table(RAW_ROOT / genes_file)
    adatas: list[ad.AnnData] = []

    for stem in file_stems:
        suffix = ".mat.gz" if (RAW_ROOT / f"{stem}.mat.gz").exists() else ".mtx.gz"
        matrix = read_matrix_market(RAW_ROOT / f"{stem}{suffix}").T.tocsr()
        barcodes = read_barcodes(RAW_ROOT / f"{stem.split('_')[0]}_barcodes.tsv.gz")

        if matrix.shape[0] != len(barcodes):
            raise ValueError(f"Barcode count mismatch for {stem}: {matrix.shape[0]} vs {len(barcodes)}")
        if matrix.shape[1] != len(genes):
            raise ValueError(f"Gene count mismatch for {stem}: {matrix.shape[1]} vs {len(genes)}")

        adata = ad.AnnData(X=matrix)
        adata.obs_names = pd.Index([f"{stem}:{barcode}" for barcode in barcodes], dtype=str)
        adata.var_names = genes["gene_symbol"].astype(str)
        adata.var["ensembl_id"] = genes["ensembl_id"].astype(str).values
        adata.var_names_make_unique()
        adata.obs["sample_accession"] = stem
        adata.obs["batch_label"] = batch_label
        adatas.append(adata)

    return ad.concat(adatas, join="inner", merge="same")


def _finalize_obs_dtypes(adata: ad.AnnData, numeric_columns: list[str]) -> ad.AnnData:
    for column in adata.obs.columns:
        if column in numeric_columns:
            adata.obs[column] = pd.to_numeric(adata.obs[column], errors="coerce")
        elif adata.obs[column].dtype == "object":
            adata.obs[column] = adata.obs[column].astype("category")
    return adata


def annotate_batch1(adata: ad.AnnData) -> ad.AnnData:
    meta = pd.read_csv(RAW_ROOT / "GSE96583_batch1.total.tsne.df.tsv.gz", sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    adata = adata.copy()
    barcodes = adata.obs_names.to_series().str.split(":", n=1).str[1]
    sample_codes = adata.obs["sample_accession"].map(BATCH1_SAMPLE_MAP)

    for column in meta.columns:
        adata.obs[column] = pd.NA

    for code in sorted(sample_codes.dropna().unique()):
        mask = sample_codes == code
        joined = meta[meta["batch"] == code].reindex(barcodes[mask])
        for column in meta.columns:
            adata.obs.loc[mask, column] = joined[column].values

    adata.obs["sample_label"] = sample_codes
    adata.obs["condition"] = "batch1"
    adata.obs["cell_type"] = adata.obs["cell.type"].astype(str)
    adata.obs["dataset"] = "GSE96583"

    return _finalize_obs_dtypes(adata, ["tsne1", "tsne2", "cluster", "ind"])


def annotate_batch2(adata: ad.AnnData) -> ad.AnnData:
    meta = pd.read_csv(RAW_ROOT / "GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    adata = adata.copy()
    barcodes = adata.obs_names.to_series().str.split(":", n=1).str[1]
    sample_conditions = adata.obs["sample_accession"].map(BATCH2_SAMPLE_MAP)

    for column in meta.columns:
        adata.obs[column] = pd.NA

    for stim in sorted(sample_conditions.dropna().unique()):
        mask = sample_conditions == stim
        joined = meta[meta["stim"] == stim].reindex(barcodes[mask])
        for column in meta.columns:
            adata.obs.loc[mask, column] = joined[column].values

    adata.obs["sample_label"] = sample_conditions
    adata.obs["condition"] = adata.obs["stim"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell"].astype(str)
    adata.obs["dataset"] = "GSE96583"

    return _finalize_obs_dtypes(adata, ["tsne1", "tsne2", "cluster", "ind"])


def add_qc_metrics(adata: ad.AnnData) -> None:
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    adata.obs["sparsity"] = 1.0 - (adata.X.getnnz(axis=1) / adata.n_vars)


def preprocess_batch(adata: ad.AnnData) -> tuple[ad.AnnData, dict[str, int]]:
    stage_summary = {
        "raw_cells": int(adata.n_obs),
        "raw_genes": int(adata.n_vars),
    }

    add_qc_metrics(adata)
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    add_qc_metrics(adata)

    stage_summary["qc_cells"] = int(adata.n_obs)
    stage_summary["qc_genes"] = int(adata.n_vars)

    adata = adata[adata.obs["multiplets"] == "singlet"].copy()
    add_qc_metrics(adata)
    stage_summary["metadata_singlet_cells"] = int(adata.n_obs)
    stage_summary["metadata_singlet_genes"] = int(adata.n_vars)

    scrublet_input = adata.copy()
    sc.pp.scrublet(
        scrublet_input,
        expected_doublet_rate=SCRUBLET_EXPECTED_DOUBLET_RATE,
        random_state=0,
        verbose=False,
    )
    adata.obs["doublet_score"] = scrublet_input.obs["doublet_score"].to_numpy()
    adata.obs["predicted_doublet"] = scrublet_input.obs["predicted_doublet"].to_numpy()

    gene_cutoff = float(adata.obs["n_genes_by_counts"].quantile(TAIL_QUANTILE))
    count_cutoff = float(adata.obs["total_counts"].quantile(TAIL_QUANTILE))
    adata.obs["high_gene_tail"] = adata.obs["n_genes_by_counts"] > gene_cutoff
    adata.obs["high_count_tail"] = adata.obs["total_counts"] > count_cutoff
    adata.obs["tail_outlier"] = adata.obs["high_gene_tail"] | adata.obs["high_count_tail"]
    adata.obs["extra_doublet_flag"] = adata.obs["predicted_doublet"] | adata.obs["tail_outlier"]

    stage_summary["scrublet_flagged"] = int(adata.obs["predicted_doublet"].sum())
    stage_summary["tail_outliers"] = int(adata.obs["tail_outlier"].sum())
    stage_summary["strict_qc_removed"] = int(adata.obs["extra_doublet_flag"].sum())

    adata = adata[~adata.obs["extra_doublet_flag"]].copy()
    add_qc_metrics(adata)
    adata.obs["is_singlet"] = True
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs["batch_label"] = adata.obs["batch_label"].astype("category")
    adata.obs["condition"] = adata.obs["condition"].astype("category")
    adata.obs["sample_label"] = adata.obs["sample_label"].astype("category")

    stage_summary["final_cells"] = int(adata.n_obs)
    stage_summary["final_genes"] = int(adata.n_vars)

    return adata, stage_summary


def compute_embedding_metrics(embedding, obs: pd.DataFrame, label: str) -> pd.DataFrame:
    rep = embedding[:, :20]
    knn = NearestNeighbors(n_neighbors=16).fit(rep)
    neighbor_idx = knn.kneighbors(return_distance=False)[:, 1:]

    batch = obs["batch_label"].astype(str).to_numpy()
    condition = obs["condition"].astype(str).to_numpy()
    cell_type = obs["cell_type"].astype(str).to_numpy()

    return pd.DataFrame(
        [
            {"metric": "silhouette_batch", "value": float(silhouette_score(rep, batch)), "embedding": label},
            {"metric": "silhouette_condition", "value": float(silhouette_score(rep, condition)), "embedding": label},
            {"metric": "silhouette_cell_type", "value": float(silhouette_score(rep, cell_type)), "embedding": label},
            {
                "metric": "neighbor_same_batch",
                "value": float((batch[neighbor_idx] == batch[:, None]).mean()),
                "embedding": label,
            },
            {
                "metric": "neighbor_same_condition",
                "value": float((condition[neighbor_idx] == condition[:, None]).mean()),
                "embedding": label,
            },
            {
                "metric": "neighbor_same_cell_type",
                "value": float((cell_type[neighbor_idx] == cell_type[:, None]).mean()),
                "embedding": label,
            },
        ]
    )


def add_harmony_correction(combined: ad.AnnData) -> pd.DataFrame:
    import harmonypy as hm

    harmony_out = hm.run_harmony(combined.obsm["X_pca"], combined.obs, ["batch_label"], verbose=False)
    combined.obsm["X_pca_harmony"] = harmony_out.Z_corr.astype("float32", copy=False)

    temp = ad.AnnData(obs=combined.obs.copy())
    temp.obsm["X_pca_harmony"] = combined.obsm["X_pca_harmony"]
    sc.pp.neighbors(temp, use_rep="X_pca_harmony", n_neighbors=15)
    sc.tl.umap(temp, min_dist=UMAP_MIN_DIST)

    combined.obsm["X_umap_harmony"] = temp.obsm["X_umap"]
    combined.uns["umap_harmony"] = temp.uns["umap"]

    before = compute_embedding_metrics(combined.obsm["X_pca"], combined.obs, "before_harmony")
    after = compute_embedding_metrics(combined.obsm["X_pca_harmony"], combined.obs, "after_harmony")
    metrics_df = pd.concat([before, after], ignore_index=True)
    combined.uns["batch_effect_metrics"] = {
        row["metric"] + "__" + row["embedding"]: float(row["value"]) for _, row in metrics_df.iterrows()
    }
    return metrics_df


def build_combined_shared_dataset(batch1: ad.AnnData, batch2: ad.AnnData) -> tuple[ad.AnnData, int, pd.DataFrame]:
    shared_genes = batch1.var_names.intersection(batch2.var_names)
    batch1_counts = batch1[:, shared_genes].copy()
    batch2_counts = batch2[:, shared_genes].copy()
    batch1_shared = batch1_counts.copy()
    batch2_shared = batch2_counts.copy()

    for adata in (batch1_shared, batch2_shared):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=HVG_PER_BATCH, flavor="seurat")

    hvg_mask = batch1_shared.var["highly_variable"].to_numpy() | batch2_shared.var["highly_variable"].to_numpy()
    hvg_genes = shared_genes[hvg_mask]

    combined = ad.concat([batch1_shared, batch2_shared], join="inner", merge="same")
    combined.layers["counts"] = ad.concat([batch1_counts, batch2_counts], join="inner", merge="same").X.copy()
    combined.var["highly_variable"] = combined.var_names.isin(hvg_genes)
    add_qc_metrics(combined)

    embed = combined[:, combined.var["highly_variable"]].copy()
    sc.pp.scale(embed, max_value=10)
    sc.tl.pca(embed, svd_solver="arpack")
    sc.pp.neighbors(embed, n_neighbors=15, n_pcs=30)
    sc.tl.umap(embed, min_dist=UMAP_MIN_DIST)

    combined.obsm["X_pca"] = embed.obsm["X_pca"]
    combined.obsm["X_umap"] = embed.obsm["X_umap"]
    combined.uns["pca"] = embed.uns["pca"]
    combined.uns["neighbors"] = embed.uns["neighbors"]
    combined.uns["umap"] = embed.uns["umap"]
    combined.uns["ms2_pipeline"] = {
        "min_genes": MIN_GENES,
        "min_cells": MIN_CELLS,
        "shared_genes": int(len(shared_genes)),
        "n_hvg": int(combined.var["highly_variable"].sum()),
        "normalization": "per-batch library-size normalization to 1e4 followed by log1p",
        "feature_selection": f"union of the top {HVG_PER_BATCH} highly variable genes selected within each batch",
        "batch_correction": "Harmony via harmonypy (Python wrapper of Harmony)",
    }

    metrics_df = add_harmony_correction(combined)

    return combined, int(len(shared_genes)), metrics_df


def build_stage_counts(stage_summaries: dict[str, dict[str, int]], shared_genes: int) -> pd.DataFrame:
    rows = []
    for batch_label, summary in stage_summaries.items():
        rows.extend(
            [
                {
                    "batch_label": batch_label,
                    "stage": "raw",
                    "cells": summary["raw_cells"],
                    "genes": summary["raw_genes"],
                },
                {
                    "batch_label": batch_label,
                    "stage": "qc_filtered",
                    "cells": summary["qc_cells"],
                    "genes": summary["qc_genes"],
                },
                {
                    "batch_label": batch_label,
                    "stage": "metadata_singlets",
                    "cells": summary["metadata_singlet_cells"],
                    "genes": summary["metadata_singlet_genes"],
                },
                {
                    "batch_label": batch_label,
                    "stage": "strict_qc_final",
                    "cells": summary["final_cells"],
                    "genes": summary["final_genes"],
                },
            ]
        )

    stage_counts = pd.DataFrame(rows)
    stage_counts["shared_genes_after_qc_singlets"] = shared_genes
    return stage_counts


def write_dataset_summary(
    batch1_path: Path,
    batch2_path: Path,
    combined_path: Path,
    stage_counts: pd.DataFrame,
    shared_genes: int,
    metrics_df: pd.DataFrame,
) -> None:
    batch1_final = stage_counts.query("batch_label == 'batch1' and stage == 'strict_qc_final'").iloc[0]
    batch2_final = stage_counts.query("batch_label == 'batch2' and stage == 'strict_qc_final'").iloc[0]

    metric_lookup = metrics_df.set_index(["embedding", "metric"])["value"]
    before_batch = metric_lookup[("before_harmony", "silhouette_batch")]
    after_batch = metric_lookup[("after_harmony", "silhouette_batch")]

    summary = f"""# Dataset Summary

Generated for milestone 2 using the GSE96583 GEO release.

## Processed Outputs

- `{batch1_path.name}`: {int(batch1_final['cells'])} cells x {int(batch1_final['genes'])} genes
- `{batch2_path.name}`: {int(batch2_final['cells'])} cells x {int(batch2_final['genes'])} genes
- `{combined_path.name}`: {int(batch1_final['cells'] + batch2_final['cells'])} cells x {shared_genes} shared genes

## Wrangling Notes

- `batch1` is assembled from samples `A`, `B`, and `C`.
- `batch2` is assembled from `ctrl` and `stim` matrices.
- QC keeps cells with at least {MIN_GENES} detected genes and genes seen in at least {MIN_CELLS} cells.
- Metadata singlet filtering keeps only `multiplets == "singlet"`.
- Strict QC then removes residual doublet-like cells using Scrublet plus conservative high-count / high-gene tail filtering.
- Cross-batch EDA is performed on the {shared_genes} genes shared by both post-QC singlet batches.
- Harmony correction is stored as `X_pca_harmony` and `X_umap_harmony` in the combined AnnData object.
- Batch silhouette changes from {before_batch:.4f} before Harmony to {after_batch:.4f} after Harmony.
"""
    (PROCESSED_ROOT / "DATASET_SUMMARY.md").write_text(summary, encoding="utf-8")


def prepare_gse96583_dataset(force: bool = False) -> dict[str, Path | int]:
    ensure_gse96583_raw_data()

    batch1_path = PROCESSED_ROOT / "GSE96583_batch1_qc_annotated_singlets.h5ad"
    batch2_path = PROCESSED_ROOT / "GSE96583_batch2_qc_annotated_singlets.h5ad"
    combined_path = PROCESSED_ROOT / "GSE96583_combined_shared_qc_singlets.h5ad"
    stage_counts_path = PROCESSED_ROOT / "GSE96583_wrangling_stage_counts.csv"
    metrics_path = PROCESSED_ROOT / "GSE96583_batch_effect_metrics.csv"

    if not force and all(path.exists() for path in [batch1_path, batch2_path, combined_path, stage_counts_path, metrics_path]):
        stage_counts = pd.read_csv(stage_counts_path)
        shared_genes = int(stage_counts["shared_genes_after_qc_singlets"].iloc[0])
        return {
            "raw_dir": RAW_ROOT,
            "processed_dir": PROCESSED_ROOT,
            "batch1_path": batch1_path,
            "batch2_path": batch2_path,
            "combined_path": combined_path,
            "stage_counts_path": stage_counts_path,
            "metrics_path": metrics_path,
            "shared_genes": shared_genes,
        }

    batch1 = annotate_batch1(load_subset(BATCH1_FILES, "GSE96583_batch1.genes.tsv.gz", "batch1"))
    batch2 = annotate_batch2(load_subset(BATCH2_FILES, "GSE96583_batch2.genes.tsv.gz", "batch2"))

    batch1, batch1_summary = preprocess_batch(batch1)
    batch2, batch2_summary = preprocess_batch(batch2)
    combined, shared_genes, metrics_df = build_combined_shared_dataset(batch1, batch2)

    batch1.write_h5ad(batch1_path)
    batch2.write_h5ad(batch2_path)
    combined.write_h5ad(combined_path)

    stage_counts = build_stage_counts(
        {
            "batch1": batch1_summary,
            "batch2": batch2_summary,
        },
        shared_genes=shared_genes,
    )
    stage_counts.to_csv(stage_counts_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    write_dataset_summary(batch1_path, batch2_path, combined_path, stage_counts, shared_genes, metrics_df)

    return {
        "raw_dir": RAW_ROOT,
        "processed_dir": PROCESSED_ROOT,
        "batch1_path": batch1_path,
        "batch2_path": batch2_path,
        "combined_path": combined_path,
        "stage_counts_path": stage_counts_path,
        "metrics_path": metrics_path,
        "shared_genes": shared_genes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild processed outputs even if they exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = prepare_gse96583_dataset(force=args.force)
    print("Prepared GSE96583 milestone 2 artifacts:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
