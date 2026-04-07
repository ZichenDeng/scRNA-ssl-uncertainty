#!/usr/bin/env python3
"""Prepare PBMC scRNA-seq datasets for cross-dataset experiments.

Current assumptions are based on the downloaded GEO files:
- GSE115189 is a single 10x H5 matrix.
- GSE96583 contains two batches with different gene spaces.
  - batch1: GSM2560245_A, GSM2560246_B, GSM2560247_C
  - batch2: GSM2560248_2.1, GSM2560249_2.2

The script builds one AnnData per dataset, using shared genes across
GSE115189 and each GSE96583 batch-specific aggregate. The outputs are meant
for initial exploration and baseline development, not final annotation.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy import io
from scipy import sparse


BATCH1_FILES = ["GSM2560245_A", "GSM2560246_B", "GSM2560247_C"]
BATCH2_FILES = ["GSM2560248_2.1", "GSM2560249_2.2"]


def read_gene_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["ensembl_id", "gene_symbol"])


def read_barcodes(path: Path) -> pd.Index:
    return pd.read_csv(path, sep="\t", header=None).iloc[:, 0].astype(str)


def read_matrix_market(path: Path) -> sparse.spmatrix:
    with gzip.open(path, "rt") as handle:
        return io.mmread(handle).tocsr()


def load_gse115189(root: Path) -> ad.AnnData:
    h5_path = next(root.glob("*.h5"))
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    adata.obs["dataset"] = "GSE115189"
    return adata


def load_gse96583_subset(root: Path, file_stems: list[str], genes_file: str, batch_name: str) -> ad.AnnData:
    genes = read_gene_table(root / genes_file)
    adatas: list[ad.AnnData] = []

    for stem in file_stems:
        matrix_path = None
        for suffix in (".mat.gz", ".mtx.gz"):
            candidate = root / f"{stem}{suffix}"
            if candidate.exists():
                matrix_path = candidate
                break
        if matrix_path is None:
            raise FileNotFoundError(f"No matrix found for {stem}")

        barcodes_path = root / f"{stem.split('_')[0]}_barcodes.tsv.gz"
        matrix = read_matrix_market(matrix_path).T
        barcodes = read_barcodes(barcodes_path)

        if matrix.shape[0] != len(barcodes):
            raise ValueError(f"Barcode count mismatch for {stem}: {matrix.shape[0]} vs {len(barcodes)}")
        if matrix.shape[1] != len(genes):
            raise ValueError(f"Gene count mismatch for {stem}: {matrix.shape[1]} vs {len(genes)}")

        obs_names = pd.Index([f"{stem}:{barcode}" for barcode in barcodes], dtype=str)
        adata = ad.AnnData(X=matrix)
        adata.obs_names = obs_names
        adata.var_names = genes["gene_symbol"].astype(str)
        adata.var["ensembl_id"] = genes["ensembl_id"].astype(str).values
        adata.var_names_make_unique()
        adata.obs["sample"] = stem
        adata.obs["batch"] = batch_name
        adata.obs["dataset"] = "GSE96583"
        adatas.append(adata)

    return ad.concat(adatas, join="inner", merge="same")


def basic_qc(adata: ad.AnnData, min_genes: int, min_cells: int) -> ad.AnnData:
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def subset_to_shared_genes(reference: ad.AnnData, target: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    shared = reference.var_names.intersection(target.var_names)
    if len(shared) == 0:
        raise ValueError("No shared genes found between datasets")
    return reference[:, shared].copy(), target[:, shared].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gse115189_root = args.raw_dir / "GSE115189"
    gse96583_root = args.raw_dir / "GSE96583"

    gse115189 = basic_qc(load_gse115189(gse115189_root), args.min_genes, args.min_cells)
    gse96583_batch1 = basic_qc(
        load_gse96583_subset(gse96583_root, BATCH1_FILES, "GSE96583_batch1.genes.tsv.gz", "batch1"),
        args.min_genes,
        args.min_cells,
    )
    gse96583_batch2 = basic_qc(
        load_gse96583_subset(gse96583_root, BATCH2_FILES, "GSE96583_batch2.genes.tsv.gz", "batch2"),
        args.min_genes,
        args.min_cells,
    )

    ref_b1, ext_b1 = subset_to_shared_genes(gse115189, gse96583_batch1)
    ref_b2, ext_b2 = subset_to_shared_genes(gse115189, gse96583_batch2)

    ref_b1.write_h5ad(args.out_dir / "GSE115189_shared_with_GSE96583_batch1.h5ad")
    ext_b1.write_h5ad(args.out_dir / "GSE96583_batch1_qc_shared.h5ad")
    ref_b2.write_h5ad(args.out_dir / "GSE115189_shared_with_GSE96583_batch2.h5ad")
    ext_b2.write_h5ad(args.out_dir / "GSE96583_batch2_qc_shared.h5ad")

    print("Saved processed datasets:")
    for name in [
        "GSE115189_shared_with_GSE96583_batch1.h5ad",
        "GSE96583_batch1_qc_shared.h5ad",
        "GSE115189_shared_with_GSE96583_batch2.h5ad",
        "GSE96583_batch2_qc_shared.h5ad",
    ]:
        print(args.out_dir / name)


if __name__ == "__main__":
    main()
