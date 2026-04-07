#!/usr/bin/env python3
"""Attach GEO-provided metadata to processed GSE96583 AnnData files."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd


BATCH1_SAMPLE_TO_CODE = {
    'GSM2560245_A': 'A',
    'GSM2560246_B': 'B',
    'GSM2560247_C': 'C',
}

BATCH2_SAMPLE_TO_STIM = {
    'GSM2560248_2.1': 'ctrl',
    'GSM2560249_2.2': 'stim',
}


NUMERIC_COLUMNS = {'tsne1', 'tsne2', 'cluster', 'ind'}


def load_batch1_metadata(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / 'GSE96583_batch1.total.tsne.df.tsv.gz', sep='\t', index_col=0)
    df.index = df.index.astype(str)
    return df


def load_batch2_metadata(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / 'GSE96583_batch2.total.tsne.df.tsv.gz', sep='\t', index_col=0)
    df.index = df.index.astype(str)
    return df


def finalize_obs_types(obs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in NUMERIC_COLUMNS:
            obs[column] = pd.to_numeric(obs[column], errors='coerce')
        else:
            obs[column] = obs[column].astype(object)
    return obs


def annotate_batch1(adata: ad.AnnData, meta: pd.DataFrame) -> ad.AnnData:
    adata = adata.copy()
    barcodes = adata.obs_names.to_series().str.split(':', n=1).str[1]
    sample_codes = adata.obs['sample'].map(BATCH1_SAMPLE_TO_CODE)
    for column in meta.columns:
        adata.obs[column] = pd.NA
    for code in sorted(sample_codes.dropna().unique()):
        mask = sample_codes == code
        subset = meta[meta['batch'] == code]
        joined = subset.reindex(barcodes[mask])
        for column in meta.columns:
            adata.obs.loc[mask, column] = joined[column].values
    adata.obs = finalize_obs_types(adata.obs, list(meta.columns))
    return adata


def annotate_batch2(adata: ad.AnnData, meta: pd.DataFrame) -> ad.AnnData:
    adata = adata.copy()
    barcodes = adata.obs_names.to_series().str.split(':', n=1).str[1]
    stim_codes = adata.obs['sample'].map(BATCH2_SAMPLE_TO_STIM)
    for column in meta.columns:
        adata.obs[column] = pd.NA
    for stim in sorted(stim_codes.dropna().unique()):
        mask = stim_codes == stim
        subset = meta[meta['stim'] == stim]
        joined = subset.reindex(barcodes[mask])
        for column in meta.columns:
            adata.obs.loc[mask, column] = joined[column].values
    adata.obs = finalize_obs_types(adata.obs, list(meta.columns))
    return adata


def main() -> None:
    project_root = Path('/home/zichende/scRNA-ssl-uncertainty')
    raw_root = project_root / 'data' / 'raw' / 'GSE96583'
    processed_root = project_root / 'data' / 'processed'

    batch1_path = processed_root / 'GSE96583_batch1_qc_shared.h5ad'
    batch2_path = processed_root / 'GSE96583_batch2_qc_shared.h5ad'

    batch1 = ad.read_h5ad(batch1_path)
    batch2 = ad.read_h5ad(batch2_path)

    batch1 = annotate_batch1(batch1, load_batch1_metadata(raw_root))
    batch2 = annotate_batch2(batch2, load_batch2_metadata(raw_root))

    batch1.write_h5ad(processed_root / 'GSE96583_batch1_qc_shared_annotated.h5ad')
    batch2.write_h5ad(processed_root / 'GSE96583_batch2_qc_shared_annotated.h5ad')

    print('Saved annotated files:')
    print(processed_root / 'GSE96583_batch1_qc_shared_annotated.h5ad')
    print(processed_root / 'GSE96583_batch2_qc_shared_annotated.h5ad')


if __name__ == '__main__':
    main()
