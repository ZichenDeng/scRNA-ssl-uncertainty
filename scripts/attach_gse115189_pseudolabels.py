#!/usr/bin/env python3
"""Attach consensus pseudo-labels to processed GSE115189 AnnData files."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd


FILES = [
    'GSE115189_shared_with_GSE96583_batch1.h5ad',
    'GSE115189_shared_with_GSE96583_batch2.h5ad',
]


def main() -> None:
    root = Path('/home/zichende/scRNA-ssl-uncertainty')
    processed_root = root / 'data' / 'processed'
    results_root = root / 'results'
    labels = pd.read_csv(results_root / 'gse115189_label_transfer_consensus.csv').set_index('cell_id')

    for name in FILES:
        adata = ad.read_h5ad(processed_root / name)
        joined = labels.reindex(adata.obs_names)
        for col in joined.columns:
            adata.obs[col] = joined[col].values
        out_name = name.replace('.h5ad', '_annotated.h5ad')
        adata.write_h5ad(processed_root / out_name)
        print(out_name, adata.shape, int(adata.obs['consensus_label'].notna().sum()))


if __name__ == '__main__':
    main()
