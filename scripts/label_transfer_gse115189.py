#!/usr/bin/env python3
"""Transfer cell-type labels from labeled GSE96583 singlets to GSE115189."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SRC_FILES = {
    'batch1': 'GSE96583_batch1_qc_shared_annotated_singlets.h5ad',
    'batch2': 'GSE96583_batch2_qc_shared_annotated_singlets.h5ad',
}
SRC_LABEL_COL = {
    'batch1': 'cell.type',
    'batch2': 'cell',
}
TGT_FILES = {
    'batch1': 'GSE115189_shared_with_GSE96583_batch1.h5ad',
    'batch2': 'GSE115189_shared_with_GSE96583_batch2.h5ad',
}


def to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def fit_transfer_model(src: ad.AnnData, label_col: str) -> Pipeline:
    x = to_dense(src.X)
    y = src.obs[label_col].astype(str).to_numpy()
    model = Pipeline([
        ('scale', StandardScaler(with_mean=False)),
        ('pca', PCA(n_components=50, random_state=0)),
        ('lr', LogisticRegression(max_iter=2000)),
    ])
    model.fit(x, y)
    return model


def transfer_one(processed_root: Path, tag: str) -> pd.DataFrame:
    src = ad.read_h5ad(processed_root / SRC_FILES[tag])
    tgt = ad.read_h5ad(processed_root / TGT_FILES[tag])
    model = fit_transfer_model(src, SRC_LABEL_COL[tag])

    probs = model.predict_proba(to_dense(tgt.X))
    labels = model.classes_
    pred_idx = probs.argmax(axis=1)

    out = pd.DataFrame({
        'cell_id': tgt.obs_names.astype(str),
        'source_model': tag,
        'predicted_label': labels[pred_idx],
        'predicted_confidence': probs.max(axis=1),
    })
    for i, label in enumerate(labels):
        out[f'prob__{label}'] = probs[:, i]
    return out


def main() -> None:
    root = Path('/home/zichende/scRNA-ssl-uncertainty')
    processed_root = root / 'data' / 'processed'
    results_root = root / 'results'
    results_root.mkdir(parents=True, exist_ok=True)

    batch1 = transfer_one(processed_root, 'batch1')
    batch2 = transfer_one(processed_root, 'batch2')

    batch1.to_csv(results_root / 'gse115189_label_transfer_from_batch1.csv', index=False)
    batch2.to_csv(results_root / 'gse115189_label_transfer_from_batch2.csv', index=False)

    merged = batch1[['cell_id', 'predicted_label', 'predicted_confidence']].merge(
        batch2[['cell_id', 'predicted_label', 'predicted_confidence']],
        on='cell_id',
        suffixes=('_from_batch1', '_from_batch2'),
    )
    merged['agrees'] = merged['predicted_label_from_batch1'] == merged['predicted_label_from_batch2']
    merged['mean_confidence'] = merged[['predicted_confidence_from_batch1', 'predicted_confidence_from_batch2']].mean(axis=1)
    merged['consensus_label'] = np.where(
        merged['agrees'],
        merged['predicted_label_from_batch1'],
        np.where(
            merged['predicted_confidence_from_batch1'] >= merged['predicted_confidence_from_batch2'],
            merged['predicted_label_from_batch1'],
            merged['predicted_label_from_batch2'],
        ),
    )
    merged.to_csv(results_root / 'gse115189_label_transfer_consensus.csv', index=False)

    summary = {
        'n_cells': len(merged),
        'agreement_rate': float(merged['agrees'].mean()),
        'mean_confidence_batch1': float(merged['predicted_confidence_from_batch1'].mean()),
        'mean_confidence_batch2': float(merged['predicted_confidence_from_batch2'].mean()),
        'mean_confidence_consensus': float(merged['mean_confidence'].mean()),
    }
    with open(results_root / 'gse115189_label_transfer_summary.txt', 'w') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')
        f.write('\nConsensus label counts:\n')
        for label, count in merged['consensus_label'].value_counts().to_dict().items():
            f.write(f'{label}: {count}\n')

    print(summary)
    print(merged['consensus_label'].value_counts().to_string())
    print('\nSaved:')
    print(results_root / 'gse115189_label_transfer_from_batch1.csv')
    print(results_root / 'gse115189_label_transfer_from_batch2.csv')
    print(results_root / 'gse115189_label_transfer_consensus.csv')
    print(results_root / 'gse115189_label_transfer_summary.txt')


if __name__ == '__main__':
    main()
