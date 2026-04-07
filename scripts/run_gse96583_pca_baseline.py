#!/usr/bin/env python3
"""Run a simple PCA + logistic regression baseline across GSE96583 batches."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CELL_COL = {
    'batch1': 'cell.type',
    'batch2': 'cell',
}


def to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def load_and_align() -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    root = Path('/home/zichende/scRNA-ssl-uncertainty/data/processed')
    b1 = ad.read_h5ad(root / 'GSE96583_batch1_qc_shared_annotated_singlets.h5ad')
    b2 = ad.read_h5ad(root / 'GSE96583_batch2_qc_shared_annotated_singlets.h5ad')

    b1.obs['label'] = b1.obs[CELL_COL['batch1']].astype(str)
    b2.obs['label'] = b2.obs[CELL_COL['batch2']].astype(str)

    shared_genes = b1.var_names.intersection(b2.var_names)
    shared_labels = sorted(set(b1.obs['label']) & set(b2.obs['label']))

    b1 = b1[b1.obs['label'].isin(shared_labels), shared_genes].copy()
    b2 = b2[b2.obs['label'].isin(shared_labels), shared_genes].copy()
    return b1, b2, shared_labels


def train_and_eval(train: ad.AnnData, test: ad.AnnData, tag: str) -> dict:
    x_train = to_dense(train.X)
    x_test = to_dense(test.X)
    y_train = train.obs['label'].to_numpy()
    y_test = test.obs['label'].to_numpy()

    clf = Pipeline([
        ('scale', StandardScaler(with_mean=False)),
        ('pca', PCA(n_components=50, random_state=0)),
        ('lr', LogisticRegression(max_iter=2000)),
    ])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    return {
        'direction': tag,
        'n_train': int(train.n_obs),
        'n_test': int(test.n_obs),
        'n_genes': int(train.n_vars),
        'accuracy': float(accuracy_score(y_test, pred)),
        'macro_f1': float(f1_score(y_test, pred, average='macro')), 
        'weighted_f1': float(f1_score(y_test, pred, average='weighted')),
        'report': classification_report(y_test, pred, digits=4),
    }


def main() -> None:
    out_root = Path('/home/zichende/scRNA-ssl-uncertainty/results')
    out_root.mkdir(parents=True, exist_ok=True)

    b1, b2, shared_labels = load_and_align()
    results = [
        train_and_eval(b1, b2, 'batch1_to_batch2'),
        train_and_eval(b2, b1, 'batch2_to_batch1'),
    ]

    rows = [{k: v for k, v in r.items() if k != 'report'} for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'gse96583_pca_baseline_metrics.csv', index=False)

    with open(out_root / 'gse96583_pca_baseline_report.txt', 'w') as f:
        f.write('Shared labels:\n')
        for label in shared_labels:
            f.write(f'- {label}\n')
        f.write('\n')
        for r in results:
            f.write(f"## {r['direction']}\n")
            f.write(f"n_train={r['n_train']} n_test={r['n_test']} n_genes={r['n_genes']}\n")
            f.write(f"accuracy={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f} weighted_f1={r['weighted_f1']:.4f}\n\n")
            f.write(r['report'])
            f.write('\n')

    print(df.to_string(index=False))
    print('\nSaved:')
    print(out_root / 'gse96583_pca_baseline_metrics.csv')
    print(out_root / 'gse96583_pca_baseline_report.txt')


if __name__ == '__main__':
    main()
