# scRNA SSL Uncertainty

This project studies whether a simple self-supervised embedding model can improve PBMC cell-type prediction and uncertainty estimation under dataset shift relative to PCA.

## Current Scope

- Primary benchmark: `GSE96583`
- Main comparison: PCA versus denoising autoencoder
- Reliability analysis: uncertainty under batch and condition shift
- Immediate benchmark: cross-batch transfer inside `GSE96583`
- Optional side experiment: transfer into `GSE115189` using pseudo-labels

## Current Status

- Proposal drafted in `PROPOSAL.md`
- Project plan written in `PROJECT_PLAN.md`
- Python 3.11 project environment available at `.conda-env`
- GEO datasets downloaded and extracted in `data/raw`
- First-pass processed `.h5ad` files created in `data/processed`
- GEO metadata attached to `GSE96583` processed files
- Singlet-only labeled `GSE96583` benchmark files created
- First `PCA + logistic regression` cross-batch baseline completed
- `GSE115189` retained only as an optional pseudo-labeled side dataset

## Key Files

- `PROPOSAL.md`: proposal draft for the course project
- `PROJECT_PLAN.md`: working execution plan
- `DATA_SOURCES.md`: dataset details and download notes
- `SESSION_LOG_2026-04-07.md`: running project log
- `scripts/download_geo_data.sh`: direct GEO download helper
- `scripts/prepare_pbmc_data.py`: first-pass preprocessing and shared-gene alignment
- `scripts/annotate_gse96583_metadata.py`: attach GEO-provided labels to `GSE96583`
- `scripts/run_gse96583_pca_baseline.py`: first cross-batch baseline
- `data/processed/DATASET_SUMMARY.md`: current processed dataset summary
- `results/gse96583_pca_baseline_metrics.csv`: baseline metrics table
- `results/gse96583_pca_baseline_report.txt`: detailed classification report

## Current Results

Cross-batch `PCA + logistic regression` on labeled `GSE96583` singlets:

- `batch1 -> batch2`: accuracy `0.9190`, macro-F1 `0.8234`
- `batch2 -> batch1`: accuracy `0.9173`, macro-F1 `0.8752`

## Open Problems

- The self-supervised autoencoder and uncertainty estimation pipeline are not implemented yet.
- The benchmark still needs a clean cross-condition experiment inside `GSE96583`.
- `GSE115189` should not be treated as ground-truth-labeled data.
