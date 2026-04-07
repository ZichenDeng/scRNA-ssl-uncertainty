# Project Plan

## Objective

Build a reproducible benchmark for testing whether self-supervised embeddings improve PBMC cell-type prediction and uncertainty estimation under batch and condition shift in `GSE96583` relative to PCA.

## Concrete Deliverables

- A reproducible data acquisition note for `GSE96583`
- A preprocessing pipeline for harmonizing the `GSE96583` batches
- A PCA baseline for representation learning and classification
- A denoising autoencoder baseline with Monte Carlo dropout
- Cross-batch and cross-condition evaluation tables and uncertainty metrics
- A final report with figures and interpretation

## Immediate Next Steps

1. Finalize the `GSE96583` benchmark splits and label taxonomy.
2. Keep the first-pass preprocessing pipeline as the canonical data loader.
3. Establish the PCA baseline for cross-batch and cross-condition transfer.
4. Add the denoising autoencoder and uncertainty evaluation.
5. Use `GSE115189` only as an optional side experiment, not as the core benchmark.

## Proposed Experiment Structure

### Dataset Preparation

- Download supplementary files directly from GEO
- Load batch-specific matrices, features, and barcodes into `AnnData`
- Apply light quality control and normalization
- Restrict to shared genes for each benchmark split
- Filter to singlets and harmonize labels into major immune cell classes

### Modeling

- Baseline 1: PCA plus logistic regression
- Baseline 2: PCA plus MLP
- Main model: denoising autoencoder plus classifier head
- Uncertainty: Monte Carlo dropout at inference time

### Evaluation

- Cross-batch transfer from `batch1` to `batch2`
- Reverse-direction transfer from `batch2` to `batch1`
- Cross-condition transfer inside `batch2` from `ctrl` to `stim` and reverse if feasible
- Macro-F1, accuracy, Brier score, ECE, and AUROC for error detection

## Notes on Scope Control

- Keep the problem limited to `GSE96583`
- Use PCA as the required baseline
- Treat advanced generative single-cell models as optional
- Collapse labels if fine-grained annotation becomes unstable
