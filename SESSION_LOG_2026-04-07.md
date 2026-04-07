# Session Log — 2026-04-07

## Completed

- Reframed the project around two no-login GEO datasets: `GSE115189` and `GSE96583`.
- Wrote the proposal draft in `PROPOSAL.md`.
- Wrote the execution plan in `PROJECT_PLAN.md`.
- Added `DATA_SOURCES.md`, `README.md`, `requirements.txt`, and `environment.yml`.
- Added `scripts/download_geo_data.sh` for direct GEO downloads.
- Added `scripts/prepare_pbmc_data.py` for first-pass dataset harmonization.
- Created a dedicated Python 3.11 environment at `.conda-env`.
- Installed the core scientific Python stack and `scanpy`.
- Downloaded both GEO archives into `data/raw`.
- Extracted both GEO archives for inspection.
- Downloaded additional `GSE96583` gene and metadata files from GEO.
- Confirmed that `GSE96583` must be treated as two batch-specific inputs with different gene spaces.
- Ran first-pass preprocessing and generated processed `.h5ad` files.
- Added GEO-provided metadata back onto processed `GSE96583` files.
- Created singlet-only labeled `GSE96583` benchmark files.
- Ran the first `PCA + logistic regression` cross-batch baseline on `GSE96583`.
- Ran label transfer from `GSE96583` to `GSE115189` and attached consensus pseudo-labels.

## Current Data Status

### GSE115189

- Archive extracted cleanly.
- Main file: `GSM3169075_filtered_gene_bc_matrices_h5.h5`
- Processed outputs:
  - `GSE115189_shared_with_GSE96583_batch1.h5ad`: 3364 cells x 12104 genes
  - `GSE115189_shared_with_GSE96583_batch2.h5ad`: 3364 cells x 12366 genes
  - `GSE115189_shared_with_GSE96583_batch1_annotated.h5ad`: pseudo-labels attached
  - `GSE115189_shared_with_GSE96583_batch2_annotated.h5ad`: pseudo-labels attached
- Pseudo-label transfer summary:
  - agreement rate between the two transfer models: `0.955707`
  - mean confidence from batch1 model: `0.953794`
  - mean confidence from batch2 model: `0.945611`
  - mean consensus confidence: `0.949702`
- Consensus pseudo-label counts:
  - `CD4 T cells`: 1622
  - `CD14+ Monocytes`: 434
  - `B cells`: 428
  - `CD8 T cells`: 402
  - `NK cells`: 395
  - `Megakaryocytes`: 36
  - `FCGR3A+ Monocytes`: 31
  - `Dendritic cells`: 16
- Important caveat: these are pseudo-labels derived from transfer, not ground truth.

### GSE96583

- Archive extracted cleanly.
- Raw package contains five per-sample sparse matrices:
  - `GSM2560245_A.mat.gz`
  - `GSM2560246_B.mat.gz`
  - `GSM2560247_C.mat.gz`
  - `GSM2560248_2.1.mtx.gz`
  - `GSM2560249_2.2.mtx.gz`
- Matching barcode files are present.
- Additional GEO supplementary files downloaded:
  - `GSE96583_batch1.genes.tsv.gz`
  - `GSE96583_batch2.genes.tsv.gz`
  - `GSE96583_batch1.total.tsne.df.tsv.gz`
  - `GSE96583_batch2.total.tsne.df.tsv.gz`
  - `GSE96583_genes.txt.gz`
- Confirmed mapping:
  - batch1 gene table matches `A/B/C`
  - batch2 gene table matches `2.1/2.2`
  - `GSM2560248_2.1` corresponds to `ctrl`
  - `GSM2560249_2.2` corresponds to `stim`
- Processed outputs:
  - `GSE96583_batch1_qc_shared.h5ad`: 13385 cells x 12104 genes
  - `GSE96583_batch2_qc_shared.h5ad`: 28871 cells x 12366 genes
  - `GSE96583_batch1_qc_shared_annotated.h5ad`: labels attached
  - `GSE96583_batch2_qc_shared_annotated.h5ad`: labels attached
  - `GSE96583_batch1_qc_shared_annotated_singlets.h5ad`: 11432 singlets
  - `GSE96583_batch2_qc_shared_annotated_singlets.h5ad`: 24250 singlets

## Current Results

### GSE96583 Cross-Batch PCA Baseline

Shared labels:
- `B cells`
- `CD14+ Monocytes`
- `CD4 T cells`
- `CD8 T cells`
- `Dendritic cells`
- `FCGR3A+ Monocytes`
- `Megakaryocytes`
- `NK cells`

Metrics:
- `batch1_to_batch2`: accuracy `0.918969`, macro-F1 `0.823378`, weighted-F1 `0.917365`
- `batch2_to_batch1`: accuracy `0.917337`, macro-F1 `0.875208`, weighted-F1 `0.913110`

Notable weaknesses in the detailed report:
- `Dendritic cells`, `Megakaryocytes`, and `FCGR3A+ Monocytes` are harder to transfer.
- `CD8 T cells` also drop more than the dominant major classes.

### GSE115189 Label Transfer

Saved result files:
- `results/gse115189_label_transfer_from_batch1.csv`
- `results/gse115189_label_transfer_from_batch2.csv`
- `results/gse115189_label_transfer_consensus.csv`
- `results/gse115189_label_transfer_summary.txt`

Interpretation:
- The two independently trained transfer models agree on most cells.
- This is strong enough to support a first cross-dataset benchmark, with the explicit caveat that `GSE115189` labels are pseudo-labels.

## Next Actions

1. Build the denoising autoencoder baseline on the same benchmark.
2. Add uncertainty estimation on top of the classifier.
3. Compare PCA against SSL embeddings under the same cross-batch transfer setting.
4. Test transfer from labeled `GSE96583` into pseudo-labeled `GSE115189`.
5. Later, replace pseudo-labels with a stronger external annotation workflow if needed.

## Notes

- The project should be framed as robustness under dataset shift, not only batch correction.
- `GSE96583` is now the primary benchmark and should be the basis of the proposal and first report.
- `GSE115189` is usable only under a pseudo-label assumption and should be treated as an optional side experiment.
