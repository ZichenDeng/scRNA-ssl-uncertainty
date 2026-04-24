# Dataset Summary

Generated for milestone 2 using the GSE96583 GEO release.

## Processed Outputs

- `GSE96583_batch1_qc_annotated_singlets.h5ad`: 11308 cells x 14794 genes
- `GSE96583_batch2_qc_annotated_singlets.h5ad`: 23906 cells x 16142 genes
- `GSE96583_combined_shared_qc_singlets.h5ad`: 35214 cells x 14222 shared genes

## Wrangling Notes

- `batch1` is assembled from samples `A`, `B`, and `C`.
- `batch2` is assembled from `ctrl` and `stim` matrices.
- QC keeps cells with at least 200 detected genes and genes seen in at least 3 cells.
- Metadata singlet filtering keeps only `multiplets == "singlet"`.
- Strict QC then removes residual doublet-like cells using Scrublet plus conservative high-count / high-gene tail filtering.
- Cross-batch EDA is performed on the 14222 genes shared by both post-QC singlet batches.
- Harmony correction is stored as `X_pca_harmony` and `X_umap_harmony` in the combined AnnData object.
- Batch silhouette changes from 0.0976 before Harmony to -0.0163 after Harmony.
