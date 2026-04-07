# Dataset Summary

Generated on 2026-04-07 after preprocessing, metadata attachment, and initial label transfer.

## Processed Outputs

- `GSE115189_shared_with_GSE96583_batch1.h5ad`: 3364 cells x 12104 genes
- `GSE115189_shared_with_GSE96583_batch2.h5ad`: 3364 cells x 12366 genes
- `GSE115189_shared_with_GSE96583_batch1_annotated.h5ad`: pseudo-labels attached
- `GSE115189_shared_with_GSE96583_batch2_annotated.h5ad`: pseudo-labels attached
- `GSE96583_batch1_qc_shared.h5ad`: 13385 cells x 12104 genes
- `GSE96583_batch2_qc_shared.h5ad`: 28871 cells x 12366 genes
- `GSE96583_batch1_qc_shared_annotated.h5ad`: GEO metadata attached
- `GSE96583_batch2_qc_shared_annotated.h5ad`: GEO metadata attached
- `GSE96583_batch1_qc_shared_annotated_singlets.h5ad`: 11432 singlets x 12104 genes
- `GSE96583_batch2_qc_shared_annotated_singlets.h5ad`: 24250 singlets x 12366 genes

## Optional Side Dataset: GSE115189 Pseudo-Label Transfer

Reference models trained on labeled `GSE96583` singlets were used to predict labels for `GSE115189`.

- Agreement between batch1- and batch2-trained transfer models: `0.9557`
- Mean confidence from batch1-trained model: `0.9538`
- Mean confidence from batch2-trained model: `0.9456`
- Mean consensus confidence: `0.9497`

Consensus pseudo-label counts:

- `CD4 T cells`: 1622
- `CD14+ Monocytes`: 434
- `B cells`: 428
- `CD8 T cells`: 402
- `NK cells`: 395
- `Megakaryocytes`: 36
- `FCGR3A+ Monocytes`: 31
- `Dendritic cells`: 16

## Notes

- `GSE96583` must be handled as two distinct batches with different gene spaces.
- Shared gene counts are lower than raw gene counts because the first pass aligns by gene symbols after QC.
- `GSE96583` is the primary benchmark.
- `GSE115189` labels are pseudo-labels derived from `GSE96583` transfer, not ground-truth annotations.
