# Data Sources

## Primary Dataset

### GSE96583

- Title: Multiplexing droplet-based single cell RNA-sequencing using genetic barcodes
- GEO page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583
- Public status: Public on June 13, 2017
- GEO summary: PBMC scRNA-seq data with processed matrices, batch-level gene tables, and cell-level metadata files
- Supplementary file listed by GEO: `GSE96583_RAW.tar`
- File type shown by GEO: TAR of MAT, MTX, and TSV files
- Additional GEO supplementary files: batch-level gene tables and t-SNE metadata TSV files
- Programmatic download pattern: `https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96583&format=file`

## Why This Dataset

- Hosted on GEO/NCBI with no login required
- Exposes direct supplementary-file downloads suitable for scripted acquisition
- Contains multiple PBMC batches with different gene spaces
- Includes per-cell metadata with usable cell-type labels
- Includes singlet/doublet indicators and condition information for richer robustness evaluation

## Supporting Side Dataset

### GSE115189

- Title: Single cell profiling of peripheral blood mononuclear cells from healthy human donor
- GEO page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115189
- Public status: Public on July 28, 2018
- GEO summary: healthy human PBMCs profiled with 10x Genomics
- Supplementary file listed by GEO: `GSE115189_RAW.tar`
- File type shown by GEO: TAR of H5 files
- Current role: optional side experiment only
- Limitation: no directly usable public per-cell label file was found during setup

## Immediate Data Questions

- Whether the benchmark should prioritize `batch1` vs `batch2`, `ctrl` vs `stim`, or both
- Whether labels should be collapsed to major immune cell classes for the first benchmark
- Whether `GSE115189` should remain in the repo only as an optional transfer experiment
