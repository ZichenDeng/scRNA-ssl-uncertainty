# MS2 Sections 1-2 Slides

## Slide 1: Why GSE96583?
- We rescoped the project to one benchmark dataset so that the milestone is about defensible data wrangling, not about uncertain cross-dataset labels.
- GSE96583 is public on GEO, directly downloadable, and already contains usable PBMC cell-type metadata.
- It also contains meaningful shift structure: batch1 versus batch2, plus control versus stimulation inside batch2.
- This gives us a realistic benchmark without adding label-accessibility risk from a second dataset.

## Slide 2: Raw Data Structure
- The GEO download is not a single tidy table. It is fragmented across compressed count matrices, gene files, and metadata tables.
- Batch1 is assembled from three source matrices: GSM2560245_A, GSM2560246_B, and GSM2560247_C.
- Batch2 is assembled from GSM2560248_2.1 and GSM2560249_2.2, which also support control-versus-stimulation comparisons.
- The raw structure itself motivates wrangling: before modeling, we need consistent genes, aligned metadata, and one clean cell-by-gene object per benchmark split.

## Slide 3: Wrangling Pipeline
- Step 1: download GEO count matrices and metadata and inspect file layout.
- Step 2: standardize matrices and attach cell-level annotations.
- Step 3: restrict analyses to shared genes so that cross-batch comparisons are valid.
- Step 4: keep QC-passed singlets only, which removes likely doublets that would distort cell-type classification.
- Step 5: save processed AnnData objects that can be reused by EDA, baselines, and later SSL models.

## Slide 4: Wrangling Outcomes
- After QC and singlet filtering, batch1 contains 11,432 cells and batch2 contains 24,250 cells.
- The processed data preserve major immune cell types needed for downstream classification.
- The class distribution is imbalanced, with broad T-cell and monocyte populations dominating while megakaryocytes and other rare types are scarce.
- This matters for later modeling because macro-F1 and per-class behavior will be more informative than accuracy alone.
