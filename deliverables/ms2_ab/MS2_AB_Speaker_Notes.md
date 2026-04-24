# MS2 Sections 1-2 Speaker Notes

## Slide 1: Data Source and Why GSE96583
- Say explicitly that we scoped MS2 to one dataset with defensible labels and real shift.
- Mention both the batch split and the condition split.

## Slide 2: GEO File Structure
- Explain that the archive alone is not enough because gene tables and metadata live outside the tarball.
- Point out that wrangling begins before modeling because the raw release is fragmented.

## Slide 3: Data Wrangling Pipeline
- Walk through download, metadata attachment, basic QC, metadata singlets, extra Scrublet filtering, and shared-gene alignment.
- Emphasize the final shared-gene count of 14222.

## Slide 4: Class Distribution and Rare Classes
- Stress that the benchmark is usable but imbalanced.
- Set up later evaluation choices such as macro-F1.

## Slide 5: Batch Effect Before Correction
- Explicitly say that the uncorrected embedding is too batch-driven.
- Use this slide to motivate Harmony rather than jumping straight to the fix.

## Slide 6: Harmony Batch Correction
- Mention that Harmony is a standard batch-correction tool and often appears in the R / Seurat workflow.
- The main result is that batch separation drops while cell-type structure is preserved.

## Slide 7: Composition and QC Context
- End by reminding the audience that this dataset still contains real condition structure and QC differences.
- That is why we treat Harmony as correction for technical batch effect, not as a way to erase all variation.
