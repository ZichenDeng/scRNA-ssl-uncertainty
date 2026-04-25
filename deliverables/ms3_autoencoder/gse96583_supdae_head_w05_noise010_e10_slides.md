# SupDAE-32-head-w0.5-noise0.10 Slides

## Slide 1: Why this run exists

- Siheng's latest notebook push finishes data wrangling, PCA baselines, and the fixed `train / val / test` split.
- This run extends that handoff by testing `SupDAE-32-head-w0.5-noise0.10` against the same `PCA-50` baseline.
- The goal is not to over-claim. The goal is to learn whether a stronger DAE setup is moving in the right direction.

## Slide 2: Setup

- Input dataset source: `full_ms2_processed`
- Input dataset path: `/Users/taosiheng/Desktop/CS1090B/scRNA-ssl-uncertainty/data/processed/GSE96583_combined_shared_qc_singlets.h5ad`
- Input dimension: `14222` genes
- Split sizes: train `24649`, val `5282`, test `5283`
- DAE hidden dimensions: `[512, 128]`
- Latent dimension: `32`
- Noise type / level: `dropout` / `0.1`
- Reconstruction loss: `MSE`
- Classifier mode: `supervised_head`
- Supervised loss weight: `0.5`
- Training epochs actually used: `6`

## Slide 3: Main result

- SupDAE-32-head-w0.5-noise0.10 improves test macro-F1 over PCA-50 by +0.016.
- Test accuracy delta: `+0.010`
- PCA-50 test metrics: accuracy `0.922`, macro-F1 `0.850`, balanced accuracy `0.894`
- SupDAE-32-head-w0.5-noise0.10 test metrics: accuracy `0.932`, macro-F1 `0.866`, balanced accuracy `0.834`
- Validation/test macro-F1 stability:
| representation | val | test | val_test_gap |
| --- | ---: | ---: | ---: |
| PCA-50 | 0.850 | 0.850 | 0.000 |
| SupDAE-32-head-w0.5-noise0.10 | 0.870 | 0.866 | 0.004 |


## Slide 4: Class-level story

- Biggest gains: FCGR3A+ Monocytes (+0.048); Dendritic cells (+0.043); B cells (+0.015)
- Biggest drops: CD8 T cells (-0.009)
- Use the matching per-class delta figure for this run.

## Slide 5: Training story

- Best validation total loss: `0.2267`
- Final restored checkpoint validation total loss: `0.2709`
- Use the matching training-curve figure for this run.

## Slide 6: Caveat / next step

- This still uses the notebook's fixed split, so it is a representation-learning comparison before the stricter cross-batch transfer test.
- If the dataset source is a `lite_fallback_*` variant, this run uses the raw GEO matrices plus metadata singlets, not the full heavy MS2 object.
- If this variant helps, the next step is to retest under the original cross-batch and cross-condition story.
- If it still loses to PCA, the next tuning knobs are latent width, corruption strength, classifier loss weight, and stronger supervision.
