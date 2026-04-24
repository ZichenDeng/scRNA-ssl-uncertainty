# SupDAE-32-head-w2.0-noise0.10 Slides

## Slide 1: Why this run exists

- Siheng's latest notebook push finishes data wrangling, PCA baselines, and the fixed `train / val / test` split.
- This run extends that handoff by testing `SupDAE-32-head-w2.0-noise0.10` against the same `PCA-50` baseline.
- The goal is not to over-claim. The goal is to learn whether a stronger DAE setup is moving in the right direction.

## Slide 2: Setup

- Input dataset source: `lite_fallback_cached`
- Input dataset path: `/home/zichende/projects/scRNA-ssl-uncertainty/data/processed/GSE96583_combined_shared_qc_singlets_lite.h5ad`
- Input dimension: `14222` genes
- Split sizes: train `24977`, val `5352`, test `5353`
- DAE hidden dimensions: `[512, 128]`
- Latent dimension: `32`
- Noise type / level: `dropout` / `0.1`
- Reconstruction loss: `MSE`
- Classifier mode: `supervised_head`
- Supervised loss weight: `2.0`
- Training epochs actually used: `6`

## Slide 3: Main result

- SupDAE-32-head-w2.0-noise0.10 improves test macro-F1 over PCA-50 by +0.018.
- Test accuracy delta: `+0.018`
- PCA-50 test metrics: accuracy `0.922`, macro-F1 `0.858`, balanced accuracy `0.901`
- SupDAE-32-head-w2.0-noise0.10 test metrics: accuracy `0.940`, macro-F1 `0.876`, balanced accuracy `0.832`
- Validation/test macro-F1 stability:
| representation | val | test | val_test_gap |
| --- | ---: | ---: | ---: |
| PCA-50 | 0.842 | 0.858 | 0.016 |
| SupDAE-32-head-w2.0-noise0.10 | 0.836 | 0.876 | 0.040 |


## Slide 4: Class-level story

- Biggest gains: FCGR3A+ Monocytes (+0.047); Megakaryocytes (+0.032); CD14+ Monocytes (+0.022)
- No cell type dropped below PCA; smallest gains: B cells (+0.001); NK cells (+0.003); CD8 T cells (+0.008)
- Use the matching per-class delta figure for this run.

## Slide 5: Training story

- Best validation total loss: `0.5616`
- Final restored checkpoint validation total loss: `0.7380`
- Use the matching training-curve figure for this run.

## Slide 6: Caveat / next step

- This still uses the notebook's fixed split, so it is a representation-learning comparison before the stricter cross-batch transfer test.
- If the dataset source is a `lite_fallback_*` variant, this run uses the raw GEO matrices plus metadata singlets, not the full heavy MS2 object.
- If this variant helps, the next step is to retest under the original cross-batch and cross-condition story.
- If it still loses to PCA, the next tuning knobs are latent width, corruption strength, classifier loss weight, and stronger supervision.
