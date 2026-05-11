# scRNA SSL Uncertainty

This project studies PBMC cell-type classification under dataset shift in `GSE96583`, with a final MS4 comparison between simple PCA baselines and a supervised denoising autoencoder.

## Canonical MS4 assets

Use these committed files as the source of truth for the final writeup and slides:

- `deliverables/ms2_ab/shift_pca_lr_metrics.csv`
- `deliverables/ms2_ab/shift_split_summary.csv`
- `deliverables/ms2_ab/ms4_primary_shift_summary.csv`
- `deliverables/ms2_ab/ms4_condition_shift_summary.csv`
- `deliverables/ms2_ab/ms4_final_ae_uncertainty_summary.csv`
- `deliverables/ms2_ab/ms4_model_selection_table.csv`
- `deliverables/ms2_ab/ms4_final_summary.md`
- `deliverables/ms2_ab/figures/ms4_shift_benchmark_by_direction.png`
- `deliverables/ms2_ab/figures/ms4_best_supdae_training_curves.png`
- `deliverables/ms2_ab/figures/ms4_best_supdae_vs_pca_per_class_delta.png`
- `deliverables/ms2_ab/figures/ms4_uncertainty_summary.png`
- `deliverables/ms4_final.ipynb`
- `deliverables/ms4_overleaf_person2_sections.tex`

## Final MS4 story

- Single benchmark dataset: `GSE96583`
- Primary evaluation: `batch1 -> batch2` and `batch2 -> batch1`
- Secondary follow-up: `batch2 ctrl -> stim` and `batch2 stim -> ctrl`
- Best simple baseline: `PCA-50-LR`
- Final supervised AE: `SupDAE-32-head-w1-noise0.05`
- Main conclusion: the supervised AE stays competitive on the harder `batch1 -> batch2` direction, but `PCA-50-LR` remains the strongest overall model on mean macro-F1 across the primary batch-transfer benchmark
- Main MS4 addition beyond the MS3 sweep: MC-dropout uncertainty on the final supervised AE

## Scope note

Random-split experiments, orchestration scripts, scratch `deliverables/final_project/` assets, and raw full-sweep result dumps can still exist in the working tree, but they are not the canonical final evidence for the course submission.
