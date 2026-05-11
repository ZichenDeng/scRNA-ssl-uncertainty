# MS4 Final Shift-First Summary

## Final model choice

- Best simple baseline: `PCA-50-LR`
- Final supervised AE for MS4: `SupDAE-32-head-w1-noise0.05`
- Main conclusion: the supervised AE is competitive on the harder `batch1 -> batch2` transfer, but `PCA-50-LR` remains the strongest overall model on mean macro-F1 across the two primary transfer directions.

## Primary batch-transfer results

| model | direction | accuracy | balanced_accuracy | macro_f1 |
| --- | --- | --- | --- | --- |
| PCA + LR | batch1_to_batch2 | 0.829 | 0.801 | 0.746 |
| PCA + LR | batch2_to_batch1 | 0.931 | 0.915 | 0.901 |
| PCA + MLP | batch1_to_batch2 | 0.772 | 0.689 | 0.713 |
| PCA + MLP | batch2_to_batch1 | 0.911 | 0.853 | 0.869 |
| Final SupDAE | batch1_to_batch2 | 0.900 | 0.750 | 0.750 |
| Final SupDAE | batch2_to_batch1 | 0.929 | 0.865 | 0.885 |

## Mean summary across primary directions

| representation | mean_macro_f1 | mean_balanced_accuracy | worst_macro_f1 |
| --- | --- | --- | --- |
| PCA-50-LR | 0.824 | 0.858 | 0.746 |
| PCA-50-MLP | 0.791 | 0.771 | 0.713 |
| SupDAE-32-head-w1-noise0.05 | 0.818 | 0.808 | 0.750 |

## Condition-transfer follow-up

| model | direction | accuracy | balanced_accuracy | macro_f1 |
| --- | --- | --- | --- | --- |
| PCA + LR | batch2_ctrl_to_stim | 0.893 | 0.847 | 0.818 |
| PCA + LR | batch2_stim_to_ctrl | 0.848 | 0.860 | 0.776 |
| PCA + MLP | batch2_ctrl_to_stim | 0.801 | 0.733 | 0.757 |
| PCA + MLP | batch2_stim_to_ctrl | 0.838 | 0.788 | 0.783 |
| Final SupDAE | batch2_ctrl_to_stim | 0.905 | 0.746 | 0.765 |
| Final SupDAE | batch2_stim_to_ctrl | 0.938 | 0.819 | 0.838 |

## Final-model uncertainty summary

| direction | test_accuracy | brier_score | ece | confidence_error_auroc |
| --- | --- | --- | --- | --- |
| batch1_to_batch2 | 0.899 | 0.149 | 0.009 | 0.872 |
| batch2_to_batch1 | 0.929 | 0.117 | 0.024 | 0.820 |

## Interpretation

- On `batch1 -> batch2`, the final SupDAE slightly improves macro-F1 over `PCA + LR` (0.750 vs 0.746).
- On `batch2 -> batch1`, `PCA + LR` remains stronger (0.901 vs 0.885).
- The AE therefore does not beat the best baseline overall, but it remains a meaningful representation-learning comparison and supports informative MC-dropout uncertainty analysis.
- Dataset imbalance is not solved; it is quantified, partially mitigated, and treated as a standing limitation.
