# Self-Supervised Representation Learning and Uncertainty-Aware Cell Type Prediction under Batch and Condition Shift in PBMC scRNA-seq Data

## Motivation

Single-cell RNA sequencing (scRNA-seq) data often contain strong batch effects, technical noise, and condition-specific variation that can reduce the transferability of learned representations across samples. In addition, many prediction pipelines focus mainly on classification accuracy and do not evaluate whether model confidence remains reliable when the test distribution differs from the training distribution. This project studies whether a simple self-supervised representation learning approach can improve robustness and provide useful uncertainty estimates in a realistic PBMC classification setting.

## Research Questions

1. Can self-supervised embeddings outperform PCA for cell-type classification when training and test data come from different PBMC batches or stimulation conditions?
2. Are the learned embeddings less sensitive to technical variation than standard linear dimensionality reduction?
3. Can predictive uncertainty help identify misclassified or shifted cells?

## Data

This project will use one publicly accessible PBMC scRNA-seq study hosted on GEO/NCBI:

- GSE96583: Multiplexing droplet-based single cell RNA-sequencing using genetic barcodes  
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583

This study is sufficient for a focused robustness benchmark because it contains multiple PBMC batches with different gene spaces, as well as control and stimulated samples. The GEO supplementary metadata also provide usable cell-type annotations and singlet/doublet indicators. If needed, labels will be collapsed to a smaller shared taxonomy of major immune cell classes such as T cells, B cells, NK cells, monocytes, and dendritic cells.

## Methods

I will preprocess the data using a standard Scanpy workflow, including quality control, normalization, log transformation, and highly variable gene selection. Because the study contains multiple batches with different gene spaces, analysis will be restricted to shared genes for each evaluation setting.

For representation learning, I will train a denoising autoencoder to produce low-dimensional embeddings from gene expression profiles. PCA will serve as the required baseline. If time permits, I may include one stronger nonlinear baseline, but the main comparison will remain between PCA and the self-supervised encoder.

A lightweight downstream classifier will then be trained on top of each embedding. The main evaluation setting will emphasize distribution shift within the study: for example, training on one batch and testing on another, or training on control cells and testing on stimulated cells. Predictive uncertainty will be estimated using Monte Carlo dropout, and I will test whether uncertainty is higher for misclassified or shifted samples.

## Evaluation

Performance will be measured using classification metrics such as macro-F1 and accuracy under cross-batch and cross-condition evaluation. I will also assess uncertainty quality using calibration-oriented metrics such as Brier score or expected calibration error, and detection metrics such as AUROC for identifying misclassified or shifted cells. Finally, low-dimensional visualizations will be used to inspect whether learned embeddings preserve biological structure while reducing batch-specific separation.

## Scope

This project is designed as a focused evaluation study rather than a large-scale biological foundation model effort. The main contribution is a careful comparison of a simple self-supervised embedding method and a standard baseline within a clearly specified public PBMC benchmark, together with an analysis of predictive uncertainty under shift.

## Risks and Mitigation

The main risk is that shift may reflect both technical and biological differences rather than pure batch effects. I will therefore frame the task as robustness under batch and condition shift, rather than as batch correction alone. A second risk is class imbalance for rare cell types such as dendritic cells or megakaryocytes; this will be addressed with macro-F1 and class-wise reporting. Computational cost is manageable because the project uses moderate dataset sizes and lightweight models.
