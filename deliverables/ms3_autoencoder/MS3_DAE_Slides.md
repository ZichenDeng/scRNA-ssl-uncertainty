# MS3 Autoencoder Section Deck Plan

## Overall goal

This section should answer one clear question:

**After Siheng finished the data pipeline and PCA baseline, does an autoencoder-based representation help classification more than PCA on the same fixed split?**

Recommended length: `3-4 minutes`

Recommended deck size:
- `6` main slides
- `2` backup slides for Q&A

Recommended style:
- Reuse the same template / font / color system as Siheng's baseline slides.
- Keep `PCA` in neutral gray and `DAE` in one accent color such as teal or dark blue.
- One message per slide. Do not turn the slides into an experiment log.

---

## Slide 1: Section handoff

### Title
`From PCA Baseline to Autoencoder`

### What goes on the slide

- `Siheng's part:` data wrangling, EDA, fixed `train / val / test` split, PCA baseline
- `My part:` train DAE variants on the same split and compare them against `PCA-50`
- `Question:` can a learned latent space beat PCA for cell-type classification?

### Layout

- Two-column slide
- Left: `What was already finished`
- Right: `What I added`
- Bottom: one-line transition

### Bottom-line sentence

`I inherited the fixed split and PCA baseline, then tested whether autoencoder-based representations improve downstream classification.`

### Speaker note (CN)

这一页主要是接前面 Siheng 的内容。  
你可以说：前面的 notebook 已经把数据处理、固定 split、PCA baseline 都做好了，我负责的是在同一个 split 上继续做 autoencoder，并看它能不能比 PCA 更强。

---

## Slide 2: Model idea

### Title
`What I Tested`

### What goes on the slide

- `Unsupervised DAE`
  reconstruct corrupted input, then train logistic regression on latent features
- `Supervised DAE`
  reconstruct input and predict cell type jointly through a classifier head
- Total loss:
  `L = reconstruction loss + lambda * classification loss`
- Best setting so far:
  `latent = 32`, `noise = 0.10`, `supervised loss weight = 0.5`

### Layout

- Left panel: simple pipeline diagram
  `input -> noisy input -> encoder -> latent z -> decoder -> reconstruction`
- Right panel: add one extra branch from `latent z -> classifier head -> cell type`
- Put the loss formula under the supervised branch

### Visual emphasis

- Mark `unsupervised = representation only`
- Mark `supervised = representation + label guidance`

### Speaker note (CN)

这一页要讲清楚 unsupervised 和 supervised 的区别。  
最简单的说法是：unsupervised 只学怎么重建输入，supervised 在重建之外，还直接利用标签去优化 latent space，所以更适合分类任务。

---

## Slide 3: Experiment summary

### Title
`What Happened Across Variants`

### What goes on the slide

Use one compact table like this:

| Representation | Classifier | Test Accuracy | Test Macro-F1 | Test Balanced Acc. |
| --- | --- | ---: | ---: | ---: |
| `PCA-50` | Logistic Regression | `0.922` | `0.858` | `0.901` |
| `DAE-32` | Logistic Regression | `0.868` | `0.765` | `0.819` |
| `SupDAE-32` `w=1.0` | Supervised Head | `0.941` | `0.876` | `0.832` |
| `SupDAE-32` `w=0.5` | Supervised Head | `0.943` | `0.898` | `0.868` |
| `SupDAE-32` `w=2.0` | Supervised Head | `0.940` | `0.876` | `0.832` |

### Main message

`Unsupervised DAE underperformed PCA, but adding a supervised head changed the result.`

### Layout

- Put the table in the center
- Highlight the `w=0.5` row with a light accent color
- Add one short takeaway line below the table

### Speaker note (CN)

这一页不要把每个实验都讲得很细。  
重点只讲一个故事：最开始的 unsupervised DAE 不如 PCA，但加上 supervised head 之后结果明显提升，其中 `w=0.5` 最好。

---

## Slide 4: Best result vs PCA

### Title
`Best Run Improved Overall Performance`

### What goes on the slide

Focus only on the best run:

- Best run:
  `SupDAE-32-head-w0.5-noise0.10`
- Compared with `PCA-50` on test split:
  - accuracy: `0.943` vs `0.922`  (`+0.021`)
  - macro-F1: `0.898` vs `0.858`  (`+0.041`)
  - balanced accuracy: `0.868` vs `0.901`  (`-0.032`)

### Suggested visual

- Three metric cards or a three-row comparison table
- Green up-arrow for accuracy and macro-F1
- Orange caution marker for balanced accuracy

### Main message

`The best supervised DAE beat PCA on overall classification metrics, but not on balanced accuracy.`

### Speaker note (CN)

这一页就是你最核心的结果页。  
要诚实一点讲：这个版本在 accuracy 和 macro-F1 上超过了 PCA，但 balanced accuracy 还没超过，所以现在可以说“有进展”，但不能说“全面胜出”。

---

## Slide 5: Class-level story and claim

### Title
`Where the Gain Came From`

### What goes on the slide

Use this figure:

- [gse96583_supdae_head_w05_noise010_e10_per_class_delta.png](/home/zichende/projects/scRNA-ssl-uncertainty/deliverables/ms3_autoencoder/figures/gse96583_supdae_head_w05_noise010_e10_per_class_delta.png)

Add these three bullets:

- Biggest gains:
  `Megakaryocytes +0.159`
  `FCGR3A+ Monocytes +0.062`
  `Dendritic cells +0.052`
- No class dropped below PCA on this fixed split
- Improvement seems to come mostly from harder or rarer classes

### Layout

- Left: figure
- Right: three bullets
- Bottom: one-sentence interpretation

### Bottom-line sentence

`The supervised DAE seems to help most where PCA struggled more, especially for minority or harder classes.`

### Speaker note (CN)

这一页帮助你解释为什么 macro-F1 提升。  
因为 macro-F1 对少数类更敏感，所以当 `Megakaryocytes`、`Dendritic cells` 这些类变好时，macro-F1 就会上升得比较明显。

---

## Slide 6: Caveat and next step

### Title
`What We Can Claim Now`

### What goes on the slide

- This result is on the notebook's fixed combined `train / val / test` split
- It is **not yet** the original `batch1 -> batch2` / `batch2 -> batch1` cross-batch benchmark
- The current run used the cached `lite` dataset, not the full heavy MS2 object
- Next step:
  retest the best supervised setting under the original cross-batch and cross-condition story

### Layout

- Top: `What I can claim`
- Bottom: `What I cannot claim yet`

### Main message

`The current result is promising, but it is still an intermediate benchmark rather than the final cross-batch conclusion.`

### Speaker note (CN)

这一页是防守页。  
你要主动把 caveat 讲出来，这样老师或者组员问的时候你会显得更清楚。  
最关键一句就是：现在这个结果说明 supervised DAE 在 fixed split 上有希望，但还没有完成项目原始想回答的 cross-batch / cross-condition 问题。

---

## Backup Slide A: Metric explanation

### Title
`How to Read the Metrics`

### Put on slide

- `Accuracy`: overall proportion of correct predictions
- `Macro-F1`: average F1 across cell types, treating each class equally
- `Balanced accuracy`: average recall across classes

### Speaker note (CN)

如果有人问为什么你主要看 macro-F1，就说：因为 cell type 不平衡，macro-F1 比 accuracy 更能反映少数类表现。

---

## Backup Slide B: Implementation detail

### Title
`Unsupervised vs Supervised Implementation`

### Put on slide

- `Unsupervised DAE`
  - train with reconstruction loss only
  - encode cells into latent `z`
  - train logistic regression on `z`
- `Supervised DAE`
  - train with reconstruction loss plus classification loss
  - predict cell type directly from classifier head
  - best run used `supervised loss weight = 0.5`

### Speaker note (CN)

如果别人追问实现差别，这一页就够了。  
一句话：unsupervised 是“先学表示，再单独分类”，supervised 是“学表示和分类一起做”。

---

## Files to use when building the actual PPT

- Main result summary:
  [gse96583_supdae_head_w05_noise010_e10_metrics.csv](/home/zichende/projects/scRNA-ssl-uncertainty/results/gse96583_supdae_head_w05_noise010_e10_metrics.csv:1)
- Per-class detail:
  [gse96583_supdae_head_w05_noise010_e10_per_class.csv](/home/zichende/projects/scRNA-ssl-uncertainty/results/gse96583_supdae_head_w05_noise010_e10_per_class.csv:1)
- Best run slide draft:
  [gse96583_supdae_head_w05_noise010_e10_slides.md](/home/zichende/projects/scRNA-ssl-uncertainty/deliverables/ms3_autoencoder/gse96583_supdae_head_w05_noise010_e10_slides.md:1)
- Per-class delta figure:
  [gse96583_supdae_head_w05_noise010_e10_per_class_delta.png](/home/zichende/projects/scRNA-ssl-uncertainty/deliverables/ms3_autoencoder/figures/gse96583_supdae_head_w05_noise010_e10_per_class_delta.png)
- Training curve figure if needed:
  [gse96583_supdae_head_w05_noise010_e10_training_curve.png](/home/zichende/projects/scRNA-ssl-uncertainty/deliverables/ms3_autoencoder/figures/gse96583_supdae_head_w05_noise010_e10_training_curve.png)

## Recommended final one-sentence takeaway

`On the notebook's fixed split, a supervised denoising autoencoder improved accuracy and macro-F1 over PCA, suggesting that label-guided representation learning is promising, although the result still needs cross-batch validation.`
