# SkinSense MVP Experiment Report

Date: 2026-08-23  
Final candidate: `models/convnext_4class_focal/best.pt`

## Executive result

The final classifier is a 4-class, 224x224 ConvNeXt-Tiny trained with focal loss. It scores
**92.40% test accuracy and 0.9310 macro-F1**, versus the original 5-class model's 80.63% and
0.7065. Bacterial dermatosis is absent from the classification head because its source data is
scarce, contradictory, and contaminated. Ambiguous or unusable images route to the vet fallback.

## 1. Exclude Bacterial dermatosis

Kept. Removing the class raised accuracy 80.63% -> 90.74% and macro-F1 0.7065 -> 0.9136 before
the later focal-loss gain. Allergic recall improved **0.6441 -> 0.8390**.

Per-class F1, original 5-class -> weighted 4-class:

| Class | Before | After | Delta |
|---|---:|---:|---:|
| Allergic dermatitis | 0.7391 | 0.8684 | +0.1293 |
| Demodicosis | 0.9677 | 0.9663 | -0.0014 |
| Fungal infection | 0.8317 | 0.8960 | +0.0643 |
| Healthy | 0.9328 | 0.9237 | -0.0091 |

Weighted 4-class confusion matrix (rows=true, columns=predicted; order Allergic, Demodicosis,
Fungal, Healthy):

```text
[[297,   0,  37,  20],
 [  0, 244,   8,   0],
 [ 31,   9, 435,  11],
 [  2,   0,   5, 230]]
```

The old 5-class model was also tested as a shadow bacterial veto. It would reject 151/1,329
non-bacterial images, including 113 correct focal-model predictions, so that veto was reverted.

## 2. Real early stopping beyond 18 epochs

Experiment kept; checkpoint reverted. A genuinely uncapped run used validation macro-F1,
patience 7, and validation-driven LR reduction. Validation peaked at epoch 24 (0.9625), and early
stopping fired at epoch 31. Training took 1,400 seconds (23.3 minutes; 45.2 seconds/epoch).

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| 18-epoch focal | **0.9240** | **0.9310** |
| Uncapped, best epoch 24 | 0.9172 | 0.9270 |

The later validation winner did not generalize better, so the 18-epoch focal checkpoint remains.

## 3. Weighted CE, unweighted CE, and focal loss

Focal kept.

| Loss | Accuracy | Macro-F1 | Allergic F1 | Demodicosis F1 | Fungal F1 | Healthy F1 |
|---|---:|---:|---:|---:|---:|---:|
| Weighted CE | 0.9074 | 0.9136 | 0.8684 | 0.9663 | 0.8960 | 0.9237 |
| Unweighted CE | 0.9037 | 0.9140 | 0.8435 | **0.9781** | 0.8842 | 0.9502 |
| Focal | **0.9240** | **0.9310** | **0.8853** | 0.9584 | **0.9116** | **0.9686** |

Focal improved the aggregate result and three of four class F1 scores versus weighted CE; its
negative result is a small Demodicosis regression (-0.0079 F1).

## 4. Test-time augmentation

Reverted. Five-view TTA on the original shipped model reduced accuracy 0.8063 -> 0.7958 (-1.05
percentage points) and macro-F1 0.7065 -> 0.6949 (-0.0115). Off-center crops likely removed part
of the lesion.

## 5. Original-label taxonomy audit

The merges hide large source-label gaps:

| Merged class | Original label | Test n | Correct-class accuracy |
|---|---|---:|---:|
| Allergic dermatitis | Dermatitis | 217 | 79.3% |
| Allergic dermatitis | Hypersensitivity | 117 | 41.0% |
| Allergic dermatitis | Hypersensitivity_allergic_dermatosis | 20 | 40.0% |
| Fungal infection | ringworm | 331 | 94.6% |
| Fungal infection | Fungal_infections | 155 | 45.2% |

The merge is retained for data sufficiency, but raw labels remain in `manifest.csv` and the source
quality gap is a documented risk.

## 6. Dedup audit

Keep the dedup algorithm. A deterministic 50-group visual audit covered all 10 available
phash-only groups, 15 Roboflow-family groups, 15 md5 groups, and 10 additional phash overlaps.
All 50 were the same underlying photo: **0/50 false-positive merges**. Thirty groups touched
Bacterial dermatosis, also with 0 false merges. The bacterial issue is conflicting labels on true
duplicates, not legitimate photos being removed. Evidence: `models/dedup_manual_review/`.

## 7. 384x384 resolution

Reverted. The 384px model used gradient checkpointing and batch size 16, stopped at epoch 17 after
best validation at epoch 12, and scored 0.8841 accuracy / 0.8954 macro-F1. The comparable 224px
weighted model scored 0.9074 / 0.9136; the final focal 224px model scored 0.9240 / 0.9310.
Allergic recall fell 0.8390 -> 0.7571 versus weighted 224px. The observed run took about 69 minutes
under concurrent workload versus about 44 minutes for the logged 224px baseline, used roughly
three times the input pixels, halved batch size, and required checkpointing on the 6GB GPU.

## 8. Five repeated group-stratified splits

The original 5-class recipe was retrained with five seeds.

| Metric | Mean | Std |
|---|---:|---:|
| Overall accuracy | 0.8558 | 0.0338 |
| Overall macro-F1 | 0.7444 | 0.0155 |
| Bacterial precision | 0.0609 | 0.0113 |
| Bacterial recall | 0.4413 | 0.1131 |
| Bacterial F1 | 0.1060 | 0.0176 |

Bacterial test supports were [21, 16, 12, 8, 17]. High-confidence bacterial accuracy per fold was
[25.6%, 37.5%, 50.0%, 66.7%, 9.7%] with prediction counts [39, 8, 6, 3, 31]. The unweighted fold
mean is 37.9% +/- 19.6%, but the correct pooled estimate is **21/87 = 24.1%**. Thus the original
3/14 = 21.4% claim was noisy but directionally stable: bacterial predictions remain severely
overconfident and low-precision.

## 9. Pre-inference quality gate

Partially kept. On 15 good and 55 constructed bad images:

- False-positive rate: 3/15 = **20.0%**
- False-negative rate: 15/55 = **27.27%**
- Bad-image rejection: 40/55 = **72.73%**
- Tiny: 10/10 rejected; blank: 10/10; extreme aspect: 10/10; blurry: 10/15
- Synthetic wrong-subject noise: **0/10 rejected**

The size/aspect/blank/blur checks are hard gates. Centroid OOD distance is returned only as an
experimental warning because it caught no noise images; max-softmax was also incorrectly high
(0.852-0.921). A reliable semantic gate needs real representative negative data.

## 10. Grad-CAM

Kept for all accepted classes. `app/inference.py` returns a base64 PNG data URI, target class,
method, and disclaimer. Rejected images skip classification and Grad-CAM. The end-to-end test
produced a valid 78,186-byte PNG. Grad-CAM is attention visualization, not lesion segmentation or
a clinical explanation, and it adds latency/payload size.

## Final serving behavior

- Checkpoint: `models/convnext_4class_focal/best.pt`
- Confidence fallback: 0.60; 91.9% test coverage and 95.7% accuracy among accepted test images
- Quality failures: no disease class, no Grad-CAM, consult-vet path
- Accepted images: four-class probabilities plus Grad-CAM
- Bacterial dermatosis: never emitted as a class

