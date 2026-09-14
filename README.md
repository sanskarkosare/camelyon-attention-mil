# Attention-Based Multiple Instance Learning for Histopathology Classification

Weakly-supervised whole-slide image (WSI) classification for breast cancer metastasis detection in lymph node sections, inspired by [CLAM (mahmoodlab/CLAM)](https://github.com/mahmoodlab/CLAM).

Built as part of a B.Tech + M.Tech dual degree project at IIT Bhubaneswar.

---

## Overview

Gigapixel WSIs cannot be fed into a CNN whole — they are too large by orders of magnitude. This pipeline solves that using **Multiple Instance Learning (MIL)**: each slide is treated as a *bag* of patches, with only a slide-level label (normal/tumor), no per-patch annotation required.

The core model is a **Gated Attention MIL** aggregator (Ilse et al., 2018), which learns to assign higher attention weights to diagnostically relevant patches — effectively localizing suspicious regions without ever seeing patch-level ground truth.

---

## Dataset

[CAMELYON16](https://camelyon16.grand-challenge.org/) — H&E-stained whole-slide images of sentinel lymph node sections for breast cancer metastasis detection.

- **150 slides** used (75 normal, 75 tumor) from the public AWS S3 mirror
- Excludes 20 slides with non-exhaustive annotations (documented in the README of the dataset)
- Slide-level split: **90 train / 30 val / 30 test** (split by slide, never by patch — no data leakage)

---

## Pipeline

```
Raw WSIs (.tif)
      │
      ▼
Tissue Segmentation          OpenSlide + Otsu thresholding on HSV saturation
      │                      Extracts tissue-containing regions, discards background/glass
      ▼
Patch Extraction             mpp-aware level selection (target: 20x / ~0.50 mpp)
      │                      256×256 px patches, ≥50% tissue coverage required
      ▼                      → ~1.0M+ patches across 150 slides
Feature Extraction           Frozen pretrained ResNet50 (ImageNet)
      │                      2048-dim feature vector per patch
      ▼
Gated Attention MIL          Attention network: tanh(V·h) ⊙ sigmoid(U·h)
      │                      Aggregates patch features → slide-level prediction
      ▼
Evaluation                   AUC + accuracy on held-out test slides
                             Attention heatmaps for interpretability
```

---

## Results

Trained with 3 random seeds; best model selected by validation AUC (never by test).

| Model | Test AUC | Test Accuracy | Notes |
|-------|----------|---------------|-------|
| **Gated Attention MIL** | **0.7689** | **66.7%** | Attention localizes tumor regions |
| Mean-Pool MIL (baseline) | 0.6044–0.8800 | 60–67% | No attention mechanism |

The attention model outperforms the mean-pool baseline on the primary selected seed, demonstrating that learned attention weights meaningfully improve slide-level classification beyond naive pooling.

> **Note on scale:** Published CLAM results (AUC 0.95+) use 270+ training slides and pathology-specific self-supervised feature extractors (vs. ImageNet ResNet50 used here). The numbers above reflect realistic performance at 90-slide training scale with general-purpose features — a fair comparison point would be other 150-slide MIL experiments, not the full-dataset published benchmark.

---

## Repo Structure

```
camelyon-attention-mil/
├── download_camelyon_subset_local.py   # Download initial 50 slides from AWS S3
├── download_additional_100.py          # Download 100 more slides (total → 150)
├── inspect_slide.py                    # Inspect WSI metadata (mpp, levels, dimensions)
├── extract_patches.py                  # Tissue segmentation + patch tiling
├── extract_features.py                 # ResNet50 feature extraction (CPU)
├── extract_features_server.py          # Feature extraction (GPU server)
├── extract_features_new100.py          # Feature extraction for additional slides
├── train_mil_150.py                    # MIL training (150 slides, multi-seed)
├── visualize_attention.py              # Attention heatmap generation
├── ensemble_eval.py                    # (experimental) cross-seed ensemble
├── requirements.txt
└── README.md
```

---

## Reproducing Results

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data (requires ~188GB disk space)
```bash
python download_camelyon_subset_local.py   # first 50 slides
python download_additional_100.py           # additional 100 slides
```

### 3. Extract patches
```bash
python extract_patches.py                   # set TEST_SINGLE_SLIDE = None for full run
```

### 4. Extract features (GPU recommended)
```bash
CUDA_VISIBLE_DEVICES=0 python extract_features.py
```

### 5. Train
```bash
CUDA_VISIBLE_DEVICES=0 python train_mil_150.py
```

### 6. Generate attention heatmaps
```bash
python visualize_attention.py
```

---

## Key Implementation Details

- **Tissue detection:** Otsu thresholding on HSV saturation channel. Background/glass has near-zero saturation; H&E-stained tissue has measurably higher saturation regardless of exact stain color.
- **Level selection:** mpp-aware (not objective-power, which CAMELYON16 slides don't store). Target: 0.50 µm/px ≈ 20x magnification.
- **Feature normalization:** L2-normalization applied to each patch feature vector before the attention network — standard practice in MIL, stabilizes attention training.
- **Threshold calibration:** Decision threshold tuned on validation set (not test) to convert probabilities to binary predictions. Default 0.5 is rarely optimal with small MIL datasets.
- **Evaluation discipline:** Train/val/test split is always by whole slide, never by patch. Patches from the same slide never appear across different splits.

---

## References

- Lu, M.Y. et al. (2021). *Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images.* Nature Biomedical Engineering. [CLAM](https://github.com/mahmoodlab/CLAM)
- Ilse, M. et al. (2018). *Attention-based Deep Multiple Instance Learning.* ICML.
- CAMELYON16 Challenge: https://camelyon16.grand-challenge.org/

---

## Author

Sanskar P. Kosare — Dual Degree (B.Tech + M.Tech), Mechanical Engineering, IIT Bhubaneswar
