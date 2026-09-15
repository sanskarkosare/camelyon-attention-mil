Attention-Based Multiple Instance Learning for Histopathology Classification
Weakly-supervised whole-slide image (WSI) classification for breast cancer metastasis detection in lymph node sections, inspired by CLAM (mahmoodlab/CLAM).
Built as part of a B.Tech + M.Tech dual degree project at IIT Bhubaneswar.
---
Overview
Gigapixel WSIs cannot be fed into a CNN whole — they are too large by orders of magnitude. This pipeline solves that using Multiple Instance Learning (MIL): each slide is treated as a bag of patches, with only a slide-level label (normal/tumor), no per-patch annotation required.
The core model is a Gated Attention MIL aggregator (Ilse et al., 2018), which learns to assign higher attention weights to diagnostically relevant patches — effectively localizing suspicious regions without ever seeing patch-level ground truth.
---
Dataset
CAMELYON16 — H&E-stained whole-slide images of sentinel lymph node sections for breast cancer metastasis detection.
150 slides (75 normal, 75 tumor) from the public AWS S3 mirror
Excludes slides with non-exhaustive annotations (documented in the dataset README)
Slide-level split: 90 train / 30 val / 30 test (always by slide, never by patch)
---
Pipeline
```
Raw WSIs (.tif, gigapixel)
      │
      ▼
Tissue Segmentation          OpenSlide + Otsu thresholding on HSV saturation
      │                      Discards background/glass, retains tissue regions
      ▼
Patch Extraction             mpp-aware level selection (target: 20x / ~0.50 mpp)
      │                      256×256 px patches, ≥50% tissue coverage required
      ▼                      → 1M+ patches across 150 slides
Feature Extraction           Phikon (ViT-B) — pathology-pretrained on 6.1M TCGA patches
      │                      768-dim CLS token feature per patch
      ▼
Gated Attention MIL          a = softmax(w^T (tanh(Vh) ⊙ sigmoid(Uh)))
      │                      z = Σ a_k · h_k  →  slide-level prediction
      ▼
Evaluation                   AUC + calibrated accuracy on 30 held-out test slides
                             Attention weights saved for heatmap visualization
```
---
Results
Trained with 10 random seeds; model selected by validation AUC (never test).
L2-normalized features, dropout=0.5, Adam lr=1e-4, gradient clipping, 60 epochs.
Model	Test AUC	Test Accuracy	Notes
Gated Attention MIL	94.2%	83.3%	Phikon features, seed=2
Mean-Pool MIL (baseline)	76.4%	66.7%	Same features, no attention
Attention model outperforms mean-pool baseline by 17.8% AUC, confirming that learned patch-level attention weights meaningfully improve slide-level classification.
> **Note on reproducibility:** Results vary across random splits (10 seeds tested; attention model median AUC ~90%). This is expected — with 30 test slides, split composition affects results. All seed results are saved in `camelyon16_results_phikon/summary.pt`.
---
Key Design Decisions
Why Phikon instead of ImageNet ResNet50?
Standard ImageNet features encode generic visual patterns (edges, textures). Phikon is pretrained on 6.1M TCGA histopathology patches and encodes pathologically meaningful tissue representations. Switching extractors improved test AUC from ~77% → 94%.
Why slide-level splits?
Patches from the same slide share tissue morphology. Splitting by patch (not slide) causes leakage — the model sees test-slide tissue during training. All splits here are strictly slide-level.
Why threshold calibration?
The default decision threshold (0.5) assumes calibrated probabilities. With small MIL datasets this rarely holds. We tune the threshold on the validation set and apply it to test — standard practice, not post-hoc tuning.
---
Repo Structure
```
camelyon-attention-mil/
├── download_camelyon_subset_local.py   # Download initial 50 slides from AWS S3
├── download_additional_100.py          # Download 100 more slides (total → 150)
├── inspect_slide.py                    # Inspect WSI metadata (mpp, levels, dims)
├── extract_patches.py                  # Tissue segmentation + patch tiling
├── extract_features.py                 # ResNet50 feature extraction (CPU baseline)
├── extract_features_phikon.py          # Phikon (ViT-B) feature extraction (GPU)
├── extract_features_server.py          # GPU server variant
├── extract_features_new100.py          # Feature extraction for additional slides
├── train_mil_150.py                    # MIL training — ResNet50 features
├── train_mil_phikon.py                 # MIL training — Phikon features (main)
├── visualize_attention.py              # Attention heatmap generation
├── requirements.txt
└── README.md
```
---
Reproducing Results
1. Install dependencies
```bash
pip install -r requirements.txt
pip install transformers   # for Phikon
```
2. Download data (~188GB disk space required)
```bash
python download_camelyon_subset_local.py
python download_additional_100.py
```
3. Extract patches
```bash
python extract_patches.py   # set TEST_SINGLE_SLIDE = None for full run
```
4. Extract Phikon features (GPU required, ~30 min on A10)
```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_phikon.py
```
5. Train
```bash
CUDA_VISIBLE_DEVICES=0 python train_mil_phikon.py
```
---
References
Lu, M.Y. et al. (2021). Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images. Nature Biomedical Engineering. — CLAM
Ilse, M. et al. (2018). Attention-based Deep Multiple Instance Learning. ICML.
Filiot, A. et al. (2023). Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling. — Phikon
CAMELYON16: https://camelyon16.grand-challenge.org/
---
Author
Sanskar P. Kosare — Dual Degree (B.Tech + M.Tech), Mechanical Engineering, IIT Bhubaneswar
