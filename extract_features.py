"""
Feature extraction for CAMELYON16 patches using a frozen pretrained ResNet50.

For each slide:
  - Loads all its patches from camelyon16_patches/<slide_name>/
  - Passes them through ResNet50 (ImageNet pretrained, final FC removed)
  - Saves a [num_patches, 2048] feature tensor to camelyon16_features/<slide_name>.pt

This step runs once and caches features to disk. Training the MIL model later
uses these cached features, not the raw images, so this slow CPU step is a
one-time cost.

Save this file to: camelyon-mil/extract_features.py
Run from:         camelyon-mil/
Command:          python extract_features.py
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ---- CONFIG ----
MANIFEST_PATH   = "camelyon16_patches/patch_manifest.csv"
PATCHES_ROOT    = "camelyon16_patches"
FEATURES_ROOT   = "camelyon16_features"
BATCH_SIZE      = 64    # patches per forward pass — reduce to 32 if you get OOM

# Standard ImageNet normalisation — must match what ResNet50 was trained with
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


def build_feature_extractor():
    """
    Load pretrained ResNet50 and strip the final FC layer.
    Output: 2048-dim feature vector per patch (after global average pooling).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove the classification head — keep everything up to and including avgpool
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    # Freeze all parameters — we are not training this network
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_features_for_slide(model, device, slide_name, patch_files):
    """
    Run all patches for one slide through the feature extractor.
    Returns a tensor of shape [num_patches, 2048].
    """
    all_features = []

    for i in range(0, len(patch_files), BATCH_SIZE):
        batch_paths = patch_files[i : i + BATCH_SIZE]
        batch_tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(TRANSFORM(img))
            except Exception as e:
                print(f"  [warn] could not load {p}: {e} — skipping")

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            feats = model(batch)           # [B, 2048, 1, 1]
            feats = feats.squeeze(-1).squeeze(-1)  # [B, 2048]

        all_features.append(feats.cpu())

    if not all_features:
        return None

    return torch.cat(all_features, dim=0)  # [num_patches, 2048]


def main():
    os.makedirs(FEATURES_ROOT, exist_ok=True)

    # Read manifest to get slide->patches mapping
    manifest = pd.read_csv(MANIFEST_PATH)
    slides = manifest["slide_name"].unique()
    print(f"Found {len(slides)} slides, {len(manifest)} total patches\n")

    # Use CPU (or CUDA if somehow available on this machine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Running on CPU — this will take 30-60 minutes. "
              "Safe to leave running in background.\n")

    model = build_feature_extractor().to(device)

    for slide_name in tqdm(slides, desc="Slides"):
        out_path = os.path.join(FEATURES_ROOT, f"{slide_name}.pt")
        if os.path.exists(out_path):
            tqdm.write(f"[skip] {slide_name} already done")
            continue

        slide_patches = manifest[manifest["slide_name"] == slide_name]
        patch_files = [
            os.path.join(PATCHES_ROOT, slide_name, row["patch_filename"])
            for _, row in slide_patches.iterrows()
        ]

        feats = extract_features_for_slide(model, device, slide_name, patch_files)

        if feats is None:
            tqdm.write(f"[warn] {slide_name} produced no features — skipping")
            continue

        torch.save(feats, out_path)
        tqdm.write(f"[done] {slide_name}: {feats.shape[0]} patches → {feats.shape}")

    # Save a slide-level label lookup for the MIL training step
    slide_labels = (
        manifest[["slide_name", "label"]]
        .drop_duplicates()
        .set_index("slide_name")["label"]
        .to_dict()
    )
    label_path = os.path.join(FEATURES_ROOT, "slide_labels.pt")
    torch.save(slide_labels, label_path)
    print(f"\nSlide labels saved to {label_path}")
    print(f"Feature files saved to {FEATURES_ROOT}/")
    print("Done. Next step: MIL bag construction + model training.")


if __name__ == "__main__":
    main()
