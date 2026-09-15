"""
Feature extraction using Phikon (owkin/phikon) — a ViT-B pretrained on
6.1M TCGA histopathology patches. Replaces ImageNet ResNet50.

Outputs 768-dim features per patch instead of 2048-dim.
Features saved to camelyon16_features_phikon/ (separate from ResNet50 features).

Save to:  /mnt/DATA/sda1/Subhabrat/camelyon_remaining/extract_features_phikon.py
Run from: /mnt/DATA/sda1/Subhabrat/camelyon_remaining/
Command:  CUDA_VISIBLE_DEVICES=0 python3 extract_features_phikon.py
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel

MANIFEST_PATH = "patch_manifest.csv"
PATCHES_ROOT   = "camelyon16_patches"
FEATURES_ROOT  = "camelyon16_features_phikon"
BATCH_SIZE     = 128
MODEL_NAME     = "owkin/phikon"


def build_extractor(device):
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = ViTModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    print("Model loaded.\n")
    return processor, model


@torch.no_grad()
def extract_for_slide(processor, model, device, patch_dir, patch_files):
    all_feats = []
    for i in range(0, len(patch_files), BATCH_SIZE):
        batch_paths = patch_files[i:i+BATCH_SIZE]
        images = []
        for fname in batch_paths:
            try:
                img = Image.open(os.path.join(patch_dir, fname)).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"  [warn] {fname}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        # CLS token output — 768-dim feature per patch
        feats = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0) if all_feats else None


def main():
    os.makedirs(FEATURES_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processor, model = build_extractor(device)

    manifest = pd.read_csv(MANIFEST_PATH)
    slides   = manifest["slide_name"].unique()
    print(f"Total slides: {len(slides)}")

    # Save slide labels
    slide_labels = (
        manifest[["slide_name", "label"]]
        .drop_duplicates()
        .set_index("slide_name")["label"]
        .to_dict()
    )
    torch.save(slide_labels,
               os.path.join(FEATURES_ROOT, "slide_labels.pt"))

    for slide_name in tqdm(slides, desc="Slides"):
        out_path  = os.path.join(FEATURES_ROOT, f"{slide_name}.pt")
        if os.path.exists(out_path):
            tqdm.write(f"[skip] {slide_name}")
            continue

        slide_patches = manifest[manifest["slide_name"] == slide_name]
        subdir = "normal" if slide_labels[slide_name] == 0 else "tumor"
        
        # Check all possible locations where the slide patches might reside
        candidates = [
            slide_name,
            os.path.join(subdir, slide_name),
            os.path.join(PATCHES_ROOT, subdir, slide_name),
            os.path.join(PATCHES_ROOT, slide_name),
        ]
        patch_dir = next((d for d in candidates if os.path.isdir(d)), None)

        if patch_dir is None:
            tqdm.write(f"[MISSING] {slide_name}")
            continue

        # Filter out non-image files to prevent loading errors
        patch_files = sorted([f for f in os.listdir(patch_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        if not patch_files:
            tqdm.write(f"[warn] no valid images found in {patch_dir}")
            continue

        feats = extract_for_slide(
            processor, model, device, patch_dir, patch_files
        )

        if feats is None:
            tqdm.write(f"[warn] no features extracted for {slide_name}")
            continue

        torch.save(feats, out_path)
        tqdm.write(f"[done] {slide_name}: {feats.shape}")

    pt_files = [f for f in os.listdir(FEATURES_ROOT)
                if f.endswith(".pt") and f != "slide_labels.pt"]
    print(f"\nTotal feature files: {len(pt_files)} (expected 150)")
    print(f"Feature dim: 768 (Phikon ViT-B CLS token)")
    print(f"\nNext: run train_mil_phikon.py")


if __name__ == "__main__":
    main()