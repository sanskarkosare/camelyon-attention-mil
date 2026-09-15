"""
Attention heatmap visualization for the trained GatedAttentionMIL model
(Phikon features, seed=2 — the best result).

For each test slide:
  1. Loads the saved attention weights
  2. Maps attention scores back to approximate patch grid positions
  3. Generates a 3-panel figure: heatmap, top-attention patches, distribution
  4. Saves as PNG to camelyon16_results/heatmaps/

Save to:  camelyon-mil/visualize_attention.py  (overwrite existing)
Run from: camelyon-mil/
Command:  python visualize_attention.py
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---- CONFIG ----
FEATURES_ROOT  = "camelyon16_features"
PATCHES_ROOT   = "camelyon16_patches"
HEATMAP_DIR    = "camelyon16_results/heatmaps"
MANIFEST_PATH  = "camelyon16_patches/patch_manifest.csv"

# FIXED: points to the actual nested folder from the scp download
ATTENTION_DIR  = "attention_weights/seed2_GatedAttentionMIL_attention"

TOP_K = 20


def visualize_slide(slide_name, label, manifest_df, attn_dir, out_dir):
    attn_path = os.path.join(attn_dir, f"{slide_name}_attn.pt")
    if not os.path.exists(attn_path):
        print(f"  [skip] no attention file for {slide_name}")
        return

    attn_weights = torch.load(attn_path, weights_only=True)
    slide_patches = manifest_df[manifest_df["slide_name"] == slide_name]
    num_patches = len(slide_patches)

    if len(attn_weights) != num_patches:
        print(f"  [warn] attention length mismatch for {slide_name} "
              f"({len(attn_weights)} vs {num_patches} patches) — skipping")
        return

    first_patch = slide_patches.iloc[0]["patch_filename"]
    sample_path = os.path.join(PATCHES_ROOT, slide_name, first_patch)
    if not os.path.exists(sample_path):
        print(f"  [skip] patches not found for {slide_name} at {sample_path}")
        return

    patch_img = Image.open(sample_path)
    patch_w, patch_h = patch_img.size

    approx_side = int(np.ceil(np.sqrt(num_patches)))
    grid_w = approx_side
    grid_h = int(np.ceil(num_patches / grid_w))

    attn_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    for i, weight in enumerate(attn_weights.numpy()):
        r, c = divmod(i, grid_w)
        if r < grid_h:
            attn_grid[r, c] = weight

    flat_idx = np.argsort(attn_weights.numpy())[-TOP_K:]
    top_patches = []
    for idx in flat_idx:
        if idx < len(slide_patches):
            fname = slide_patches.iloc[idx]["patch_filename"]
            ppath = os.path.join(PATCHES_ROOT, slide_name, fname)
            if os.path.exists(ppath):
                top_patches.append(Image.open(ppath).convert("RGB"))

    fig = plt.figure(figsize=(14, 6))
    label_str = "TUMOR" if label == 1 else "NORMAL"
    fig.suptitle(
        f"{slide_name}  |  True label: {label_str}  |  {num_patches} patches",
        fontsize=12, fontweight="bold"
    )

    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(attn_grid, cmap="hot", interpolation="nearest", aspect="auto")
    ax1.set_title("Attention heatmap\n(bright = high attention)", fontsize=9)
    ax1.axis("off")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 3, 2)
    if top_patches:
        cols = 4
        rows = int(np.ceil(len(top_patches) / cols))
        grid_img = Image.new("RGB", (cols * patch_w, rows * patch_h), (255, 255, 255))
        for i, p in enumerate(top_patches):
            r, c = divmod(i, cols)
            grid_img.paste(p, (c * patch_w, r * patch_h))
        ax2.imshow(np.array(grid_img))
    ax2.set_title(f"Top-{TOP_K} highest\nattention patches", fontsize=9)
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(attn_weights.numpy(), bins=50, color="steelblue", edgecolor="none")
    ax3.axvline(np.percentile(attn_weights.numpy(), 90),
                color="red", linestyle="--", label="90th percentile")
    ax3.set_xlabel("Attention weight")
    ax3.set_ylabel("Patch count")
    ax3.set_title("Attention distribution", fontsize=9)
    ax3.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{slide_name}_heatmap.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    if not os.path.exists(ATTENTION_DIR):
        print(f"ERROR: {ATTENTION_DIR} not found.")
        print("Check the actual folder path under attention_weights/ and update ATTENTION_DIR.")
        return

    manifest_df  = pd.read_csv(MANIFEST_PATH)
    slide_labels = torch.load(
        os.path.join(FEATURES_ROOT, "slide_labels.pt"), weights_only=True
    )

    attn_files = [f for f in os.listdir(ATTENTION_DIR) if f.endswith("_attn.pt")]
    slide_names = [f.replace("_attn.pt", "") for f in attn_files]

    print(f"Generating heatmaps for {len(slide_names)} test slides...\n")

    for slide_name in sorted(slide_names):
        label = slide_labels.get(slide_name, -1)
        print(f"[{slide_name}] label={label}")
        visualize_slide(slide_name, label, manifest_df, ATTENTION_DIR, HEATMAP_DIR)

    print(f"\nAll heatmaps saved to {HEATMAP_DIR}/")
    print("Pick the 2-3 clearest tumor-slide heatmaps for the README.")


if __name__ == "__main__":
    main()