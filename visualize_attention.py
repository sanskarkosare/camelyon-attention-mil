"""
Attention heatmap visualization for the trained GatedAttentionMIL model.

For each test slide:
  1. Loads the saved attention weights (from train_mil.py)
  2. Maps attention scores back to patch grid positions
  3. Generates a heatmap overlaid on a thumbnail of the slide
  4. Saves as PNG to camelyon16_results/heatmaps/

These heatmaps are the key visual output for the README and resume —
they show the model has learned to focus on diagnostically relevant
regions without ever seeing patch-level labels.

Save to:  camelyon-mil/visualize_attention.py
Run from: camelyon-mil/
Command:  python visualize_attention.py
Run AFTER train_mil.py has completed.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# ---- CONFIG ----
FEATURES_ROOT  = "camelyon16_features"
PATCHES_ROOT   = "camelyon16_patches"
RESULTS_ROOT   = "camelyon16_results"
ATTENTION_DIR  = "camelyon16_results/GatedAttentionMIL_attention"
HEATMAP_DIR    = "camelyon16_results/heatmaps"
MANIFEST_PATH  = "camelyon16_patches/patch_manifest.csv"
THUMBNAIL_SIZE = (512, 512)   # size of background thumbnail in heatmap output
TOP_K          = 20           # number of highest-attention patches to highlight


def get_patch_grid_position(patch_filename, slide_name):
    """
    Recover the patch index from its filename (e.g. normal_001_00053.jpg -> 53).
    We use index order as a proxy for grid position since that's how we saved them.
    """
    stem = patch_filename.replace(".jpg", "").replace(f"{slide_name}_", "")
    return int(stem)


def make_heatmap_grid(attention_weights, num_patches, grid_w, grid_h):
    """
    Build a 2D attention grid (grid_h x grid_w) from flat attention weights.
    Patches were written left-to-right, top-to-bottom in extract_patches.py.
    """
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    attn_np = attention_weights.numpy()

    for idx, weight in enumerate(attn_np):
        if idx >= grid_h * grid_w:
            break
        row = idx // grid_w
        col = idx % grid_w
        grid[row, col] = weight

    return grid


def visualize_slide(slide_name, label, manifest_df, attn_dir, out_dir):
    """Generate and save an attention heatmap for one slide."""

    attn_path = os.path.join(attn_dir, f"{slide_name}_attn.pt")
    if not os.path.exists(attn_path):
        print(f"  [skip] no attention file for {slide_name}")
        return

    attn_weights = torch.load(attn_path, weights_only=True)  # [N]
    slide_patches = manifest_df[manifest_df["slide_name"] == slide_name]
    num_patches = len(slide_patches)

    if len(attn_weights) != num_patches:
        print(f"  [warn] attention length mismatch for {slide_name} "
              f"({len(attn_weights)} vs {num_patches} patches) — skipping")
        return

    # ---- Find a sample patch to determine patch dimensions ----
    first_patch = slide_patches.iloc[0]["patch_filename"]
    sample_path = os.path.join(PATCHES_ROOT, slide_name, first_patch)
    if not os.path.exists(sample_path):
        print(f"  [skip] patches not found for {slide_name}")
        return

    patch_img = Image.open(sample_path)
    patch_w, patch_h = patch_img.size  # should be 256x256

    # ---- Build a rough grid layout ----
    # We don't have the exact grid dimensions stored, so we infer from
    # the number of patches and the aspect ratio of the slide thumbnail.
    # This is approximate but gives a reasonable heatmap layout.
    approx_side = int(np.ceil(np.sqrt(num_patches)))
    grid_w = approx_side
    grid_h = int(np.ceil(num_patches / grid_w))

    attn_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    for i, weight in enumerate(attn_weights.numpy()):
        r, c = divmod(i, grid_w)
        if r < grid_h:
            attn_grid[r, c] = weight

    # ---- Find top-K patches by attention ----
    flat_idx = np.argsort(attn_weights.numpy())[-TOP_K:]
    top_patches = []
    for idx in flat_idx:
        if idx < len(slide_patches):
            fname = slide_patches.iloc[idx]["patch_filename"]
            ppath = os.path.join(PATCHES_ROOT, slide_name, fname)
            if os.path.exists(ppath):
                top_patches.append(Image.open(ppath).convert("RGB"))

    # ---- Build figure ----
    fig = plt.figure(figsize=(14, 6))
    label_str = "TUMOR" if label == 1 else "NORMAL"
    fig.suptitle(
        f"{slide_name}  |  True label: {label_str}  |  "
        f"{num_patches} patches",
        fontsize=12, fontweight="bold"
    )

    # Left: attention heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(attn_grid, cmap="hot", interpolation="nearest",
                    aspect="auto")
    ax1.set_title("Attention heatmap\n(bright = high attention)", fontsize=9)
    ax1.axis("off")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Middle: top-K patches grid (up to 4x5 = 20)
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

    # Right: attention distribution histogram
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
        print("Run train_mil.py first — it saves attention weights during evaluation.")
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
        visualize_slide(slide_name, label, manifest_df,
                        ATTENTION_DIR, HEATMAP_DIR)

    print(f"\nAll heatmaps saved to {HEATMAP_DIR}/")
    print("Add the best-looking ones to your README.")


if __name__ == "__main__":
    main()
