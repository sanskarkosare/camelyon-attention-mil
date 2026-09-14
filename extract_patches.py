"""
Tissue segmentation + patch extraction for CAMELYON16 WSIs.
Level selection uses mpp (microns-per-pixel) instead of objective power,
since CAMELYON16 slides don't store objective power but do store mpp.

Convention: 0.25 mpp = 40x, 0.50 mpp = 20x, 1.0 mpp = 10x
We target 20x (0.50 mpp) for patch extraction — standard for MIL on CAMELYON16.

Run on ONE slide first (TEST_SINGLE_SLIDE = "normal_001"), check the output
patches look like real tissue, then set TEST_SINGLE_SLIDE = None for full batch.
"""

import openslide
import numpy as np
import cv2
import os
import csv

# ---- CONFIG ----
DATA_ROOT        = "camelyon16_subset"   # contains normal/ and tumor/
OUTPUT_ROOT      = "camelyon16_patches"  # extracted patches go here
PATCH_SIZE       = 256                   # pixels at the extraction level
TARGET_MPP       = 0.50                  # 0.50 mpp = 20x magnification
TISSUE_THRESHOLD = 0.50                  # min fraction of patch that must be tissue
THUMBNAIL_SIZE   = 2048                  # longest side of thumbnail for tissue mask

# Single slide test mode — set to None to run the full batch
TEST_SINGLE_SLIDE = None


def get_tissue_mask(slide):
    thumb = slide.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    thumb_rgb = np.array(thumb.convert("RGB"))
    thumb_h, thumb_w = thumb_rgb.shape[:2]

    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask, thumb_w, thumb_h


def pick_level_by_mpp(slide, target_mpp):
    base_mpp = slide.properties.get("openslide.mpp-x")

    if base_mpp is None:
        print("  [warn] mpp-x not found - defaulting to level 1 (assumed 20x)")
        return 1, slide.level_downsamples[1]

    base_mpp = float(base_mpp)
    best_level = 0
    best_diff  = float("inf")

    for level, downsample in enumerate(slide.level_downsamples):
        effective_mpp = base_mpp * downsample
        diff = abs(effective_mpp - target_mpp)
        if diff < best_diff:
            best_diff  = diff
            best_level = level

    effective = base_mpp * slide.level_downsamples[best_level]
    print(f"  extraction level: {best_level} "
          f"(downsample={slide.level_downsamples[best_level]:.0f}, "
          f"effective mpp={effective:.3f})")
    return best_level, slide.level_downsamples[best_level]


def extract_patches_for_slide(slide_path, label, slide_name, writer):
    slide = openslide.OpenSlide(slide_path)

    tissue_mask, thumb_w, thumb_h = get_tissue_mask(slide)
    level, downsample = pick_level_by_mpp(slide, TARGET_MPP)
    level_w, level_h  = slide.level_dimensions[level]

    scale_x = level_w / thumb_w
    scale_y = level_h / thumb_h

    patch_w_thumb = max(1, int(PATCH_SIZE / scale_x))
    patch_h_thumb = max(1, int(PATCH_SIZE / scale_y))

    slide_out_dir = os.path.join(OUTPUT_ROOT, slide_name)
    os.makedirs(slide_out_dir, exist_ok=True)

    patch_count = 0

    for ty in range(0, thumb_h - patch_h_thumb, patch_h_thumb):
        for tx in range(0, thumb_w - patch_w_thumb, patch_w_thumb):

            tile_mask   = tissue_mask[ty:ty + patch_h_thumb,
                                      tx:tx + patch_w_thumb]
            tissue_frac = np.mean(tile_mask > 0)
            if tissue_frac < TISSUE_THRESHOLD:
                continue

            level0_x = int(tx * scale_x * downsample)
            level0_y = int(ty * scale_y * downsample)

            region = slide.read_region(
                (level0_x, level0_y), level, (PATCH_SIZE, PATCH_SIZE)
            ).convert("RGB")

            patch_filename = f"{slide_name}_{patch_count:05d}.jpg"
            region.save(
                os.path.join(slide_out_dir, patch_filename), quality=90
            )
            writer.writerow([slide_name, patch_filename, label])
            patch_count += 1

    slide.close()
    print(f"  -> {patch_count} patches saved")
    return patch_count


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_ROOT, "patch_manifest.csv")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_name", "patch_filename", "label"])

        for label, subdir in [(0, "normal"), (1, "tumor")]:
            folder = os.path.join(DATA_ROOT, subdir)
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith(".tif"):
                    continue
                slide_name = fname.replace(".tif", "")

                if (TEST_SINGLE_SLIDE is not None
                        and slide_name != TEST_SINGLE_SLIDE):
                    continue

                print(f"\n[{slide_name}] label={label}")
                extract_patches_for_slide(
                    os.path.join(folder, fname), label, slide_name, writer
                )

    print(f"\nManifest written to {manifest_path}")
    if TEST_SINGLE_SLIDE is not None:
        print("\nTEST MODE - only ran one slide.")
        print(f"Check patches in {OUTPUT_ROOT}/{TEST_SINGLE_SLIDE}/")
        print("If they look like real tissue, set TEST_SINGLE_SLIDE = None "
              "and re-run for the full batch.")


if __name__ == "__main__":
    main()
