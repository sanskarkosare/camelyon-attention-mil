"""
Download the 50-slide CAMELYON16 subset (25 normal + 25 tumor, clean annotations
only) into the existing camelyon16_subset/normal and camelyon16_subset/tumor folders.

Run with: python download_camelyon_subset.py
Requires: pip install boto3
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

BUCKET = "camelyon-dataset"
IMAGES_PREFIX = "CAMELYON16/images/"

OUT_DIR = "camelyon16_subset"
NORMAL_OUT = os.path.join(OUT_DIR, "normal")
TUMOR_OUT = os.path.join(OUT_DIR, "tumor")

# 25 normal slides (no exclusions in this range)
NORMAL_IDS = list(range(1, 26))

# 25 tumor slides, skipping the non-exhaustively-annotated ones
# (010, 015, 018, 020, 025 fall in this range and are excluded)
TUMOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17,
             19, 21, 22, 23, 24, 26, 27, 28, 30, 31]

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def fetch(name, out_dir):
    out_path = os.path.join(out_dir, f"{name}.tif")
    if os.path.exists(out_path):
        print(f"[skip] {name}.tif already exists")
        return
    key = f"{IMAGES_PREFIX}{name}.tif"
    print(f"[downloading] {name}.tif")
    s3.download_file(BUCKET, key, out_path)


def main():
    os.makedirs(NORMAL_OUT, exist_ok=True)
    os.makedirs(TUMOR_OUT, exist_ok=True)

    print(f"Fetching {len(NORMAL_IDS)} normal + {len(TUMOR_IDS)} tumor slides...\n")

    for i in NORMAL_IDS:
        fetch(f"normal_{i:03d}", NORMAL_OUT)

    for i in TUMOR_IDS:
        fetch(f"tumor_{i:03d}", TUMOR_OUT)

    print("\nDone.")


if __name__ == "__main__":
    main()
