"""
Download 100 additional CAMELYON16 slides to go from 50 -> 150 total.
Adding: normal_026 to normal_075 (50 slides)
        50 clean tumor slides not yet downloaded

Save to:  camelyon-mil\download_additional_100.py
Run from: D:\Sk Work\Placement-Things\Machine Learning\ML projects\camelyon-mil
Command:  python download_additional_100.py
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

BUCKET        = "camelyon-dataset"
IMAGES_PREFIX = "CAMELYON16/images/"
NORMAL_OUT    = "camelyon16_subset/normal"
TUMOR_OUT     = "camelyon16_subset/tumor"

# 50 additional normal slides — no exclusions in 026-075
NORMAL_IDS = list(range(26, 76))

# Slides with non-exhaustive annotations — skip these
SKIP_TUMOR = {10,15,18,20,25,29,33,34,44,46,51,54,55,56,67,79,85,92,95,110}

# Already downloaded tumor slides
ALREADY_HAVE = {1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,19,
                21,22,23,24,26,27,28,30,31}

# Pick next 50 clean tumor slides in order
TUMOR_IDS = [
    i for i in range(1, 112)
    if i not in SKIP_TUMOR and i not in ALREADY_HAVE
][:50]

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
    os.makedirs(TUMOR_OUT,  exist_ok=True)

    print(f"Normal IDs to download ({len(NORMAL_IDS)}): "
          f"{NORMAL_IDS[0]}...{NORMAL_IDS[-1]}")
    print(f"Tumor IDs to download ({len(TUMOR_IDS)}): {TUMOR_IDS}\n")

    for i in NORMAL_IDS:
        fetch(f"normal_{i:03d}", NORMAL_OUT)

    for i in TUMOR_IDS:
        fetch(f"tumor_{i:03d}", TUMOR_OUT)

    print("\nDone. You now have 150 slides total (75 normal + 75 tumor).")
    print("Next: run extract_patches.py with TEST_SINGLE_SLIDE = None")
    print("      (it skips slides already extracted, only tiles new ones)")


if __name__ == "__main__":
    main()
