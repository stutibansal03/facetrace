"""
Script 01: Extract features from the xhlulu dataset (and optionally FF++ frames).

Usage:
    python training/01_feature_extract_dataset.py

Outputs:
    outputs/X_train.npy  — feature matrix (N, 9)
    outputs/y_train.npy  — labels (N,) where 0=real, 1=fake
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from facetrace.features import extract_features
from facetrace.config import DATA_DIR, OUTPUTS_DIR

XHLULU_REAL = os.path.join(DATA_DIR, "xhlulu", "real", "real")
XHLULU_FAKE = os.path.join(DATA_DIR, "xhlulu", "fake", "fake")
OUTPUT_X = os.path.join(OUTPUTS_DIR, "X_train.npy")
OUTPUT_Y = os.path.join(OUTPUTS_DIR, "y_train.npy")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images_from_dir(directory: str, label: int) -> tuple[list, list]:
    features_list = []
    labels_list = []
    path = Path(directory)
    files = [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Found {len(files)} images in {directory} (label={label})")

    for i, img_path in enumerate(files):
        try:
            pil = Image.open(img_path).convert("RGB")
            feats = extract_features(pil, use_face_crop=False)
            features_list.append(feats)
            labels_list.append(label)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(files)}")
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")

    return features_list, labels_list


def main():
    all_features = []
    all_labels = []

    real_feats, real_labels = load_images_from_dir(XHLULU_REAL, label=0)
    all_features.extend(real_feats)
    all_labels.extend(real_labels)

    fake_feats, fake_labels = load_images_from_dir(XHLULU_FAKE, label=1)
    all_features.extend(fake_feats)
    all_labels.extend(fake_labels)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print(f"\nSaved X ({X.shape}) to {OUTPUT_X}")
    print(f"Saved y ({y.shape}) to {OUTPUT_Y}")
    print(f"Class balance — Real: {int((y==0).sum())}, Fake: {int((y==1).sum())}")


if __name__ == "__main__":
    main()
