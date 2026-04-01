"""
Script 03: Evaluate FaceTrace on FF++ video dataset.

Usage:
    python training/03_ffpp_eval.py

Expects FF++ frames in:
    data/ffpp/<manipulation>/<real|fake>/<video_id>/frame_*.jpg

Outputs:
    outputs/ffpp_eval_results.csv
    outputs/ffpp_metrics.txt
"""

import os
import sys
import csv
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from facetrace.features import extract_features
from facetrace.classifier import load_model, predict
from facetrace.config import DATA_DIR, OUTPUTS_DIR

FFPP_DIR = os.path.join(DATA_DIR, "ffpp")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

pipeline = load_model()

results = []
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def eval_video_dir(video_dir: Path, true_label: int, manipulation: str) -> dict | None:
    frames = sorted(
        [f for f in video_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    )[:60]

    if not frames:
        return None

    probs = []
    for frame_path in frames:
        try:
            pil = Image.open(frame_path).convert("RGB")
            feats = extract_features(pil, use_face_crop=True)
            _, prob = predict(pipeline, feats)
            probs.append(prob)
        except Exception:
            continue

    if not probs:
        return None

    mean_prob = float(np.mean(probs))
    pred_label = int(mean_prob > 0.5)

    return {
        "video": video_dir.name,
        "manipulation": manipulation,
        "true_label": true_label,
        "pred_label": pred_label,
        "mean_prob": mean_prob,
        "n_frames": len(probs),
    }


ffpp_path = Path(FFPP_DIR)
if not ffpp_path.exists():
    print(f"FF++ directory not found: {FFPP_DIR}")
    print("Please download and place FF++ data in data/ffpp/")
    sys.exit(0)

for manip_dir in ffpp_path.iterdir():
    if not manip_dir.is_dir():
        continue
    manipulation = manip_dir.name
    for split_name, label in [("real", 0), ("fake", 1)]:
        split_dir = manip_dir / split_name
        if not split_dir.exists():
            continue
        for vid_dir in split_dir.iterdir():
            if not vid_dir.is_dir():
                continue
            result = eval_video_dir(vid_dir, label, manipulation)
            if result:
                results.append(result)
                print(f"  {manipulation}/{split_name}/{vid_dir.name}: p={result['mean_prob']:.3f}")

if not results:
    print("No results found. Check your FF++ data structure.")
    sys.exit(0)

csv_path = os.path.join(OUTPUTS_DIR, "ffpp_eval_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

y_true = [r["true_label"] for r in results]
y_pred = [r["pred_label"] for r in results]
y_prob = [r["mean_prob"] for r in results]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print(f"\nFF++ Evaluation Results:")
print(f"  Videos evaluated : {len(results)}")
print(f"  Accuracy         : {acc:.4f}")
print(f"  F1 Score         : {f1:.4f}")
print(f"  ROC-AUC          : {auc:.4f}")

metrics_path = os.path.join(OUTPUTS_DIR, "ffpp_metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Videos evaluated : {len(results)}\n")
    f.write(f"Accuracy         : {acc:.4f}\n")
    f.write(f"F1 Score         : {f1:.4f}\n")
    f.write(f"ROC-AUC          : {auc:.4f}\n")

print(f"\nResults saved to {csv_path}")
print(f"Metrics saved to {metrics_path}")
