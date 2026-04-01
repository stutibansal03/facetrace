"""
Script 02: Train Logistic Regression classifier on extracted features.

Usage:
    python training/02_train_lr.py

Inputs:
    outputs/X_train.npy
    outputs/y_train.npy

Outputs:
    models/facetrace_lr.joblib
    outputs/train_metrics.txt
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from facetrace.classifier import build_pipeline, save_model
from facetrace.config import OUTPUTS_DIR

X = np.load(os.path.join(OUTPUTS_DIR, "X_train.npy"))
y = np.load(os.path.join(OUTPUTS_DIR, "y_train.npy"))

print(f"Loaded X: {X.shape}, y: {y.shape}")
print(f"Class balance — Real: {int((y==0).sum())}, Fake: {int((y==1).sum())}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
y_prob = pipeline.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

report = classification_report(y_val, y_pred, target_names=["Real", "Fake"])

print(f"\nValidation Results:")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"\n{report}")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
metrics_path = os.path.join(OUTPUTS_DIR, "train_metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {auc:.4f}\n\n")
    f.write(report)

save_model(pipeline)
print(f"\nMetrics saved to {metrics_path}")
