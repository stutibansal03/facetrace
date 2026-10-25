import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from facetrace.config import OUTPUTS_DIR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

TAG = sys.argv[1] if len(sys.argv) > 1 else "n5000"
load = lambda n: np.load(f"{OUTPUTS_DIR}/{n}_{TAG}.npy")
emb, fft, y, size, ext = load("emb"), load("fft"), load("y"), load("size"), load("ext")
print("samples:", len(y), "| real:", int((y == 0).sum()), "fake:", int((y == 1).sum()))

print("\n--- format check (real vs fake should look alike) ---")
for lab, name in [(0, "real"), (1, "fake")]:
    m = y == lab
    u, c = np.unique(ext[m], return_counts=True)
    print(name, "extensions:", dict(zip(u, c)),
          "| distinct sizes:", len(np.unique(size[m], axis=0)),
          "| e.g.", np.unique(size[m], axis=0)[:2].tolist())

idx = np.arange(len(y))
dev_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)

sets = {"fft": fft, "clip": emb, "clip+fft": np.hstack([emb, fft])}
models = {}
for C in [0.01, 0.1, 1, 10]:
    models[f"LR C={C}"] = lambda C=C: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=3000))
for C in [1, 10]:
    models[f"SVM-RBF C={C}"] = lambda C=C: make_pipeline(
        StandardScaler(), SVC(C=C, kernel="rbf"))

print("\n--- 5-fold CV accuracy (dev set only) ---")
for sname, X in sets.items():
    for mname, make in models.items():
        s = cross_val_score(make(), X[dev_idx], y[dev_idx], cv=5, scoring="accuracy", n_jobs=-1)
        print(f"{sname:9s} {mname:14s} acc={s.mean():.4f} ±{s.std():.4f}", flush=True)