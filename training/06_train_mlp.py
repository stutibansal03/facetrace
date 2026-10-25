import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from facetrace.config import OUTPUTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

TAG = sys.argv[1] if len(sys.argv) > 1 else "n5000"
load = lambda n: np.load(f"{OUTPUTS_DIR}/{n}_{TAG}.npy")
X, y = load("emb"), load("y")

idx = np.arange(len(y))
dev_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
tr_idx, val_idx = train_test_split(dev_idx, test_size=0.15, random_state=0, stratify=y[dev_idx])
print(f"train {len(tr_idx)} | val {len(val_idx)} | (test {len(test_idx)} untouched)")

for hidden, alpha in [((512,), 1e-3), ((512, 128), 1e-3), ((1024, 256), 1e-2)]:
    mlp = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=8, max_iter=200, random_state=0))
    t = time.time()
    mlp.fit(X[tr_idx], y[tr_idx])
    acc = mlp.score(X[val_idx], y[val_idx])
    print(f"hidden={hidden} alpha={alpha}  val acc={acc:.4f}  ({time.time() - t:.0f}s)", flush=True)