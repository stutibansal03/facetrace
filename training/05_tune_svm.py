import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from facetrace.config import OUTPUTS_DIR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TAG = sys.argv[1] if len(sys.argv) > 1 else "n5000"
load = lambda n: np.load(f"{OUTPUTS_DIR}/{n}_{TAG}.npy")
X, y = load("emb"), load("y")
idx = np.arange(len(y))
dev_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = {"svc__C": [10, 30, 100], "svc__gamma": ["scale", 0.0005, 0.002]}
gs = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
gs.fit(X[dev_idx], y[dev_idx])

res = sorted(zip(gs.cv_results_["mean_test_score"], gs.cv_results_["params"]), key=lambda t: -t[0])
for s, p in res:
    print(f"{s:.4f}  {p}")
print("\nBEST:", gs.best_params_, f"{gs.best_score_:.4f}")