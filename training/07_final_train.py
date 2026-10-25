import os, sys
import numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from facetrace.config import OUTPUTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
load = lambda n: np.load(f"{OUTPUTS_DIR}/{n}_{TAG}.npy")
X, y = load("emb"), load("y")

idx = np.arange(len(y))
dev_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
print(f"training on {len(dev_idx)} images, testing on {len(test_idx)} never-seen images")

model = make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(512,), alpha=1e-3, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=8, max_iter=200, random_state=0))
model.fit(X[dev_idx], y[dev_idx])

pred = model.predict(X[test_idx])
prob = model.predict_proba(X[test_idx])[:, 1]
acc = accuracy_score(y[test_idx], pred)
f1 = f1_score(y[test_idx], pred)
auc = roc_auc_score(y[test_idx], prob)
report = classification_report(y[test_idx], pred, target_names=["Real", "Fake"])
print(f"\nTEST  accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}\n{report}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/facetrace_mlp.joblib")
with open(f"{OUTPUTS_DIR}/final_test_metrics.txt", "w") as f:
    f.write(f"Accuracy : {acc:.4f}\nF1 Score : {f1:.4f}\nROC-AUC  : {auc:.4f}\n\n{report}")
print("saved models/facetrace_mlp.joblib")