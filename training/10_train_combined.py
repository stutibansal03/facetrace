import os, sys, numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from facetrace.config import OUTPUTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

L = lambda n: np.load(f"{OUTPUTS_DIR}/{n}.npy")
Xa, ya = L("emb_full"), L("y_full")
Xb, yb = L("emb_deepguard"), L("y_deepguard")

ia_tr, ia_te = train_test_split(np.arange(len(ya)), test_size=0.15, random_state=42, stratify=ya)
ib_tr, ib_te = train_test_split(np.arange(len(yb)), test_size=0.15, random_state=42, stratify=yb)

REP = 4
Xtr = np.vstack([Xa[ia_tr]] + [Xb[ib_tr]] * REP)
ytr = np.concatenate([ya[ia_tr]] + [yb[ib_tr]] * REP)
print("training on", len(ytr), "images")

m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=200, random_state=0))
m.fit(ytr.reshape(-1, 1) * 0 + Xtr if False else Xtr, ytr)

for name, X, y, te in [("xhlulu (StyleGAN)", Xa, ya, ia_te), ("DeepGuard (face-swap)", Xb, yb, ib_te)]:
    p = m.predict(X[te]); pr = m.predict_proba(X[te])[:, 1]
    print(f"{name}: test acc={accuracy_score(y[te], p):.4f}  AUC={roc_auc_score(y[te], pr):.4f}  (n={len(te)})")

joblib.dump(m, "models/facetrace_mlp2.joblib")
print("saved models/facetrace_mlp2.joblib")
