import os, sys, numpy as np, joblib
sys.path.insert(0, os.getcwd())
from facetrace.config import OUTPUTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

L = lambda n: np.load(f"{OUTPUTS_DIR}/{n}.npy")
sets = {"xhlulu (StyleGAN)": (L("emb_full"), L("y_full"), 1),
        "DeepGuard (face-swap)": (L("emb_deepguard"), L("y_deepguard"), 4),
        "Diffusion (MJ/DALLE/SD)": (L("emb_diffusion"), L("y_diffusion"), 8)}
Xtr, ytr, tests = [], [], {}
for name, (X, y, rep) in sets.items():
    tr, te = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
    Xtr += [X[tr]] * rep
    ytr += [y[tr]] * rep
    tests[name] = (X[te], y[te])
Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
print("training on", len(ytr), "images", flush=True)
m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=200, random_state=0))
m.fit(Xtr, ytr)
for name, (X, y) in tests.items():
    p = m.predict(X); pr = m.predict_proba(X)[:, 1]
    print(f"{name}: test acc={accuracy_score(y, p):.4f}  AUC={roc_auc_score(y, pr):.4f}  (n={len(y)})")
joblib.dump(m, "models/facetrace_mlp3.joblib")
print("saved models/facetrace_mlp3.joblib")
if len(sys.argv) > 1:
    from PIL import Image
    from facetrace.features import extract_features
    for p in sys.argv[1:]:
        im = Image.open(p).convert("RGB")
        for crop in (False, True):
            e = extract_features(im, use_face_crop=crop).reshape(1, -1)
            print(os.path.basename(p), "| crop:", crop, "| fake prob:", round(float(m.predict_proba(e)[0][1]), 2))
