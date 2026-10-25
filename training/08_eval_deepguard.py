import os, sys
import numpy as np, torch, torch.nn.functional as F, joblib
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from facetrace.clip_model import load_clip_model, get_device
from facetrace.config import DATA_DIR, OUTPUTS_DIR, MODEL_PATH
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

ROOT = os.path.join(DATA_DIR, "deepguard")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
model, preprocess, _ = load_clip_model()
dev = get_device()


def embed(folder):
    files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in EXT)
    out = []
    for i in tqdm(range(0, len(files), 64), desc=Path(folder).name):
        imgs = []
        for f in files[i:i + 64]:
            try:
                imgs.append(Image.open(f).convert("RGB"))
            except Exception:
                pass
        x = torch.stack([preprocess(im) for im in imgs]).to(dev)
        with torch.no_grad():
            out.append(F.normalize(model.encode_image(x), dim=-1).cpu().numpy())
    return np.concatenate(out)


E, y = [], []
for name, lab in [("real", 0), ("fake", 1)]:
    folder = os.path.join(ROOT, name)
    if not os.path.isdir(folder):
        sys.exit(f"Missing folder {folder} - wait for the download to finish.")
    e = embed(folder)
    E.append(e)
    y.append(np.full(len(e), lab))
E, y = np.concatenate(E), np.concatenate(y)
np.save(f"{OUTPUTS_DIR}/emb_deepguard.npy", E)
np.save(f"{OUTPUTS_DIR}/y_deepguard.npy", y)
print("images:", len(y), "| real:", int((y == 0).sum()), "fake:", int((y == 1).sum()))

clf = joblib.load(MODEL_PATH)
pred = clf.predict(E)
prob = clf.predict_proba(E)[:, 1]
print("\nXHLULU-TRAINED MODEL ON DEEPGUARD (face-swap, never seen in training)")
print(f"accuracy={accuracy_score(y, pred):.4f}  F1={f1_score(y, pred):.4f}  ROC-AUC={roc_auc_score(y, prob):.4f}")
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f"real correctly called real: {tn/(tn+fp):.1%}   fake correctly called fake: {tp/(tp+fn):.1%}")