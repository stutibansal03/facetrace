import os, sys
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from facetrace.clip_model import load_clip_model, get_device
from facetrace.config import DATA_DIR, OUTPUTS_DIR

REAL = os.path.join(DATA_DIR, "xhlulu", "real", "real")
FAKE = os.path.join(DATA_DIR, "xhlulu", "fake", "fake")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BINS = 48
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None   # images per class
from facetrace.config import CLIP_MODEL_NAME
TAG = (f"n{LIMIT}" if LIMIT else "full") + ("" if CLIP_MODEL_NAME == "ViT-B-32" else "_" + CLIP_MODEL_NAME)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

model, preprocess, _ = load_clip_model()
dev = get_device()
print("device:", dev)

def fft_profile(img):
    gray = np.array(img.convert("L"), dtype=np.float32)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    m = r < 1
    idx = (r[m] * BINS).astype(int)
    sums = np.bincount(idx, weights=mag[m], minlength=BINS)
    cnts = np.maximum(np.bincount(idx, minlength=BINS), 1)
    return sums / cnts

def run(folder, label):
    files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in EXT)
    if LIMIT:
        files = random.Random(0).sample(files, min(LIMIT, len(files)))
    E, P, S, X = [], [], [], []
    for i in tqdm(range(0, len(files), 64), desc=f"label={label}"):
        imgs = []
        for f in files[i:i + 64]:
            try:
                imgs.append((Image.open(f).convert("RGB"), f.suffix.lower()))
            except Exception:
                pass
        x = torch.stack([preprocess(im) for im, _ in imgs]).to(dev)
        with torch.no_grad():
            e = F.normalize(model.encode_image(x), dim=-1).cpu().numpy()
        E.append(e)
        P.extend(fft_profile(im) for im, _ in imgs)
        S.extend(im.size for im, _ in imgs)
        X.extend(ext for _, ext in imgs)
    E = np.concatenate(E)
    return E, np.array(P, dtype=np.float32), np.array(S), np.array(X), np.full(len(E), label)

Er, Pr, Sr, Xr, yr = run(REAL, 0)
Ef, Pf, Sf, Xf, yf = run(FAKE, 1)
for name, a, b in [("emb", Er, Ef), ("fft", Pr, Pf), ("size", Sr, Sf), ("ext", Xr, Xf), ("y", yr, yf)]:
    np.save(f"{OUTPUTS_DIR}/{name}_{TAG}.npy", np.concatenate([a, b]))
print("saved", TAG, len(yr) + len(yf), "images")