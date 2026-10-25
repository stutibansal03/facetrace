import os, sys, io, glob
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from facetrace.clip_model import load_clip_model, get_device
from facetrace.config import OUTPUTS_DIR

model, preprocess, _ = load_clip_model()
dev = get_device()
E, Y = [], []
for f in sorted(glob.glob("data/diffusion/data/*.parquet")):
    df = pd.read_parquet(f)
    print(f, len(df), flush=True)
    for i in tqdm(range(0, len(df), 64)):
        imgs, labs = [], []
        for j in range(i, min(i + 64, len(df))):
            try:
                imgs.append(Image.open(io.BytesIO(df["image"].iloc[j]["bytes"])).convert("RGB"))
                labs.append(1 - int(df["label"].iloc[j]))
            except Exception:
                pass
        x = torch.stack([preprocess(im) for im in imgs]).to(dev)
        with torch.no_grad():
            e = F.normalize(model.encode_image(x), dim=-1).cpu().numpy()
        E.append(e)
        Y.extend(labs)
    del df
E, Y = np.concatenate(E), np.array(Y)
np.save(f"{OUTPUTS_DIR}/emb_diffusion.npy", E)
np.save(f"{OUTPUTS_DIR}/y_diffusion.npy", Y)
print("saved", len(Y), "| real:", int((Y == 0).sum()), "fake:", int((Y == 1).sum()))
