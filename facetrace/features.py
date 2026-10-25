import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facetrace.clip_model import load_clip_model, get_device
from facetrace.face_crop import detect_and_crop_face


def extract_features(pil_image: Image.Image, use_face_crop: bool = True) -> np.ndarray:
    img = pil_image.convert("RGB")
    if use_face_crop:
        face = detect_and_crop_face(img)
        if face is not None:
            img = face

    model, preprocess, _ = load_clip_model()
    device = get_device()
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = F.normalize(model.encode_image(x), dim=-1)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)