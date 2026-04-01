import torch
import torch.nn.functional as F
from PIL import Image
from facetrace.clip_model import load_clip_model, get_device
from facetrace.config import PROMPTS


def compute_clip_probe_scores(pil_image: Image.Image) -> list[float]:
    model, preprocess, tokenizer = load_clip_model()
    device = get_device()

    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    text_tokens = tokenizer(PROMPTS).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarities = (image_features @ text_features.T).squeeze(0)

    return similarities.cpu().tolist()
