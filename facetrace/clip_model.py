import open_clip
import torch
from facetrace.config import CLIP_MODEL_NAME, CLIP_PRETRAINED

_model = None
_preprocess = None
_tokenizer = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        device = get_device()
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        _model = _model.to(device)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return _model, _preprocess, _tokenizer
