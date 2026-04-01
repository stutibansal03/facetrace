import numpy as np
from PIL import Image
from facetrace.clip_features import compute_clip_probe_scores
from facetrace.fft_features import compute_fft_features
from facetrace.face_crop import detect_and_crop_face, preprocess_image


def extract_features(pil_image: Image.Image, use_face_crop: bool = True) -> np.ndarray:
    if use_face_crop:
        face = detect_and_crop_face(pil_image)
        img = face if face is not None else preprocess_image(pil_image)
    else:
        img = preprocess_image(pil_image)

    clip_scores = compute_clip_probe_scores(img)
    fft_scores = compute_fft_features(img)

    feature_vector = np.array(clip_scores + fft_scores, dtype=np.float32)
    return feature_vector
