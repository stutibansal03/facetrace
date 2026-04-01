import cv2
import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from facetrace.features import extract_features
from facetrace.classifier import predict
from facetrace.config import VIDEO_FRAME_STEP, VIDEO_MAX_FRAMES, TOP_SUSPICIOUS_FRAMES


def infer_video(video_path: str, pipeline: Pipeline) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    sampled = 0
    frame_probs = []
    frame_images = []

    while sampled < VIDEO_MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % VIDEO_FRAME_STEP == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                feats = extract_features(pil, use_face_crop=True)
                _, prob = predict(pipeline, feats)
            except Exception:
                prob = 0.0
            frame_probs.append((frame_idx, prob))
            frame_images.append(pil.copy())
            sampled += 1
        frame_idx += 1

    cap.release()

    if not frame_probs:
        return {"verdict": "unknown", "confidence": 0.0, "frame_probs": [], "top_frames": []}

    probs = [p for _, p in frame_probs]
    mean_p = float(np.mean(probs))
    std_p = float(np.std(probs))
    top5_mean = float(np.mean(sorted(probs, reverse=True)[:5]))
    spike_count = int(sum(1 for p in probs if p > 0.7))

    verdict = "FAKE" if mean_p > 0.5 else "AUTHENTIC"
    confidence = float(top5_mean if verdict == "FAKE" else 1.0 - mean_p)

    sorted_frames = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top_frame_indices = sorted_frames[:TOP_SUSPICIOUS_FRAMES]
    top_frames = [(frame_probs[i][0], probs[i], frame_images[i]) for i in top_frame_indices]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "mean_prob": mean_p,
        "std_prob": std_p,
        "top5_mean": top5_mean,
        "spike_count": spike_count,
        "frame_probs": frame_probs,
        "top_frames": top_frames,
    }
