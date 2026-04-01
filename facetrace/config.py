import os

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

PROMPTS = [
    "a real human face",
    "a fake or digitally generated face",
    "a deepfake video frame",
    "an authentic photograph of a person",
    "a manipulated or altered face image",
    "a synthetic or AI-generated face",
    "a natural unmodified human photo",
]

IMAGE_SIZE = 224
FFT_HIGH_FREQ_RATIO = 0.3
VIDEO_FRAME_STEP = 3
VIDEO_MAX_FRAMES = 60
TOP_SUSPICIOUS_FRAMES = 5

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "facetrace_lr.joblib")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
