import cv2
import numpy as np
from PIL import Image
from facetrace.config import IMAGE_SIZE

try:
    import mediapipe as mp
    _mp_face = mp.solutions.face_detection
    _MEDIAPIPE_OK = True
except AttributeError:
    _MEDIAPIPE_OK = False


def detect_and_crop_face(pil_image, padding=0.2):
    if not _MEDIAPIPE_OK:
        return preprocess_image(pil_image)

    image_np = np.array(pil_image.convert("RGB"))
    h, w = image_np.shape[:2]

    with _mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    if not results.detections:
        return preprocess_image(pil_image)

    det = results.detections[0]
    bbox = det.location_data.relative_bounding_box

    x = int((bbox.xmin - padding * bbox.width) * w)
    y = int((bbox.ymin - padding * bbox.height) * h)
    bw = int((1 + 2 * padding) * bbox.width * w)
    bh = int((1 + 2 * padding) * bbox.height * h)

    x, y = max(0, x), max(0, y)
    bw, bh = min(bw, w - x), min(bh, h - y)

    face = image_np[y:y+bh, x:x+bw]
    if face.size == 0:
        return preprocess_image(pil_image)

    return Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def preprocess_image(pil_image):
    return pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)