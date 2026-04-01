import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from facetrace.config import MODEL_PATH


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict(pipeline: Pipeline, features: np.ndarray) -> tuple[int, float]:
    features_2d = features.reshape(1, -1)
    label = int(pipeline.predict(features_2d)[0])
    prob = float(pipeline.predict_proba(features_2d)[0][1])
    return label, prob
