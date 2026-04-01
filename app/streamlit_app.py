import sys
import os
import json
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from facetrace.features import extract_features
from facetrace.classifier import load_model, predict
from facetrace.video_infer import infer_video
from facetrace.config import PROMPTS, MODEL_PATH

st.set_page_config(
    page_title="FaceTrace — Deepfake Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("FaceTrace — Deepfake Detection")
st.markdown("*Powered by CLIP probes + FFT artifact analysis + Logistic Regression*")

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model not found at `{MODEL_PATH}`. "
            "Please run `training/01_feature_extract_dataset.py` then `training/02_train_lr.py` first."
        )
        return None
    return load_model()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Detection Mode", ["Image (Photo)", "Video"])
    st.markdown("---")
    st.markdown("**About FaceTrace**")
    st.markdown(
        "Uses CLIP semantic probes (7 features) and FFT frequency "
        "artifact analysis (2 features) fed into a Logistic Regression "
        "classifier for deepfake detection."
    )

pipeline = get_model()

if mode == "Image (Photo)":
    st.header("Image Deepfake Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded and pipeline:
        col1, col2 = st.columns([1, 2])

        with col1:
            pil = Image.open(uploaded).convert("RGB")
            st.image(pil, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                feats = extract_features(pil, use_face_crop=True)
                label, prob = predict(pipeline, feats)

            verdict = "FAKE" if label == 1 else "AUTHENTIC"
            color = "red" if label == 1 else "green"

            st.markdown(f"### Verdict: :{color}[{verdict}]")
            st.metric("Fake Probability", f"{prob:.1%}")
            st.progress(float(prob))

            clip_scores = feats[:7].tolist()
            fft_scores = feats[7:].tolist()

            st.markdown("#### CLIP Probe Scores")
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ["tomato" if i % 2 == 1 else "steelblue" for i in range(len(PROMPTS))]
            ax.barh(
                [p[:35] + "..." if len(p) > 35 else p for p in PROMPTS],
                clip_scores,
                color=colors,
            )
            ax.set_xlabel("Cosine Similarity")
            ax.set_title("CLIP Probe Similarities")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### FFT Artifact Features")
            st.table({
                "Feature": ["High-Freq Energy Ratio", "Mid-High Band Ratio"],
                "Value": [f"{fft_scores[0]:.4f}", f"{fft_scores[1]:.4f}"],
            })

            explanation = {
                "verdict": verdict,
                "fake_probability": round(prob, 4),
                "clip_probe_scores": {p: round(s, 4) for p, s in zip(PROMPTS, clip_scores)},
                "fft_features": {
                    "high_freq_energy_ratio": round(fft_scores[0], 4),
                    "mid_high_band_ratio": round(fft_scores[1], 4),
                },
                "top_contributing_prompt": PROMPTS[int(np.argmax(np.abs(clip_scores)))],
            }

            st.markdown("#### Explanation JSON")
            st.json(explanation)

            json_bytes = json.dumps(explanation, indent=2).encode()
            st.download_button(
                "Download Explanation (JSON)",
                data=json_bytes,
                file_name="facetrace_explanation.json",
                mime="application/json",
            )

elif mode == "Video":
    st.header("Video Deepfake Detection")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded and pipeline:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            progress_bar = st.progress(0, text="Analyzing video frames...")
            with st.spinner("Running frame-level analysis..."):
                result = infer_video(tmp_path, pipeline)
            progress_bar.progress(100, text="Done!")

            verdict = result["verdict"]
            confidence = result["confidence"]
            color = "red" if verdict == "FAKE" else "green"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Verdict", verdict)
            col2.metric("Confidence", f"{confidence:.1%}")
            col3.metric("Mean Fake Prob", f"{result['mean_prob']:.1%}")
            col4.metric("High-Risk Frames", result["spike_count"])

            st.markdown(f"### Overall: :{color}[{verdict}]")

            st.markdown("#### Frame-wise Fake Probability")
            frame_probs = result["frame_probs"]
            if frame_probs:
                indices = [fp[0] for fp in frame_probs]
                probs = [fp[1] for fp in frame_probs]

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(indices, probs, color="tomato", linewidth=1.5, label="Fake Prob")
                ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Threshold")
                ax.fill_between(indices, probs, 0.5, where=[p > 0.5 for p in probs],
                                alpha=0.2, color="red", label="Suspected fake")
                ax.set_xlabel("Frame Index")
                ax.set_ylabel("Fake Probability")
                ax.set_title("Frame-Level Analysis")
                ax.legend()
                ax.set_ylim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("#### Most Suspicious Frames")
            top_frames = result.get("top_frames", [])
            if top_frames:
                cols = st.columns(len(top_frames))
                for col, (frame_idx, prob, img) in zip(cols, top_frames):
                    col.image(img, caption=f"Frame {frame_idx}\nP(fake)={prob:.1%}", use_container_width=True)

            explanation = {
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "mean_fake_prob": round(result["mean_prob"], 4),
                "std_fake_prob": round(result["std_prob"], 4),
                "top5_mean_prob": round(result["top5_mean"], 4),
                "high_risk_spike_count": result["spike_count"],
                "frames_analyzed": len(frame_probs),
            }

            st.markdown("#### Explanation JSON")
            st.json(explanation)

            json_bytes = json.dumps(explanation, indent=2).encode()
            st.download_button(
                "Download Explanation (JSON)",
                data=json_bytes,
                file_name="facetrace_video_explanation.json",
                mime="application/json",
            )

        finally:
            os.unlink(tmp_path)
