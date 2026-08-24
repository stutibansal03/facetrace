# FaceTrace — Deepfake Detection Tool

🔗 **Live App**: https://facetrace.streamlit.app

A Python-based deepfake detection pipeline using CLIP semantic probes, FFT frequency artifact features, and a Logistic Regression classifier. Includes a Streamlit web UI for image and video analysis.

---

## Project Structure

```
facetrace/
├── facetrace/               # Core library
│   ├── config.py            # Paths, constants, prompt definitions
│   ├── clip_model.py        # OpenCLIP model loading (cached singleton)
│   ├── clip_features.py     # CLIP probe cosine similarity (7 features)
│   ├── fft_features.py      # FFT frequency artifact features (2 features)
│   ├── face_crop.py         # MediaPipe face detection + crop
│   ├── features.py          # Combined 9D feature vector
│   ├── classifier.py        # Scikit-learn pipeline (train/load/predict)
│   └── video_infer.py       # Frame-level video inference + aggregation
├── training/
│   ├── 01_feature_extract_dataset.py   # Extract features from xhlulu dataset
│   ├── 02_train_lr.py                  # Train + evaluate Logistic Regression
│   └── 03_ffpp_eval.py                 # Evaluate on FF++ video subset
├── app/
│   └── streamlit_app.py     # Streamlit web UI (image + video modes)
├── data/
│   ├── xhlulu/              # Image dataset (real/ and fake/ subfolders)
│   └── ffpp/                # FF++ video frames (optional)
├── models/                  # Saved model (facetrace_lr.joblib)
├── outputs/                 # Feature arrays, metrics, evaluation results
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# or
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS (Apple Silicon):** PyTorch with MPS support is auto-detected. No extra steps needed.

---

## Data Preparation

### Image Dataset (xhlulu)

Place cropped face images in:
```
data/xhlulu/real/    ← real face images (jpg/png)
data/xhlulu/fake/    ← deepfake images (jpg/png)
```

### Video Dataset (FF++) — Optional

Download a subset of FaceForensics++ (compression c23, 2 manipulations: DeepFakes + FaceSwap). Extract frames and place them as:
```
data/ffpp/<manipulation>/real/<video_id>/frame_001.jpg
data/ffpp/<manipulation>/fake/<video_id>/frame_001.jpg
```

---

## Training Pipeline

### Step 1 — Extract features from the image dataset

```bash
python training/01_feature_extract_dataset.py
```

Outputs: `outputs/X_train.npy` and `outputs/y_train.npy`

### Step 2 — Train the classifier

```bash
python training/02_train_lr.py
```

Outputs: `models/facetrace_lr.joblib` and `outputs/train_metrics.txt`

### Step 3 (Optional) — Evaluate on FF++ videos

```bash
python training/03_ffpp_eval.py
```

Outputs: `outputs/ffpp_eval_results.csv` and `outputs/ffpp_metrics.txt`

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

Open https://facetrace.streamlit.app in your browser (or run locally at http://localhost:8501).

---

## Feature Pipeline

For each image or video frame:

| Step | Method | Output |
|------|--------|--------|
| Face detection | MediaPipe FaceDetection | Cropped face (224×224) |
| CLIP probe scores | OpenCLIP ViT-B-32, 7 prompts | 7 cosine similarity scores |
| FFT artifact features | 2D FFT magnitude spectrum | 2 frequency energy ratios |
| Combined feature vector | Concatenation | 9D float vector |

### CLIP Prompts Used

1. "a real human face"
2. "a fake or digitally generated face"
3. "a deepfake video frame"
4. "an authentic photograph of a person"
5. "a manipulated or altered face image"
6. "a synthetic or AI-generated face"
7. "a natural unmodified human photo"

### FFT Features

- **High-Freq Energy Ratio** — proportion of spectral energy in the high-frequency band (outer 30% of the spectrum)
- **Mid-High Band Ratio** — proportion in the mid-to-high band (30–70% of radius)

---

## Video Inference

- Samples every 3rd frame, capped at 60 frames per video
- Per-frame: face crop → 9D features → Logistic Regression probability
- Video-level verdict: mean probability > 0.5 → FAKE
- Reports: mean/std/top-5 probabilities, spike count, most suspicious frames

---

## Evaluation Metrics

- Accuracy
- F1 Score (macro/binary)
- ROC-AUC

---

## Architecture Diagram

```
Image/Video Input
       │
       ▼
 Face Detection (MediaPipe)
       │
       ▼
 Resize to 224×224
       │
       ├─────────────────────┐
       ▼                     ▼
 CLIP Probes (7D)      FFT Features (2D)
       │                     │
       └──────────┬──────────┘
                  ▼
         9D Feature Vector
                  │
                  ▼
  StandardScaler + Logistic Regression
                  │
                  ▼
         Verdict + Probability
                  │
                  ▼
     Explanation + Visualization
```

---

## Notes

- The model file (`models/facetrace_lr.joblib`) is generated by training — it is **not** included in the project until you run the training scripts.
- CLIP model weights are downloaded automatically from OpenAI on first run (~350 MB).
- MediaPipe face detection works best with front-facing portrait images.
