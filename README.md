cat > README.md << 'EOF'
# FaceTrace: Deepfake Face Detection

Live demo: https://facetrace.streamlit.app

FaceTrace classifies a face image (or video frames) as real or AI-generated. It uses a frozen CLIP ViT-B/32 image encoder to produce a 512-dimensional embedding, and a small neural network (MLP) classifier trained on top of it.

## Results

Each number is accuracy on a held-out test split (15%) that was never used for training or tuning.

| Test set | Fake type | Test images | Accuracy | ROC-AUC |
|---|---|---|---|---|
| xhlulu | StyleGAN faces | 15,000 | 97.3% | 0.996 |
| DeepGuard | Face-swap (InsightFace) | 1,628 | 95.3% | 0.989 |
| Diffusion set | Midjourney / DALL-E / Stable Diffusion | 900 | 91.7% | 0.972 |

How the model improved:

| Version | What changed | xhlulu | DeepGuard |
|---|---|---|---|
| v1 (original) | 7 CLIP text-prompt scores + 2 FFT ratios, logistic regression | 63.1% | not tested |
| v2 | Full 512-d CLIP embedding + MLP | 97.4% | 53.2% (never seen face-swap) |
| v3 (current) | Trained on all three fake types together | 97.3% | 95.3% |

## What we learned

- Replacing 7 text-prompt similarity scores with the full 512-d CLIP embedding was the biggest gain (63% to about 95% in cross-validation).
- A 48-bin FFT frequency profile added nothing measurable (94.8% vs 94.6%, within noise), so it was dropped.
- Cross-dataset testing exposed a real weakness: the StyleGAN-only model scored 53.2% on face-swap fakes (it caught only 9.1% of them). Training on several kinds of fakes fixed this.
- An MLP beat an RBF SVM once there was enough data (100,000 images).

## Limitations

- Results are per dataset. Accuracy on images from a generator or a pipeline not represented in training can be much lower.
- On two modern AI portraits tested by hand, the final model flagged one and missed the other (a small, heavily compressed WhatsApp image). Compression removes the artifacts the model relies on.
- The diffusion test set contains general images, not only faces, so 91.7% partly reflects "AI-generated look vs ordinary photo".
- The DeepGuard real and fake images may come from the same source faces. A random split can therefore make its score somewhat optimistic.
- The classifier can output near-0% or near-100% probabilities on images unlike its training data. Treat the probability as a model score, not as certainty.

## Datasets

- xhlulu: 50,000 real + 50,000 fake faces (StyleGAN), 256x256 JPG
- DeepGuard (Hugging Face: Sowaiba01/deepguard-dataset): about 5,400 real + 5,400 face-swap fakes
- Midjourney / DALL-E / Stable Diffusion vs real (Hugging Face: julienlucas/midjourney-dalle-sd-dataset): 6,000 images

Datasets are not included in this repository.

## Project structure

    app/streamlit_app.py      Streamlit web app (image and video)
    facetrace/                CLIP loading, face crop, feature extraction, classifier, video inference
    training/                 Feature extraction, experiments, and training scripts
    models/facetrace_mlp3.joblib   Final trained classifier

## Run locally

    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

## Reproduce training

    python3 training/01b_extract_features.py            # xhlulu embeddings
    python3 training/08_eval_deepguard.py                # DeepGuard embeddings
    python3 training/13_extract_diffusion.py             # diffusion embeddings
    python3 training/14_train_three.py                   # trains and evaluates the final model

## Future work

- Add more recent generators (Flux, newer Midjourney versions) and face-only diffusion data
- Train with compression and resize augmentation so the model survives re-saved images
- Add probability calibration and an "uncertain" output for unfamiliar images
- Fine-tune the last CLIP layers
EOF
