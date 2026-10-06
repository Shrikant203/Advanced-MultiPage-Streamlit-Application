# Digit Intelligence Suite

A professional, multi-page deep-learning web application built with **Streamlit**. It wraps a **CNN trained from scratch in NumPy** (hand-derived backpropagation, no TensorFlow/PyTorch) that recognizes handwritten digits, and adds a live classifier, a model-performance dashboard, a dataset explorer and built-in documentation.

> **Objective:** To develop a professional deep learning application with multiple pages and interactive visualizations.

---

## Features

| Page | What it does |
|---|---|
| **Home** | Headline metrics, architecture summary, parameter table and navigation cards |
| **Try It Live** | Classify a labeled sample, **draw** a digit on a canvas, or **upload** a photo. Shows top-3 guesses, a full probability chart and a downloadable session history |
| **Model Dashboard** | Confusion matrices, precision / recall / F1 per class, macro-F1, calibration error (ECE), reliability diagram, confidence histogram, top confusions, misclassified examples, CSV report download |
| **Dataset Explorer** | Class balance, per-class mean digit images, raw dataset browser, live preview of the augmentation pipeline |
| **Documentation** | Objective, dataset, approach, deliverables, methodology and known limitations, inside the app |

## Model

```
Input 8×8×1 → Conv(3×3) → ReLU → MaxPool(2×2) → Conv(3×3) → ReLU → MaxPool(2×2)
            → Flatten → Dense → ReLU → Dense(10) → Softmax
```

- **Dataset:** UCI Optical Recognition of Handwritten Digits (`sklearn.datasets.load_digits`): 1,797 images, 8×8 grayscale, 10 classes.
- **Training:** 24 epochs with ~18× on-the-fly augmentation (rotation, independent-axis scaling, translation, elastic warp, stroke thickening/thinning, noise).
- **Preprocessing (same for training and inference):** crop to ink → scale → center by mass, so real drawings look like the training data.
- **Test-time augmentation:** live predictions average 11 variants (±1px shifts, thicker/thinner strokes).

### Performance

| Evaluation set | Accuracy |
|---|---|
| Clean held-out validation (270 images) | ~98.5% |
| Messy / augmented validation (1,620 variants) | ~90% |

Metrics are recomputed live from the model file the first time the dashboard or explorer is opened (roughly 5–10 seconds depending on the machine, then cached), so the dashboard always reflects the loaded weights.

## Project Structure

```
.
├── app.py                  # The entire application (all pages, model, preprocessing, evaluation)
├── cnn_digits_model.pkl    # Trained weights + metadata
├── requirements.txt        # Python dependencies
└── README.md
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/<username>/<your-repo>.git
cd <repo>
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

Requires **Streamlit 1.53 or newer** (the app uses `width="stretch"`, which older versions reject, and `streamlit-drawable-canvas` needs 1.53+).

**4. Run the app**
```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Tech Stack

Python · Streamlit · NumPy · pandas · Altair · scikit-learn (dataset + metrics only) · Pillow · streamlit-drawable-canvas

## ⚠️ Known Limitations

- **Small, low-resolution data:** about 1,800 images at 8×8 is not production-grade.
- **Extreme proportions:** very tall or wide digits become ambiguous once compressed to 8×8.
- **Stylistic mismatch:** ornate or heavily cursive digits differ from the plain training style.
- **Model changes:** restart the app after replacing `cnn_digits_model.pkl` so cached metrics refresh.
- **Drawing canvas:** if `streamlit-drawable-canvas` is incompatible with your Streamlit version, the app falls back to a clickable pixel grid.

## ☁️ Deploying

The app can be deployed free on [Streamlit Community Cloud](https://streamlit.io/cloud): push this repo to GitHub, choose the repo, and set the main file to `app.py`.
