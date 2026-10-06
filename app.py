"""
app.py  -  Digit Intelligence Suite
====================================
A professional multi-page deep-learning application in ONE file.

Run:  streamlit run app.py
Needs (same folder): cnn_digits_model.pkl

Pages (sidebar navigation):
  Home              - overview, headline metrics, architecture, quick links
  Try It Live       - sample gallery / draw / upload -> CNN prediction
  Model Dashboard   - confusion matrices, P/R/F1, calibration, errors, report download
  Dataset Explorer  - class balance, mean digits, raw browser, augmentation preview
  Documentation     - objective, approach, deliverables, architecture, limitations

Everything (model, preprocessing, augmentation, evaluation, theme) lives in
this file. Evaluation is computed live from the frozen model and cached, so
eval_results.pkl is no longer needed.
"""
import hashlib
import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE, CANVAS_IMPORT_ERROR = True, None
except Exception as _e:  # version mismatches raise non-ImportErrors too
    CANVAS_AVAILABLE, CANVAS_IMPORT_ERROR = False, str(_e)

MODEL_PATH = Path(__file__).parent / "cnn_digits_model.pkl"
DIGITS = [str(i) for i in range(10)]
ACCENT, GOOD, MID, BAD, NAVY = "#FF4B4B", "#21C55D", "#F59E0B", "#EF4444", "#1A3C6E"


# =====================================================================
# 1. THEME
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1300px; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero { background: linear-gradient(180deg,#17181c 0%,#101114 100%);
        border: 1px solid rgba(255,255,255,.08); border-top: 3px solid #FF4B4B;
        border-radius: 10px; padding: 1.5rem 2rem; margin-bottom: 1.4rem; }
.hero-eyebrow { display:inline-flex; align-items:center; gap:.5rem;
        font-family:'JetBrains Mono',monospace; font-size:.75rem; font-weight:600;
        letter-spacing:.14em; text-transform:uppercase; color:#9CA3AF; margin-bottom:.8rem; }
.hero-eyebrow::before { content:''; width:8px; height:8px; border-radius:2px; background:#FF4B4B; }
.hero h1 { margin:0 !important; font-weight:800 !important; letter-spacing:-.02em; color:#F5F5F7; font-size:2.1rem !important; }
.hero p.hero-sub { margin:.6rem 0 0 0; color:#c9ccd1; font-size:1.02rem; line-height:1.55; max-width:800px; }
.metric-row { display:flex; gap:.9rem; flex-wrap:wrap; margin-bottom:1.4rem; }
.metric-card { flex:1; min-width:160px; border:1px solid #e5e7eb; border-radius:10px;
        padding:.9rem 1.1rem; background:#fafafa; }
.mc-label { font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; color:#6b7280; }
.mc-value { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#111827; margin-top:.2rem; }
.mc-sub { font-size:.78rem; color:#6b7280; margin-top:.1rem; }
.result-card { border-radius:18px; padding:1.6rem 1.4rem; text-align:center;
        border:2px solid var(--card-border); background:var(--card-bg); }
.result-digit { font-size:4.5rem; font-weight:800; line-height:1; margin:.2rem 0; }
.result-label { font-size:1rem; opacity:.75; text-transform:uppercase; letter-spacing:.06em; }
.confidence-pill { display:inline-block; padding:.35rem 1rem; border-radius:999px;
        font-weight:700; font-size:1.02rem; margin-top:.5rem; }
.step-badge { display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px;
        border-radius:50%; background:#FF4B4B; color:#fff; font-weight:800; margin-right:10px; flex-shrink:0; }
.step-row { display:flex; align-items:flex-start; margin-bottom:.7rem; }
.step-text { padding-top:4px; font-size:1.02rem; }
.app-footer { text-align:center; opacity:.55; font-size:.85rem; margin-top:2rem; }
</style>
"""


def hero(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="hero"><span class="hero-eyebrow">{eyebrow}</span>'
        f'<h1>{title}</h1><p class="hero-sub">{subtitle}</p></div>',
        unsafe_allow_html=True)


def metric_row(items):
    cards = "".join(
        f'<div class="metric-card"><div class="mc-label">{a}</div>'
        f'<div class="mc-value">{b}</div><div class="mc-sub">{c}</div></div>'
        for a, b, c in items)
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


def steps(*texts):
    st.markdown("".join(
        f'<div class="step-row"><div class="step-badge">{i}</div><div class="step-text">{t}</div></div>'
        for i, t in enumerate(texts, 1)), unsafe_allow_html=True)


def footer():
    st.markdown("<div class='app-footer'>Digit Intelligence Suite &middot; multi-page Streamlit "
                "application &middot; CNN trained from scratch in NumPy</div>", unsafe_allow_html=True)


def confidence_style(c):
    if c >= 0.80:
        return GOOD, "High confidence"
    if c >= 0.50:
        return MID, "Moderate confidence"
    return BAD, "Low confidence"


# =====================================================================
# 2. MODEL  (inference-only CNN, NumPy)
# =====================================================================
def im2col(x, kh, kw, stride=1, pad=0):
    N, H, W, C = x.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    cols = np.zeros((N, out_h, out_w, kh, kw, C), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, i, j, :] = x[:, i:i + stride * out_h:stride, j:j + stride * out_w:stride, :]
    return cols.reshape(N, out_h, out_w, kh * kw * C), out_h, out_w


def conv_forward(x, W, b, stride=1, pad=1):
    N = x.shape[0]
    cols, oh, ow = im2col(x, W.shape[0], W.shape[0], stride, pad)
    out = cols.reshape(N * oh * ow, -1) @ W.reshape(-1, W.shape[-1]) + b
    return out.reshape(N, oh, ow, W.shape[-1])


def maxpool_forward(x, size=2, stride=2):
    N, H, W, C = x.shape
    oh, ow = H // stride, W // stride
    x = x[:, :oh * stride, :ow * stride, :]
    out = np.zeros((N, oh, ow, C), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j, :] = x[:, i * stride:i * stride + size, j * stride:j * stride + size, :].max(axis=(1, 2))
    return out


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class DigitCNN:
    """Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Dense -> ReLU -> Dense -> Softmax."""

    def __init__(self, path):
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self.w = {k: v for k, v in raw.items() if k != "_meta"}
        self.meta = raw.get("_meta")
        if self.meta is None:
            h = hashlib.sha256()
            for k in sorted(self.w):
                h.update(self.w[k].tobytes())
            self.meta = {"augmented_training": False, "weights_hash": h.hexdigest()[:10]}

    def predict_proba(self, x):
        w = self.w
        x = np.maximum(0, conv_forward(x, w["conv1_W"], w["conv1_b"]))
        x = maxpool_forward(x)
        x = np.maximum(0, conv_forward(x, w["conv2_W"], w["conv2_b"]))
        x = maxpool_forward(x)
        x = np.maximum(0, x.reshape(x.shape[0], -1) @ w["fc1_W"] + w["fc1_b"])
        return softmax(x @ w["fc2_W"] + w["fc2_b"])

    def layer_table(self):
        rows = []
        for k, v in self.w.items():
            rows.append({"Parameter": k, "Shape": " × ".join(map(str, v.shape)), "Count": int(v.size)})
        return pd.DataFrame(rows)


# =====================================================================
# 3. PREPROCESSING  (identical for training and inference)
# =====================================================================
def pixels_to_display_image(p, size=192):
    arr = np.clip(p, 0, 16) / 16.0 * 255
    return Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.NEAREST)


def crop_to_ink(gray, pad_frac=0.07):
    arr = gray.astype(np.float64)
    if arr.max() <= arr.min():
        return None
    norm = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    mask = norm > max(25.0, norm.max() * 0.18)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = max(1, int((y1 - y0 + 1) * pad_frac)), max(1, int((x1 - x0 + 1) * pad_frac))
    y0, y1 = max(0, y0 - py), min(norm.shape[0] - 1, y1 + py)
    x0, x1 = max(0, x0 - px), min(norm.shape[1] - 1, x1 + px)
    return norm[y0:y1 + 1, x0:x1 + 1]


def shift_2d(arr, sy, sx):
    out = np.zeros_like(arr)
    h, w = arr.shape
    out[max(0, sy):min(h, h + sy), max(0, sx):min(w, w + sx)] = \
        arr[max(0, -sy):min(h, h - sy), max(0, -sx):min(w, w - sx)]
    return out


def center_by_mass(c):
    total = c.sum()
    if total <= 0:
        return c
    yy, xx = np.mgrid[0:8, 0:8]
    cy, cx = (yy * c).sum() / total, (xx * c).sum() / total
    return shift_2d(c, int(round(3.5 - cy)), int(round(3.5 - cx)))


def dilate3(a):
    return np.maximum.reduce([shift_2d(a, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def erode3(a):
    return np.minimum.reduce([shift_2d(a, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def smart_preprocess_to_8x8(gray):
    cropped = crop_to_ink(gray)
    if cropped is None:
        return np.zeros((8, 8))
    ch, cw = cropped.shape
    scale = 7.0 / max(ch, cw)
    nh = int(np.clip(int(round(ch * scale)), 3.2, 8))
    nw = int(np.clip(int(round(cw * scale)), 3.2, 8))
    resized = np.array(Image.fromarray(cropped.astype(np.uint8)).resize((nw, nh), Image.LANCZOS)).astype(np.float64)
    canvas = np.zeros((8, 8))
    oy, ox = (8 - nh) // 2, (8 - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    canvas = center_by_mass(canvas)
    return canvas / canvas.max() * 16.0 if canvas.max() > 0 else canvas


def to_input(X):
    return (np.clip(X, 0, 16) / 16.0).reshape(-1, 8, 8, 1).astype(np.float32)


def predict_with_tta(model, pixels):
    """Average predictions over ±1px shifts and thicker/thinner strokes."""
    base = np.clip(pixels, 0, 16)
    variants = [base] + [shift_2d(base, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    variants += [np.clip(dilate3(base), 0, 16), np.clip(erode3(base), 0, 16)]
    return model.predict_proba(to_input(np.stack(variants))).mean(axis=0)


# =====================================================================
# 4. AUGMENTATION  (same recipe used at training time)
# =====================================================================
def elastic_warp(arr, rng, upsample):
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    amp = upsample * 0.035
    fy, fx = rng.uniform(1.5, 3.0), rng.uniform(1.5, 3.0)
    py, px = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    dx = amp * np.sin(2 * np.pi * fy * yy / h + py)
    dy = amp * np.sin(2 * np.pi * fx * xx / w + px)
    return arr[np.clip(yy + dy, 0, h - 1).astype(np.int32), np.clip(xx + dx, 0, w - 1).astype(np.int32)]


def augment_one(base, rng, upsample=160):
    img = Image.fromarray((np.clip(base, 0, 16) / 16.0 * 255.0).astype(np.uint8)).resize((upsample, upsample), Image.LANCZOS)
    img = img.rotate(rng.uniform(-22, 22), resample=Image.BILINEAR, fillcolor=0)
    if rng.random() < 0.3:
        if rng.random() < 0.5:
            sx, sy = rng.uniform(0.45, 0.7), rng.uniform(1.3, 1.7)
        else:
            sx, sy = rng.uniform(1.3, 1.7), rng.uniform(0.45, 0.7)
    else:
        sx, sy = rng.uniform(0.65, 1.4), rng.uniform(0.65, 1.4)
    nw, nh = max(4, int(upsample * sx)), max(4, int(upsample * sy))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (upsample, upsample), 0)
    px = (upsample - nw) // 2 if nw <= upsample else 0
    py = (upsample - nh) // 2 if nh <= upsample else 0
    if nw > upsample or nh > upsample:
        left, top = max(0, (nw - upsample) // 2), max(0, (nh - upsample) // 2)
        img = img.crop((left, top, left + min(nw, upsample), top + min(nh, upsample)))
        nw, nh = img.size
        px, py = (upsample - nw) // 2, (upsample - nh) // 2
    canvas.paste(img, (px, py))
    arr = np.array(canvas).astype(np.float64)
    arr = shift_2d(arr, int(rng.integers(-upsample // 7, upsample // 7 + 1)),
                   int(rng.integers(-upsample // 7, upsample // 7 + 1)))
    if rng.random() < 0.6:
        arr = elastic_warp(arr, rng, upsample)
    if rng.random() < 0.55:
        for _ in range(int(rng.integers(1, 3))):
            arr = dilate3(arr)
    elif rng.random() < 0.35:
        arr = erode3(arr)
    arr = np.clip(arr + rng.normal(0, 7, size=arr.shape), 0, 255)
    return smart_preprocess_to_8x8(arr)


# =====================================================================
# 5. CACHED LOADERS + LIVE EVALUATION
# =====================================================================
@st.cache_resource
def get_model():
    return DigitCNN(MODEL_PATH)


@st.cache_data
def get_dataset():
    d = load_digits()
    return d.images, d.target


@st.cache_data(show_spinner="Evaluating the frozen model on clean + augmented validation data…")
def run_evaluation(augs_per_image=6, seed=7):
    """Replaces the old evaluate.py + eval_results.pkl: computed once, cached."""
    model = get_model()
    images, labels = get_dataset()
    rng = np.random.default_rng(seed)

    idx = np.random.default_rng(42).permutation(len(images))      # same 85/15 split as training
    val_idx = idx[int(len(images) * 0.85):]
    vi, vl = images[val_idx], labels[val_idx]

    X_clean = np.stack([smart_preprocess_to_8x8(np.clip(i, 0, 16) / 16.0 * 255.0) for i in vi])
    p_clean = model.predict_proba(to_input(X_clean))
    pred_clean = p_clean.argmax(1)

    imgs, ys = [], []
    for im, lb in zip(vi, vl):
        for _ in range(augs_per_image):
            imgs.append(augment_one(im, rng))
            ys.append(lb)
    X_aug, y_aug = np.stack(imgs), np.array(ys)
    p_aug = model.predict_proba(to_input(X_aug))
    pred_aug = p_aug.argmax(1)
    conf_aug = p_aug[np.arange(len(p_aug)), pred_aug]
    correct = pred_aug == y_aug

    edges = np.linspace(0, 1, 11)
    cal = []
    for b in range(10):
        m = (conf_aug >= edges[b]) & ((conf_aug < edges[b + 1]) if b < 9 else (conf_aug <= edges[b + 1]))
        if m.sum():
            cal.append({"conf": float(conf_aug[m].mean()), "acc": float(correct[m].mean()), "n": int(m.sum())})
    ece = float(sum(c["n"] * abs(c["acc"] - c["conf"]) for c in cal) / len(y_aug))

    wrong = np.where(~correct)[0]
    rng.shuffle(wrong)
    errors = [{"image": X_aug[i], "true": int(y_aug[i]), "pred": int(pred_aug[i]),
               "conf": float(conf_aug[i])} for i in wrong[:12]]

    mean_images = [np.stack([smart_preprocess_to_8x8(np.clip(im, 0, 16) / 16.0 * 255.0)
                             for im in images[labels == d]]).mean(0) for d in range(10)]

    def pack(y, p):
        return {"acc": float((y == p).mean()),
                "cm": confusion_matrix(y, p, labels=list(range(10))),
                "report": classification_report(y, p, labels=list(range(10)), output_dict=True, zero_division=0),
                "n": int(len(y))}

    return {"clean": pack(vl, pred_clean), "aug": pack(y_aug, pred_aug), "calibration": cal, "ece": ece,
            "errors": errors, "mean_images": mean_images,
            "class_counts": [int((labels == d).sum()) for d in range(10)],
            "n_total": int(len(images)), "n_val": int(len(vl)), "augs_per_image": augs_per_image,
            "conf_all": conf_aug, "correct_all": correct}


# =====================================================================
# 6. PAGES
# =====================================================================
def sidebar_badge(meta):
    with st.sidebar:
        st.markdown("#### 🔢 Model status")
        if meta.get("augmented_training"):
            st.success(f"**Robust model loaded** ✓\n\nFingerprint: `{meta.get('weights_hash', '?')}`\n\n"
                       f"Clean {meta.get('clean_val_accuracy', 0)*100:.1f}% · "
                       f"Messy {meta.get('augmented_val_accuracy', 0)*100:.1f}%")
        else:
            st.warning(f"Legacy model — fingerprint `{meta.get('weights_hash', '?')}`")


# ---------------------------- HOME -----------------------------------
def page_home():
    model, meta = get_model(), get_model().meta
    hero("Advanced Multi-Page Streamlit Application", "Digit Intelligence Suite",
         "A deep-learning application with a live classifier, a model-performance dashboard, "
         "a dataset explorer and built-in documentation — one model, one design system.")
    metric_row([
        ("Clean-digit accuracy", f"{meta.get('clean_val_accuracy', 0)*100:.2f}%", "held-out validation split"),
        ("Messy accuracy", f"{meta.get('augmented_val_accuracy', 0)*100:.2f}%", "rotated / scaled / noisy"),
        ("Training epochs", f"{meta.get('epochs', '?')}", f"{meta.get('augments_per_image', '?')}× augmentation"),
        ("Parameters", f"{int(model.layer_table()['Count'].sum()):,}", "trainable weights"),
    ])

    st.markdown("### What's inside")
    c1, c2 = st.columns(2)
    cards = [
        (c1, "🧪 Try It Live", "Classify a sample, draw a digit, or upload a photo — with test-time augmentation and a session history.", "page_live"),
        (c1, "📊 Model Dashboard", "Confusion matrices, precision/recall/F1, calibration, top confusions and misclassified examples.", "page_dash"),
        (c2, "🔍 Dataset Explorer", "Class balance, per-class mean digits, raw browser and a live augmentation preview.", "page_data"),
        (c2, "📄 Documentation", "Objective, approach, deliverables, architecture and known limitations.", "page_docs"),
    ]
    for col, title, text, key in cards:
        with col, st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(text)
            st.page_link(PAGES[key], label=f"Open {title.split(' ', 1)[1]}")

    st.markdown("### Model architecture")
    st.code("Input 8×8×1 → Conv(3×3, same) → ReLU → MaxPool 2×2 → Conv(3×3, same) → ReLU → "
            "MaxPool 2×2 → Flatten → Dense → ReLU → Dense(10) → Softmax", language=None, wrap_lines=True)
    st.dataframe(model.layer_table(), hide_index=True, width="stretch")
    footer()


# ------------------------- TRY IT LIVE -------------------------------
def render_prediction(pixels, source):
    model = get_model()
    with st.spinner("Running the CNN with test-time augmentation…"):
        probs = predict_with_tta(model, pixels)
    pred, conf = int(np.argmax(probs)), float(probs.max())
    color, word = confidence_style(conf)

    key = hashlib.md5(np.round(pixels, 2).tobytes()).hexdigest()
    hist = st.session_state.setdefault("history", [])
    last_logged = st.session_state.setdefault("last_logged", {})
    # Log once per distinct input per source. Comparing only with hist[-1] re-logged a
    # still-displayed sample result every time another tab added a newer entry.
    if last_logged.get(source) != key:
        hist.append({"key": key, "Source": source, "Prediction": pred, "Confidence": round(conf * 100, 1)})
        last_logged[source] = key

    st.markdown("#### Result")
    c1, c2, c3 = st.columns([1, 1.1, 1.3])
    with c1:
        st.image(pixels_to_display_image(pixels, 220), caption="What the model saw (8×8)", width="stretch")
    with c2:
        st.markdown(
            f'<div class="result-card" style="--card-border:{color}55; --card-bg:{color}14;">'
            f'<div class="result-label">Predicted digit</div>'
            f'<div class="result-digit" style="color:{color};">{pred}</div>'
            f'<span class="confidence-pill" style="background:{color}22; color:{color};">'
            f'{conf*100:.1f}% · {word}</span></div>', unsafe_allow_html=True)
        if conf < 0.5:
            st.caption("⚠ Not very sure — try a bolder, more centered digit.")
    with c3:
        st.markdown("**Top 3 guesses**")
        for rank, d in enumerate(np.argsort(probs)[::-1][:3]):
            st.markdown(f"{'🥇🥈🥉'[rank]} **{d}** — {probs[d]*100:.1f}%")
            st.progress(float(np.clip(probs[d], 0, 1)))

    with st.expander("Full probability breakdown"):
        df = pd.DataFrame({"Digit": DIGITS, "Probability": probs, "Top": [d == pred for d in range(10)]})
        st.altair_chart(
            alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Digit:N", sort=None),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(alt.datum.Top, alt.value(ACCENT), alt.value("#5B7FDE88")),
                tooltip=["Digit", alt.Tooltip("Probability:Q", format=".1%")]).properties(height=240),
            width="stretch")


def page_live():
    sample_images, sample_labels = get_dataset()
    hero("Interactive", "Try It Live",
         "Classify a digit three ways: pick a labeled sample, draw one by hand, or upload a photo. "
         "Every mode runs the same crop → scale → center pipeline plus test-time augmentation.")

    t_samples, t_draw, t_upload = st.tabs(["🖼️ Sample Digits", "✏️ Draw Your Own", "📤 Upload a Photo"])

    # ---- samples
    with t_samples:
        steps("Filter by digit (optional), then click <b>Use this</b> under a thumbnail.",
              "Press <b>Predict</b> to see what the model thinks.")
        f_col, s_col = st.columns([2, 3])
        with f_col:
            dfilter = st.selectbox("Filter gallery by true digit", ["All digits"] + DIGITS)
        cand = np.arange(len(sample_images)) if dfilter == "All digits" else np.where(sample_labels == int(dfilter))[0]
        st.session_state.setdefault("gallery_seed", 0)
        if st.session_state.get("selected_sample") not in cand:
            st.session_state.selected_sample = int(cand[0])
            st.session_state.show_sample_result = False
        with s_col:
            st.write("")
            if st.button("🔀 Shuffle gallery"):
                st.session_state.gallery_seed += 1
        show = np.random.RandomState(st.session_state.gallery_seed).choice(cand, min(8, len(cand)), replace=False)
        cols = st.columns(4)
        for i, idx in enumerate(show):
            with cols[i % 4], st.container(border=True):
                st.image(pixels_to_display_image(sample_images[idx], 150), width="stretch")
                st.caption(f"True label: **{sample_labels[idx]}**")
                if st.button("Use this", key=f"use_{idx}", width="stretch"):
                    st.session_state.selected_sample = int(idx)
                    st.session_state.show_sample_result = False
        st.divider()
        sel = st.session_state.selected_sample
        st.markdown(f"**Selected** — index `{sel}`, true label **{sample_labels[sel]}**")
        l, r = st.columns([1, 3])
        l.image(pixels_to_display_image(sample_images[sel], 200), width="stretch")
        with r:
            st.write("")
            if st.button("🚀 Predict", key="p_sample", type="primary", width="stretch"):
                st.session_state.show_sample_result = True
        if st.session_state.get("show_sample_result"):
            st.divider()
            render_prediction(sample_images[sel].copy(), "Sample")

    # ---- draw
    with t_draw:
        if CANVAS_AVAILABLE:
            steps("Draw a single digit (0–9) with mouse or touch.", "Press <b>Predict</b> when happy with it.")
            d_col, p_col = st.columns([3, 2])
            canvas_result, err = None, None
            with d_col:
                brush = st.slider("Brush size", 8, 40, 22)
                try:
                    canvas_result = st_canvas(fill_color="white", stroke_width=brush, stroke_color="white",
                                              background_color="black", height=380, width=380,
                                              drawing_mode="freedraw", return_image_data=True, key="canvas")
                except Exception as e:
                    err = str(e)
                go = st.button("🚀 Predict", key="p_canvas", type="primary") if canvas_result is not None else False
            if err:
                st.warning(f"⚠ The drawing canvas couldn't load (version mismatch with Streamlit). Detail: `{err}`")
            else:
                drawn = None
                data = canvas_result.image_data
                if data is not None and data[:, :, :3].sum() > 0:
                    gray = np.array(Image.fromarray(data.astype(np.uint8)).convert("L"))
                    drawn = smart_preprocess_to_8x8(gray)
                with p_col:
                    st.markdown("#### Live model-input preview")
                    if drawn is not None and drawn.sum() > 0:
                        st.image(pixels_to_display_image(drawn, 300), width="stretch")
                    else:
                        st.info("Start drawing to see a live preview.")
                if go:
                    if drawn is None or drawn.sum() == 0:
                        st.warning("The canvas looks empty — draw a digit first!")
                    else:
                        st.divider()
                        render_prediction(drawn, "Drawn")
        else:
            st.info("Drawing canvas not installed (`pip install streamlit-drawable-canvas`). "
                    "Use this pixel-brush grid instead:")
            st.session_state.setdefault("grid", np.zeros((8, 8), dtype=np.int32))
            strength = st.slider("Brush strength", 2, 16, 8, key="fb_brush")
            g_l, g_r = st.columns([3, 1])
            with g_r:
                if st.button("Load example '0'", width="stretch"):
                    st.session_state.grid = sample_images[np.where(sample_labels == 0)[0][0]].astype(np.int32).copy()
                if st.button("Clear grid", width="stretch"):
                    st.session_state.grid = np.zeros((8, 8), dtype=np.int32)
                st.image(pixels_to_display_image(st.session_state.grid), width="stretch")
            with g_l:
                for r_ in range(8):
                    rc = st.columns(8)
                    for c_ in range(8):
                        v = int(st.session_state.grid[r_, c_])
                        if rc[c_].button("⬛" if v == 0 else ("⬜" if v >= 12 else "◾"), key=f"cell_{r_}_{c_}"):
                            st.session_state.grid[r_, c_] = min(16, v + strength)
                            st.rerun()
            if st.button("🚀 Predict", key="p_grid", type="primary"):
                if st.session_state.grid.sum() == 0:
                    st.warning("The grid is empty — click some cells first!")
                else:
                    st.divider()
                    render_prediction(st.session_state.grid.astype(np.float64), "Grid")

    # ---- upload
    with t_upload:
        steps("Upload a clear photo or scan with a single handwritten digit.",
              "Toggle <b>Invert colors</b> if the preview looks wrong.")
        u_col, o_col = st.columns([2, 1])
        invert = o_col.checkbox("Invert colors", value=True, help="ON for a dark digit on a light background.")
        up = u_col.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if up is not None:
            gray = np.array(Image.open(up).convert("L")).astype(np.float64)
            if invert:
                gray = 255.0 - gray
            crop, arr = crop_to_ink(gray), smart_preprocess_to_8x8(gray)
            st.divider()
            a, b, c = st.columns(3)
            a.markdown("**Original upload**")
            a.image(up, width="stretch")
            b.markdown("**Auto-cropped**")
            if crop is not None:
                b.image(Image.fromarray(crop.astype(np.uint8)).resize((240, 240), Image.NEAREST), width="stretch")
            else:
                b.warning("No clear digit found — try the invert toggle.")
            c.markdown("**Model input**")
            c.image(pixels_to_display_image(arr, 240), width="stretch")
            if st.button("🚀 Predict", key="p_upload", type="primary"):
                if arr.sum() == 0:
                    st.warning("No digit detected — try another photo.")
                else:
                    st.divider()
                    render_prediction(arr, "Upload")
        else:
            st.info("📤 Upload a PNG or JPG of a handwritten digit to get started.")

    hist = st.session_state.get("history", [])
    if hist:
        st.markdown("### Session history")
        h_df = pd.DataFrame(hist).drop(columns="key")
        h_df.index = h_df.index + 1
        hc1, hc2 = st.columns([4, 1])
        hc1.dataframe(h_df.iloc[::-1], width="stretch")
        with hc2:
            st.download_button("⬇ Download CSV", h_df.to_csv(index_label="#"), "prediction_history.csv", "text/csv",
                               width="stretch")
            if st.button("Clear history", width="stretch"):
                st.session_state.history = []
                st.session_state.last_logged = {}
                st.session_state.show_sample_result = False   # otherwise the sample would be re-logged
                st.rerun()


# ------------------------ MODEL DASHBOARD ----------------------------
def confusion_chart(cm, title):
    df = pd.DataFrame(cm, index=DIGITS, columns=DIGITS).reset_index().melt(
        id_vars="index", var_name="Predicted", value_name="Count").rename(columns={"index": "True"})
    enc = dict(x=alt.X("Predicted:N", sort=DIGITS), y=alt.Y("True:N", sort=DIGITS))
    rect = alt.Chart(df).mark_rect().encode(
        **enc, color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")), tooltip=["True", "Predicted", "Count"])
    text = alt.Chart(df).mark_text(baseline="middle").encode(
        **enc, text="Count:Q",
        color=alt.condition(alt.datum.Count > cm.max() / 2, alt.value("white"), alt.value("black")))
    return (rect + text).properties(height=380, title=title)


def per_class_chart(report, title):
    rows = [{"Digit": d, "Precision": report[d]["precision"], "Recall": report[d]["recall"], "F1": report[d]["f1-score"]}
            for d in DIGITS]
    df = pd.DataFrame(rows).melt(id_vars="Digit", var_name="Metric", value_name="Score")
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Digit:N", sort=DIGITS), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Metric:N", scale=alt.Scale(range=[ACCENT, NAVY, GOOD])), xOffset="Metric:N",
        tooltip=["Digit", "Metric", alt.Tooltip("Score:Q", format=".3f")]).properties(height=320, title=title)


def top_confusions(cm, k=5):
    pairs = [(i, j, int(cm[i, j])) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: -t[2])
    return pd.DataFrame([{"True": a, "Predicted as": b, "Count": c} for a, b, c in pairs[:k]])


def page_dash():
    meta = get_model().meta
    res = run_evaluation()
    hero("Dashboard", "Model Performance Dashboard",
         "Confusion matrices, per-class precision / recall / F1, confidence calibration and worked "
         "misclassification examples — computed live from the frozen model.")
    metric_row([
        ("Clean accuracy", f"{res['clean']['acc']*100:.2f}%", f"{res['n_val']} held-out images"),
        ("Messy accuracy", f"{res['aug']['acc']*100:.2f}%", f"{res['aug']['n']} augmented variants"),
        ("Macro F1 (messy)", f"{res['aug']['report']['macro avg']['f1-score']:.3f}", "average over 10 classes"),
        ("Calibration error", f"{res['ece']*100:.2f}%", "ECE on messy set (lower = better)"),
    ])
    st.caption(f"Training-time numbers stored in the model file: clean "
               f"{meta.get('clean_val_accuracy', 0)*100:.2f}% / messy {meta.get('augmented_val_accuracy', 0)*100:.2f}%. "
               "Close agreement with the recomputed numbers above indicates the model generalizes.")

    tab_c, tab_m, tab_t = st.tabs(["Clean validation set", "Messy / augmented set", "Per-class table"])
    for tab, key, name in ((tab_c, "clean", "clean"), (tab_m, "aug", "messy")):
        with tab:
            a, b = st.columns(2)
            a.altair_chart(confusion_chart(res[key]["cm"], f"Confusion matrix — {name}"), width="stretch")
            b.altair_chart(per_class_chart(res[key]["report"], f"Precision / Recall / F1 — {name}"), width="stretch")
            st.markdown("**Most frequent confusions**")
            tc = top_confusions(res[key]["cm"])
            if tc.empty:
                st.success("No misclassifications in this set.")
            else:
                st.dataframe(tc, hide_index=True, width="stretch")
    with tab_t:
        rows = []
        for d in DIGITS:
            rc, ra = res["clean"]["report"][d], res["aug"]["report"][d]
            rows.append({"Digit": d, "Clean P": rc["precision"], "Clean R": rc["recall"], "Clean F1": rc["f1-score"],
                         "Messy P": ra["precision"], "Messy R": ra["recall"], "Messy F1": ra["f1-score"],
                         "Messy support": int(ra["support"])})
        tbl = pd.DataFrame(rows)
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if c not in ("Digit", "Messy support")}),
                     hide_index=True, width="stretch")
        st.download_button("⬇ Download per-class report (CSV)", tbl.to_csv(index=False), "per_class_report.csv", "text/csv")

    st.markdown("### Confidence calibration")
    st.caption("Bars should follow the dashed diagonal: when the model says 70% sure, it should be right ~70% of the time.")
    cal = pd.DataFrame(res["calibration"]).rename(columns={"conf": "Mean confidence", "acc": "Actual accuracy", "n": "Count"})
    bars = alt.Chart(cal).mark_bar(color=ACCENT, opacity=0.85, size=22).encode(
        x=alt.X("Mean confidence:Q", scale=alt.Scale(domain=[0, 1], nice=False), title="Mean confidence"),
        y=alt.Y("Actual accuracy:Q", scale=alt.Scale(domain=[0, 1], nice=False), title="Actual accuracy"),
        tooltip=[alt.Tooltip("Mean confidence:Q", format=".2f"), alt.Tooltip("Actual accuracy:Q", format=".2f"), "Count"])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#999").encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1], nice=False), title="Mean confidence"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1], nice=False), title="Actual accuracy"))
    cc1, cc2 = st.columns(2)
    cc1.altair_chart((bars + diag).properties(height=300, title="Reliability diagram"), width="stretch")
    hdf = pd.DataFrame({"Confidence": res["conf_all"], "Outcome": np.where(res["correct_all"], "Correct", "Wrong")})
    cc2.altair_chart(
        alt.Chart(hdf).mark_bar(opacity=0.85).encode(
            x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=20)), y=alt.Y("count():Q", title="Predictions"),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Correct", "Wrong"], range=[GOOD, BAD]))
        ).properties(height=300, title="Confidence distribution: correct vs wrong"), width="stretch")

    st.markdown("### Misclassification examples")
    st.caption("Genuine errors from the messy evaluation set — not cherry-picked successes.")
    cols = st.columns(4)
    for i, ex in enumerate(res["errors"][:8]):
        with cols[i % 4]:
            st.image(pixels_to_display_image(ex["image"], 130), width="stretch")
            st.caption(f"True **{ex['true']}** → predicted **{ex['pred']}** ({ex['conf']*100:.0f}%)")
    footer()


# ------------------------ DATASET EXPLORER ---------------------------
def page_data():
    images, labels = get_dataset()
    res = run_evaluation()
    hero("Exploration", "Dataset Explorer",
         "The UCI handwritten-digits dataset the model was trained on, and the augmentation pipeline "
         "that turns 1,797 clean 8×8 digits into something closer to real freehand input.")
    metric_row([("Images", f"{res['n_total']:,}", "8×8 grayscale"), ("Classes", "10", "digits 0–9"),
                ("Per class", f"{min(res['class_counts'])}–{max(res['class_counts'])}", "reasonably balanced"),
                ("Pixel range", "0–16", "4-bit intensity")])

    st.markdown("### Class balance")
    dist = pd.DataFrame({"Digit": DIGITS, "Count": res["class_counts"]})
    st.altair_chart(alt.Chart(dist).mark_bar(color=NAVY, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Digit:N", sort=None), y="Count:Q", tooltip=["Digit", "Count"]).properties(height=260), width="stretch")

    st.markdown("### What a “typical” digit looks like")
    st.caption("Per-class mean image after the same crop/scale/center preprocessing the model sees at inference.")
    cols = st.columns(5)
    for d in range(10):
        with cols[d % 5]:
            st.image(pixels_to_display_image(res["mean_images"][d], 110), width="stretch")
            st.caption(f"Digit {d}")

    st.markdown("### Browse the raw dataset")
    f = st.selectbox("Filter by true digit", ["All digits"] + DIGITS, key="explorer_filter")
    idxs = np.arange(len(images)) if f == "All digits" else np.where(labels == int(f))[0]
    if st.button("🔀 Shuffle"):
        st.session_state.explorer_seed = st.session_state.get("explorer_seed", 0) + 1
    show = np.random.RandomState(st.session_state.get("explorer_seed", 0)).choice(idxs, min(10, len(idxs)), replace=False)
    cols = st.columns(5)
    for i, idx in enumerate(show):
        with cols[i % 5]:
            st.image(pixels_to_display_image(images[idx], 110), width="stretch")
            st.caption(f"index {idx} · label {labels[idx]}")

    st.markdown("### Augmentation pipeline preview")
    st.caption("Rotation, independent-axis scaling, translation, elastic warp, stroke thickening/thinning and noise — "
               "the same `augment_one` used at training time.")
    a1, a2 = st.columns([1, 3])
    with a1:
        sidx = st.number_input("Sample index", 0, len(images) - 1, int(np.where(labels == 8)[0][0]), 1)
        nv = st.slider("Number of variants", 3, 8, 6)
        regen = st.button("🎲 Generate variants", type="primary")
    with a2:
        st.markdown("**Original**")
        st.image(pixels_to_display_image(images[sidx], 110))
        if regen or "aug_cache" not in st.session_state or st.session_state.get("aug_idx") != sidx:
            rng = np.random.default_rng()
            st.session_state.aug_cache = [augment_one(images[sidx], rng) for _ in range(8)]
            st.session_state.aug_idx = sidx
        st.markdown("**Augmented variants**")
        vc = st.columns(nv)
        for i, v in enumerate(st.session_state.aug_cache[:nv]):
            vc[i].image(pixels_to_display_image(v, 100), width="stretch")
    footer()


# --------------------------- DOCUMENTATION ---------------------------
def page_docs():
    hero("Reference", "Documentation", "Objective, approach, deliverables, architecture and known limitations.")
    st.markdown("### Objective")
    st.write("To develop a professional deep-learning application with multiple pages and interactive visualizations.")
    st.markdown("### Reference dataset")
    st.write("UCI Optical Recognition of Handwritten Digits (scikit-learn `load_digits`) — an image classification "
             "dataset of 1,797 8×8 grayscale digits across 10 classes.")

    st.markdown("### Approach → where it is implemented")
    st.markdown("""
| Requirement | Implementation |
|---|---|
| Create a multi-page architecture | `st.navigation` with five pages (Home, Try It Live, Model Dashboard, Dataset Explorer, Documentation) in this one file |
| Add dashboards and visualizations | Altair confusion matrices, grouped P/R/F1 bars, reliability diagram, confidence histogram, class balance, mean digits |
| Display model performance metrics | Accuracy, macro-F1, expected calibration error, per-class table, top confusions |
| Integrate charts and reports | Interactive tooltips, plus CSV downloads of the per-class report and prediction history |
| Improve navigation and user experience | Sidebar navigation, page-link cards, model-status badge, tabs, session history, spinners, empty-state guidance |
""")
    st.markdown("### Deliverables")
    st.markdown("- **Multi-page Streamlit application** — `app.py`\n"
                "- **Dashboard visualizations** — *Model Dashboard* and *Dataset Explorer* pages\n"
                "- **Project documentation** — this page")

    st.markdown("### Methodology")
    st.write("A small CNN (Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Dense → ReLU → Dense → Softmax) trained "
             "from scratch in NumPy with hand-derived backpropagation. The identical crop-to-ink, independent-axis "
             "scale and center-by-mass pipeline is used for training augmentation and for every live input, so "
             "training and inference never drift apart. Live predictions average 11 test-time-augmentation variants "
             "(±1px shifts, thicker/thinner strokes).")
    st.markdown("### Project files")
    st.markdown("""
| File | Role |
|---|---|
| `app.py` | The entire application (this file) |
| `cnn_digits_model.pkl` | Trained weights + metadata |
| `requirements.txt` | Python dependencies |
""")
    st.markdown("### Known limitations")
    st.markdown("""
- **Small, low-resolution data** — ~1,800 images at 8×8 is not production-grade.
- **Extreme proportions** — very tall/wide digits become ambiguous when compressed to 8×8.
- **Stylistic mismatch** — ornate or heavily cursive digits differ from the plain training style.
- **Evaluation recency** — metrics are computed from the model file currently loaded and cached; restart the app
  after replacing `cnn_digits_model.pkl`.
""")
    footer()


# =====================================================================
# 7. ENTRY POINT
# =====================================================================
PAGES = {}


def main():
    st.set_page_config(page_title="Digit Intelligence Suite", page_icon="🔢", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: `{MODEL_PATH.name}`. Put it in the same folder as app.py.")
        st.stop()

    PAGES["page_home"] = st.Page(page_home, title="Home", icon="🏠", url_path="home", default=True)
    PAGES["page_live"] = st.Page(page_live, title="Try It Live", icon="🧪", url_path="live")
    PAGES["page_dash"] = st.Page(page_dash, title="Model Dashboard", icon="📊", url_path="dashboard")
    PAGES["page_data"] = st.Page(page_data, title="Dataset Explorer", icon="🔍", url_path="dataset")
    PAGES["page_docs"] = st.Page(page_docs, title="Documentation", icon="📄", url_path="docs")

    nav = st.navigation(list(PAGES.values()))
    sidebar_badge(get_model().meta)
    nav.run()


if __name__ == "__main__":
    main()
