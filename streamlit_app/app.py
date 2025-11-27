import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import pandas as pd

from ultralytics import YOLO   # YOLOv8
import torch                   # for patching torch.load
import matplotlib.cm as cm     # for Grad-CAM colormap


# ==========================
# 🔧 PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Aerial Object Detection Dashboard",
    page_icon="🛰️",
    layout="wide",
)

# ==========================
# 🎨 GLOBAL CSS (Admin Panel Style)
# ==========================
st.markdown(
    """
    <style>
    /* ---------- GLOBAL ---------- */
    .stApp {
        background-color: #f3f4f6;
        color: #111827;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #111827;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 0.25rem;
    }
    .sidebar-sub {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }

    .profile-card {
        background: #020617;
        border-radius: 18px;
        padding: 12px 12px 14px 12px;
        border: 1px solid #111827;
        margin-bottom: 14px;
    }
    .profile-name {
        font-size: 15px;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 2px;
    }
    .profile-role {
        font-size: 11px;
        color: #9ca3af;
        margin-bottom: 10px;
    }

    .social-link {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 999px;
        background: #020617;
        border: 1px solid #111827;
        text-decoration: none;
        color: #e5e7eb;
        font-size: 13px;
        margin-bottom: 8px;
        transition: all 0.15s ease-out;
    }
    .social-link:hover {
        background: #111827;
        border-color: #1d4ed8;
        text-decoration: none;
    }
    .social-icon {
        width: 26px;
        height: 26px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 700;
        color: #f9fafb;
    }
    .social-icon.github {
        background: #111827;
        border: 1px solid #6b7280;
    }
    .social-icon.linkedin {
        background: #2563eb;
    }
    .social-label {
        display: flex;
        flex-direction: column;
        gap: 0;
    }
    .social-label-main {
        font-size: 13px;
        font-weight: 600;
    }
    .social-label-sub {
        font-size: 11px;
        color: #9ca3af;
    }

    .sidebar-section-title {
        font-size: 12px;
        color: #6b7280;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    /* ---------- HEADER CARD ---------- */
    .app-header-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 18px 22px;
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.12);
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
    }
    .app-header-left {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .app-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
    }
    .app-subtitle {
        font-size: 13px;
        color: #6b7280;
    }
    .app-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }
    .pill {
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .pill-blue {
        background: rgba(56, 189, 248, 0.1);
        border-color: #38bdf8;
        color: #0369a1;
    }
    .pill-green {
        background: rgba(52, 211, 153, 0.1);
        border-color: #22c55e;
        color: #047857;
    }
    .pill-purple {
        background: rgba(168, 85, 247, 0.1);
        border-color: #a855f7;
        color: #7c3aed;
    }
    .app-header-right {
        text-align: right;
        font-size: 12px;
        color: #6b7280;
    }

    /* ---------- CARDS ---------- */
    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }
    .card h4, .card h3 {
        margin: 0 0 10px 0;
        font-weight: 700;
        color: #111827;
    }
    .card-subtitle {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 10px;
    }

    .result-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 600;
        margin-top: 4px;
        font-size: 13px;
    }
    .bird {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid #38bdf8;
        color: #0369a1;
    }
    .drone {
        background: rgba(248, 113, 113, 0.15);
        border: 1px solid #f97373;
        color: #b91c1c;
    }

    .confidence-box {
        padding: 10px 16px;
        border-radius: 12px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        margin-top: 8px;
        color: #111827;
        font-size: 13px;
    }

    .meta-table td {
        padding: 2px 8px;
        font-size: 12px;
        color: #374151;
    }
    .meta-label {
        color: #9ca3af;
    }

    h3.section-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #0f172a;
    }

    /* ---------- EXTRA: MAKE INPUT CONTROLS VISIBLE ---------- */

    /* Radio buttons: "Upload Image" / "Use Camera" */
    div[role="radiogroup"] label {
        color: #111827 !important;
        font-weight: 600;
    }
    div[role="radiogroup"] label p {
        color: #111827 !important;
    }

    /* File uploader label & helper text */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] span {
        color: #111827 !important;
        font-weight: 500;
    }

    /* Camera preview styling */
    video {
        border-radius: 16px !important;
        border: 2px solid #38bdf8 !important;
    }

    /* Primary buttons */
    button[kind="primary"] {
        background: linear-gradient(45deg, #6366f1, #a855f7);
        border-radius: 12px;
        color: #ffffff !important;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: 0.2s ease;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(45deg, #4f46e5, #9333ea);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================
# 📁 MODEL PATHS
# ==========================
BASE_DIR = Path(__file__).resolve().parent

TRANSFER_LEARNING_PATH = BASE_DIR / "models" / "Checkpoint" / "Transfer_Classification" / "best_transfer_learning.keras"
YOLO_PATH = BASE_DIR / "models" / "Checkpoint" / "YoloV8_Detection" / "best.pt"

IMG_SIZE = (224, 224)

# ==========================
# 🧠 LOAD MODELS
# ==========================
@st.cache_resource
def load_transfer_model():
    return tf.keras.models.load_model(str(TRANSFER_LEARNING_PATH))


@st.cache_resource
def load_yolo_model():
    if not YOLO_PATH.exists():
        st.sidebar.error(f"YOLO file not found at: {YOLO_PATH}")
        return None
    try:
        real_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return real_load(*args, **kwargs)

        torch.load = patched_load
        model = YOLO(str(YOLO_PATH))
        return model
    except Exception as e:
        st.sidebar.error(f"YOLO model failed to load: {e}")
        return None


transfer_model = load_transfer_model()
yolo_model = load_yolo_model()

# ==========================
# 🔎 GRAD-CAM HELPERS
# ==========================
def build_gradcam_model(base_model: tf.keras.Model):
    try:
        last_conv_layer_name = None
        for layer in reversed(base_model.layers):
            try:
                out_shape = layer.output_shape
            except Exception:
                continue
            if hasattr(out_shape, "__len__") and len(out_shape) == 4:
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            return None, None

        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            [base_model.inputs],
            [last_conv_layer.output, base_model.output],
        )
        return grad_model, last_conv_layer_name
    except Exception:
        return None, None


def make_gradcam_heatmap(img_array, grad_model):
    try:
        if grad_model is None:
            return None

        img_array = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads
        heatmap = tf.nn.relu(heatmap)
        denom = tf.reduce_max(heatmap) + 1e-8
        heatmap = heatmap / denom
        return heatmap.numpy()
    except Exception:
        return None


def overlay_gradcam_on_image(heatmap, original_img: Image.Image, alpha: float = 0.4):
    if heatmap is None:
        return None

    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap / 255.0)
    colored = np.uint8(colored * 255)
    heatmap_img = Image.fromarray(colored).resize(original_img.size)
    heatmap_img = heatmap_img.convert("RGBA")

    base = original_img.convert("RGBA")
    overlay = Image.blend(base, heatmap_img, alpha)
    return overlay

# ==========================
# ⚙️ CLASSIFICATION
# ==========================
def classify_image_and_prepare(img: Image.Image):
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0).astype("float32")

    prob_drone = float(transfer_model.predict(arr)[0][0])
    prob_bird = 1.0 - prob_drone

    if prob_drone > 0.5:
        label = "Drone"
        max_conf = prob_drone
    else:
        label = "Bird"
        max_conf = prob_bird

    return label, max_conf, prob_drone, prob_bird, arr


# ==========================
# 🧱 SESSION-STATE
# ==========================
if "conf_history" not in st.session_state:
    st.session_state["conf_history"] = []


# ==========================
# 🧱 SIDEBAR – PERSONAL INTRO
# ==========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Aerial Panel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Bird / Drone monitoring dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="profile-card">
          <div class="profile-name">SVS SUJAL</div>
          <div class="profile-role">Developer · Aerial Object Detection Project</div>
          <a href="https://github.com/SVSS13" target="_blank" class="social-link">
            <div class="social-icon github">GH</div>
            <div class="social-label">
              <span class="social-label-main">GitHub</span>
              <span class="social-label-sub">github.com/SVSS13</span>
            </div>
          </a>
          <a href="https://www.linkedin.com/in/svs-sujal-05219a316" target="_blank" class="social-link">
            <div class="social-icon linkedin">in</div>
            <div class="social-label">
              <span class="social-label-main">LinkedIn</span>
              <span class="social-label-sub">SVS Sujal</span>
            </div>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">Settings</div>', unsafe_allow_html=True)
    use_yolo = st.toggle(
        "Enable YOLOv8 Detection",
        value=True if yolo_model is not None else False,
    )
    st.caption("Toggle YOLO if you want bounding boxes on the image.")

# ==========================
# 🧪 HEADER
# ==========================
st.markdown(
    """
    <div class="app-header-card">
      <div class="app-header-left">
        <div class="app-title">Aerial Object Detection Dashboard</div>
        <div class="app-subtitle">
          Monitor aerial images and classify them as <b>Bird</b> or <b>Drone</b> with detection overlays and insights.
        </div>
        <div class="app-badges">
          <span class="pill pill-blue">Transfer Learning · MobileNetV2</span>
          <span class="pill pill-green">YOLOv8 Detection</span>
          <span class="pill pill-purple">Grad-CAM & Analytics</span>
        </div>
      </div>
      <div class="app-header-right">
        <div>Project Status: <b>Ready</b></div>
        <div>Models loaded on demand with caching</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================
# 📸 MAIN CONTENT
# ==========================
# Row 1: Input + Classification
c1, c2 = st.columns([1.2, 1.3])

uploaded = None
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📤 Input Source</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">You can either upload an aerial image or capture one directly from your camera.</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Choose input method",
        ["Upload Image", "Use Camera"],
        horizontal=True,
    )

    if input_mode == "Upload Image":
        uploaded = st.file_uploader("Select a Bird / Drone image", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("Capture image from camera")

    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">🔍 Classification Result</h3>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Transfer Learning (MobileNetV2) classifier.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ IMPORTANT FIX: use `is not None` instead of `if uploaded`
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    # Classification card content
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">🔍 Classification Result</h3>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Transfer Learning (MobileNetV2) classifier.</div>', unsafe_allow_html=True)

        label, max_conf, prob_drone, prob_bird, input_arr = classify_image_and_prepare(img)
        badge_class = "bird" if label == "Bird" else "drone"

        st.markdown(
            f'<span class="result-badge {badge_class}">{label}</span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="confidence-box">Predicted as <b>{label}</b> with '
            f'overall confidence <b>{max_conf*100:.2f}%</b>.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("##### Class Probabilities")
        st.write(f"Bird: **{prob_bird*100:.2f}%**")
        st.progress(prob_bird)
        st.write(f"Drone: **{prob_drone*100:.2f}%**")
        st.progress(prob_drone)

        st.markdown("---")
        st.markdown("##### Image & Model Info")

        width, height = img.size
        aspect = width / height if height != 0 else 0
        file_size_kb = getattr(uploaded, "size", 0) / 1024 if hasattr(uploaded, "size") else 0

        st.markdown(
            f"""
            <table class="meta-table">
              <tr><td class="meta-label">Resolution</td><td>{width} × {height}</td></tr>
              <tr><td class="meta-label">Aspect Ratio</td><td>{aspect:.2f}</td></tr>
              <tr><td class="meta-label">File Size</td><td>{file_size_kb:.1f} KB</td></tr>
              <tr><td class="meta-label">Mode</td><td>{img.mode}</td></tr>
              <tr><td class="meta-label">Model</td><td>MobileNetV2 Transfer Learning</td></tr>
              <tr><td class="meta-label">Input Size</td><td>{IMG_SIZE[0]} × {IMG_SIZE[1]}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Preview card
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">🖼️ Preview</h3>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # update confidence history
    st.session_state["conf_history"].append(max_conf)

    # Row 2: Grad-CAM + YOLO
    g1, g2 = st.columns([1, 1])

    # Grad-CAM / insights
    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">🔥 Grad-CAM & Model Insights</h3>', unsafe_allow_html=True)

        with st.spinner("Generating Grad-CAM heatmap..."):
            grad_model, _ = build_gradcam_model(transfer_model)
            heatmap = make_gradcam_heatmap(input_arr, grad_model)
            overlay = overlay_gradcam_on_image(heatmap, img)

        if overlay is not None:
            st.image(overlay, caption="Grad-CAM Focus Region", use_container_width=True)
        else:
            st.info(
                "Grad-CAM could not be computed reliably for this saved model, "
                "so showing model insights instead."
            )

            st.markdown("#### 📊 Model Insights (from offline evaluation)")
            st.markdown(
                """
                - **Task**: Binary classification – Bird vs Drone  
                - **Backbone**: MobileNetV2 (pretrained on ImageNet)  
                - **Head**: GlobalAveragePooling + Dense layers  
                - **Input size**: 224 × 224 RGB  
                """
            )

            st.markdown("##### Test Performance (Transfer Learning)")
            st.markdown(
                """
                - **Accuracy**: ~98% on held-out test set  
                - **Bird** – very high precision & recall  
                - **Drone** – very high precision & recall  
                """
            )

            metrics_df = pd.DataFrame(
                {
                    "Class": ["Bird", "Drone"],
                    "Precision": ["≈ 0.98", "≈ 0.98"],
                    "Recall": ["≈ 0.98", "≈ 0.97"],
                    "F1-Score": ["≈ 0.98", "≈ 0.97"],
                }
            )
            st.dataframe(metrics_df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # YOLO detection
    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">🎯 YOLOv8 Detection</h3>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Bounding boxes for detected aerial objects.</div>', unsafe_allow_html=True)

        if use_yolo:
            if yolo_model is not None:
                with st.spinner("Running YOLOv8 on this image..."):
                    results_list = yolo_model.predict(img, verbose=False)
                    result = results_list[0]

                    det_img = img.copy()
                    draw = ImageDraw.Draw(det_img)

                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except Exception:
                        font = ImageFont.load_default()

                    try:
                        names = yolo_model.names
                    except Exception:
                        names = {0: "object"}

                    detected_labels = []

                    if result.boxes is not None:
                        boxes = result.boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        confs = boxes.conf.cpu().numpy()

                        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls_ids, confs):
                            label_name = names.get(int(c), str(int(c)))
                            detected_labels.append(label_name)
                            text = f"{label_name} {cf:.2f}"

                            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

                            bbox = font.getbbox(text)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]

                            text_bg = [x1, y1 - text_height, x1 + text_width + 4, y1]
                            draw.rectangle(text_bg, fill=(0, 255, 0))

                            draw.text((x1 + 2, y1 - text_height), text, fill=(0, 0, 0), font=font)

                    st.image(det_img, caption="YOLOv8 Detection Output", use_container_width=True)

                    if detected_labels:
                        summary = ", ".join(detected_labels)
                        st.markdown(f"**Detected {len(detected_labels)} object(s):** {summary}")
                    else:
                        st.info("No objects detected with current confidence threshold.")
            else:
                st.error("YOLOv8 model could not be loaded. Check YOLO_PATH / file.")
        else:
            st.info("Turn on *Enable YOLOv8 Detection* from the sidebar to see bounding boxes.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Confidence history
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📈 Confidence History (Session)</h3>', unsafe_allow_html=True)

    if st.session_state["conf_history"]:
        conf_list = st.session_state["conf_history"]
        df_hist = pd.DataFrame(
            {
                "Prediction #": list(range(1, len(conf_list) + 1)),
                "Confidence": conf_list,
            }
        ).set_index("Prediction #")

        st.line_chart(df_hist)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("Upload or capture images to build a confidence history.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # empty state
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Getting Started</h3>', unsafe_allow_html=True)
    st.write(
        "Use **Upload Image** or **Use Camera** in the left card to provide an aerial image. "
        "The dashboard will then show classification, YOLO detection and analytics."
    )
    st.markdown('</div>', unsafe_allow_html=True)
