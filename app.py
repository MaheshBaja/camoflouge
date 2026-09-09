"""
Camouflaged Object Detection (COD) - Streamlit Web Application
Architecture: U-Net + ResNet34 trained on COD10K-v3
"""

import io
import os
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

from model import load_model, get_default_device
from preprocessing import (
    preprocess_image,
    postprocess_prediction,
    create_highlight_overlay,
    TARGET_SIZE,
)


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Camouflaged Object Detection",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #00E676;
    }
    .metric-label {
        font-size: 13px;
        color: #B0BEC5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Header subtitle */
    .hero-subtitle {
        font-size: 1.15rem;
        color: #90CAF9;
        margin-top: -10px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Loading (Cached for fast re-use)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Loads the model checkpoint once and caches it in memory."""
    device = get_default_device()
    checkpoint_path = "best_model.pth" if os.path.exists("best_model.pth") else "best_model_final_one.pth"
    model, device, metadata = load_model(checkpoint_path, device=device)
    return model, device, metadata


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🦎 Camouflaged Object Detection")
st.markdown('<p class="hero-subtitle">U-Net + ResNet34 trained on COD10K-v3</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Configuration & Model Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Info")

    # Load model
    try:
        with st.spinner("Loading PyTorch model..."):
            model, device, metadata = get_cached_model()
        st.success("Model loaded successfully!", icon="✅")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Model details
    st.markdown("### Model Details")
    st.markdown(f"""
    - **Architecture:** U-Net
    - **Backbone:** ResNet34
    - **Dataset:** COD10K-v3
    - **Input Size:** {TARGET_SIZE[0]} × {TARGET_SIZE[1]}
    - **Device:** `{device.type.upper()}`
    - **Best Checkpoint:** Epoch {metadata.get('epoch', 38)}
    - **Best Validation IoU:** `{metadata.get('best_iou', 0.6751):.4f}`
    """)

    st.markdown("---")
    st.markdown("### Detection Settings")
    threshold = st.slider(
        "Binary Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability cutoff for segmenting camouflaged pixels."
    )

    overlay_opacity = st.slider(
        "Highlight Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Transparency of the colored camouflage mask."
    )

    color_choice = st.selectbox(
        "Highlight Color",
        options=["Emerald Green", "Neon Cyan", "Coral Red", "Amber Yellow"],
        index=0
    )

    color_map = {
        "Emerald Green": (0, 230, 118),
        "Neon Cyan": (0, 229, 255),
        "Coral Red": (255, 82, 82),
        "Amber Yellow": (255, 215, 64)
    }
    chosen_rgb = color_map[color_choice]


# -----------------------------------------------------------------------------
# Main Input Section: Sample selector or custom upload
# -----------------------------------------------------------------------------
st.markdown("### 1. Select or Upload Image")

input_tab1, input_tab2 = st.tabs(["📁 Choose Sample Image", "⬆️ Upload Your Own Image"])

selected_image = None
image_name = "sample_image"

with input_tab1:
    sample_dir = "test_images"
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = [
            f for f in sorted(os.listdir(sample_dir))
            if os.path.splitext(f.lower())[1] in {".jpg", ".jpeg", ".png", ".webp"}
        ]

    if sample_files:
        sample_choice = st.radio(
            "Select sample:",
            options=sample_files,
            horizontal=True,
            format_func=lambda x: os.path.splitext(x)[0].replace("_", " ").title()
        )
        sample_path = os.path.join(sample_dir, sample_choice)
        selected_image = Image.open(sample_path)
        image_name = os.path.splitext(sample_choice)[0]
    else:
        st.info("No sample images found in 'test_images/'. Please upload an image.")

with input_tab2:
    uploaded_file = st.file_uploader(
        "Upload a photograph with a hidden/camouflaged animal or object:",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        image_name = os.path.splitext(uploaded_file.name)[0]


# -----------------------------------------------------------------------------
# Inference Execution & Results Display
# -----------------------------------------------------------------------------
if selected_image is not None:
    st.markdown("---")
    st.markdown("### 2. Prediction Results")

    # Run inference with timing
    t0 = time.perf_counter()

    # Preprocessing
    input_tensor, original_rgb, orig_shape = preprocess_image(selected_image, target_size=TARGET_SIZE)
    input_tensor = input_tensor.to(device)

    # PyTorch Model Forward Pass
    with torch.no_grad():
        logits = model(input_tensor)

    # Postprocessing
    post_results = postprocess_prediction(
        logits,
        original_shape=orig_shape,
        threshold=threshold,
        highlight_color=chosen_rgb
    )

    binary_mask = post_results["binary_mask"]
    overlay_image = create_highlight_overlay(
        original_rgb,
        binary_mask,
        color=chosen_rgb,
        alpha=overlay_opacity,
        draw_contour=True
    )
    heatmap = post_results["heatmap"]

    inference_ms = (time.perf_counter() - t0) * 1000.0

    # Summary metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{inference_ms:.1f} ms</div>
            <div class="metric-label">Inference Time</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{post_results['coverage_percent']:.2f}%</div>
            <div class="metric-label">Camouflage Coverage</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{post_results['max_confidence']:.3f}</div>
            <div class="metric-label">Max Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{orig_shape[1]} × {orig_shape[0]}</div>
            <div class="metric-label">Original Resolution</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visual display in columns
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("1. Original Image")
        st.image(original_rgb, use_container_width=True)

    with c2:
        st.subheader("2. Predicted Mask")
        st.image(binary_mask, use_container_width=True, caption=f"Binary threshold at {threshold}")

    with c3:
        st.subheader("3. Highlighted Detection")
        st.image(overlay_image, use_container_width=True, caption="Camouflaged Object Isolated")

    # Optional Confidence Heatmap expandable section
    with st.expander("🔥 View Confidence Heatmap (Probability Distribution)", expanded=False):
        st.image(heatmap, use_container_width=True, caption="Jet Colormap (Blue=Background, Red=Camouflage)")

    # -----------------------------------------------------------------------------
    # Download Buttons
    # -----------------------------------------------------------------------------
    st.markdown("### 3. Export Results")
    d1, d2 = st.columns(2)

    # Convert binary mask to PNG buffer
    mask_pil = Image.fromarray(binary_mask)
    mask_buf = io.BytesIO()
    mask_pil.save(mask_buf, format="PNG")
    mask_bytes = mask_buf.getvalue()

    # Convert overlay to PNG buffer
    overlay_pil = Image.fromarray(overlay_image)
    overlay_buf = io.BytesIO()
    overlay_pil.save(overlay_buf, format="PNG")
    overlay_bytes = overlay_buf.getvalue()

    with d1:
        st.download_button(
            label="⬇️ Download Predicted Mask (PNG)",
            data=mask_bytes,
            file_name=f"{image_name}_predicted_mask.png",
            mime="image/png",
            use_container_width=True
        )

    with d2:
        st.download_button(
            label="⬇️ Download Highlighted Result (PNG)",
            data=overlay_bytes,
            file_name=f"{image_name}_highlighted.png",
            mime="image/png",
            use_container_width=True
        )

else:
    st.info("Please select a sample image or upload one to run detection.")
