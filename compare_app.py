"""
compare_app.py

Streamlit viewer to compare restoration results across all your models:
Real-ESRGAN, SwinIR, DeblurGAN-v2, NAFNet, BSRGAN, DANet -- side by side,
against the clean original and the degraded input.

Expects this folder layout (matches your D:\\model setup):

    D:\\model\\
        input\\
            clean\\
            blur\\
            noise\\
            compress\\
        output\\             <- Real-ESRGAN
        output_swinir\\
        output_deblurgan\\   <- blur only
        output_nafnet\\      <- blur + noise only
        output_bsrgan\\
        output_danet\\       <- noise only

Run with:
    pip install streamlit pillow
    streamlit run compare_app.py

(Run this from D:\\model so the relative folder paths below resolve correctly.)
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Restoration Model Comparison", layout="wide")

# --- Folder layout config ---------------------------------------------------
ROOT = Path(".")

INPUT_DIR = ROOT / "input"
CATEGORIES = ["blur", "noise", "compress"]

# name shown in UI -> (output folder, applies_to categories, notes)
MODELS = {
    "Real-ESRGAN":   {"dir": ROOT / "output",            "categories": ["blur", "noise", "compress"]},
    "SwinIR (GAN)":  {"dir": ROOT / "output_swinir",      "categories": ["blur", "noise", "compress"]},
    "BSRGAN":        {"dir": ROOT / "output_bsrgan",      "categories": ["blur", "noise", "compress"]},
    "NAFNet":        {"dir": ROOT / "output_nafnet",      "categories": ["blur", "noise"]},
    "DeblurGAN-v2":  {"dir": ROOT / "output_deblurgan",   "categories": ["blur"]},
    "DANet":         {"dir": ROOT / "output_danet",       "categories": ["noise"]},
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(
        f.name for f in folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )


def find_matching_file(folder: Path, base_id: str):
    """
    Files across categories/models don't share identical names (e.g.
    img_001_blur_gaussian_k31.png vs img_001_clean.png), but they DO share
    the same leading id like 'img_001'. Match on that prefix.
    """
    if not folder.is_dir():
        return None
    for f in folder.iterdir():
        if f.suffix.lower() in IMAGE_EXTS and f.name.startswith(base_id):
            return f
    return None


def load_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None


# --- Sidebar controls --------------------------------------------------------
st.sidebar.title("Comparison Controls")

category = st.sidebar.selectbox("Degradation category", CATEGORIES)

degraded_dir = INPUT_DIR / category
degraded_files = list_images(degraded_dir)

if not degraded_files:
    st.error(f"No images found in {degraded_dir}. Check that you're running "
             f"this from D:\\model (or wherever your input/output folders live).")
    st.stop()

# Extract base ids like 'img_001' from filenames such as img_001_blur_gaussian_k31.png
base_ids = sorted({f.split("_blur_")[0].split("_noise_")[0].split("_compress_")[0]
                    for f in degraded_files})

selected_id = st.sidebar.selectbox("Image", base_ids)

active_models = [name for name, cfg in MODELS.items() if category in cfg["categories"]]
st.sidebar.markdown(f"**Models applicable to '{category}':**")
for m in active_models:
    st.sidebar.markdown(f"- {m}")

skipped_models = [name for name in MODELS if name not in active_models]
if skipped_models:
    st.sidebar.markdown("**Not applicable to this category:**")
    for m in skipped_models:
        st.sidebar.markdown(f"- ~~{m}~~")

# --- Main display -------------------------------------------------------------
st.title(f"Restoration Comparison — {selected_id} ({category})")

clean_file = find_matching_file(INPUT_DIR / "clean", selected_id)
degraded_file = find_matching_file(degraded_dir, selected_id)

top_cols = st.columns(2)
with top_cols[0]:
    st.subheader("Clean (ground truth)")
    if clean_file:
        st.image(load_image(clean_file), use_container_width=True)
        st.caption(clean_file.name)
    else:
        st.warning("No matching clean image found.")

with top_cols[1]:
    st.subheader(f"Degraded ({category})")
    if degraded_file:
        st.image(load_image(degraded_file), use_container_width=True)
        st.caption(degraded_file.name)
    else:
        st.warning("No matching degraded image found.")

st.divider()
st.subheader("Restored by each model")

n_cols = 3
model_rows = [active_models[i:i + n_cols] for i in range(0, len(active_models), n_cols)]

for row in model_rows:
    cols = st.columns(len(row))
    for col, model_name in zip(cols, row):
        cfg = MODELS[model_name]
        model_cat_dir = cfg["dir"] / category
        restored_file = find_matching_file(model_cat_dir, selected_id)
        with col:
            st.markdown(f"**{model_name}**")
            if restored_file:
                st.image(load_image(restored_file), use_container_width=True)
                st.caption(restored_file.name)
            else:
                st.warning(f"No output found in {model_cat_dir}")

st.divider()
st.caption(
    "Tip: use the sidebar to switch categories and images. "
    "Models not applicable to a category (e.g. DeblurGAN-v2 on noise) are skipped automatically."
)