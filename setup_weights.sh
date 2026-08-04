#!/bin/bash
# setup_weights.sh
#
# Downloads all pretrained model weights from this repo's GitHub Release
# into the correct folders. Run this once, right after cloning.
#
# Usage:
#   bash setup_weights.sh
#
# Requires: curl (present by default on almost all Linux systems, including HPC login nodes)

set -e  # stop on first error

# --- EDIT THESE TWO LINES to match your actual repo/release ---
REPO="YOUR_USERNAME/YOUR_REPO_NAME"
TAG="v1.0"
# ----------------------------------------------------------------

BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

download() {
    local filename="$1"
    local dest_dir="$2"
    local dest_path="${dest_dir}/${filename}"

    mkdir -p "$dest_dir"

    if [ -f "$dest_path" ]; then
        echo "[skip] already exists: $dest_path"
        return
    fi

    echo "[download] ${filename} -> ${dest_path}"
    curl -L -o "$dest_path" "${BASE_URL}/${filename}"
}

echo "Downloading model weights from ${BASE_URL} ..."
echo ""

download "RealESRGAN_x4plus.pth"                                  "weights"
download "realesr-general-x4v3.pth"                                "weights"
download "NAFNet-GoPro-width64.pth"                                "weights_nafnet"
download "NAFNet-SIDD-width64.pth"                                 "weights_nafnet"
download "best_fpn.h5"                                             "weights_deblurgan"
download "DANet.pt"                                                "DANet/model_states"
download "DANetPlus.pt"                                            "DANet/model_states"
download "GDANet.pt"                                               "DANet/model_states"
download "GDANetPlus_fake05.pt"                                    "DANet/model_states"
download "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"          "SwinIR/model_zoo/swinir"
download "BSRGAN.pth"                                               "BSRGAN/model_zoo"

echo ""
echo "Done. All weights downloaded (or already present)."
