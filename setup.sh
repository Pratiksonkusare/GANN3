#!/bin/bash
# setup.sh
#
# Full one-shot setup after cloning this repo. Recreates the entire working
# folder structure: clones all 5 model repos, installs Python deps,
# downloads all pretrained weights, and handles known repo-specific quirks.
#
# Usage (from inside the freshly cloned repo folder):
#   bash setup.sh
#
# After this finishes, restore_*.py and compare_app.py should just work.

set -e  # stop on first real error

echo "=================================================="
echo " Restoration pipeline -- full environment setup"
echo "=================================================="
echo ""

# --- 1. Install Python dependencies -----------------------------------------
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt
echo ""

# --- 2. Clone all model repos (skip if already present) --------------------
echo "[2/5] Cloning model repos..."

clone_repo() {
    local name="$1"
    local url="$2"
    if [ -d "$name" ]; then
        echo "  [skip] $name already exists"
    else
        echo "  [clone] $name"
        git clone "$url" "$name"
    fi
}

clone_repo "BSRGAN"      "https://github.com/cszn/BSRGAN.git"
clone_repo "DANet"       "https://github.com/zsyOAOA/DANet.git"
clone_repo "DeblurGANv2" "https://github.com/VITA-Group/DeblurGANv2.git"
clone_repo "NAFNet"      "https://github.com/megvii-research/NAFNet.git"
clone_repo "SwinIR"      "https://github.com/JingyunLiang/SwinIR.git"
echo ""

# --- 3. Download pretrained weights (from this repo's GitHub Release) ------
echo "[3/5] Downloading pretrained weights..."
bash setup_weights.sh
echo ""

# --- 4. Fix known repo-specific quirks --------------------------------------
echo "[4/5] Applying known fixes..."

# NAFNet needs to be registered as an importable package
echo "  [fix] registering NAFNet package (setup.py develop)"
(cd NAFNet && python setup.py develop)

# DeblurGANv2's predict.py hardcodes 'config/config.yaml' relative to CWD,
# not relative to the repo -- so we mirror the config folder up to root.
echo "  [fix] copying DeblurGANv2 config to repo root"
mkdir -p config
cp -r DeblurGANv2/config/* config/ 2>/dev/null || echo "    (config already present or DeblurGANv2/config missing -- check manually if DeblurGAN fails)"

echo ""

# --- 5. Done -----------------------------------------------------------------
echo "[5/5] Setup complete."
echo ""
echo "You can now run any of:"
echo "  python restore_categories.py -i input -o output"
echo "  python restore_swinir.py -i input -o output_swinir"
echo "  python restore_bsrgan.py -i input -o output_bsrgan"
echo "  python restore_nafnet.py -i input/blur -o output_nafnet/blur --weights weights_nafnet/NAFNet-GoPro-width64.pth --enc-blk-nums 1 1 1 28 --middle-blk-num 1 --dec-blk-nums 1 1 1 1"
echo "  python restore_nafnet.py -i input/noise -o output_nafnet/noise --weights weights_nafnet/NAFNet-SIDD-width64.pth"
echo "  python restore_deblurgan.py -i input -o output_deblurgan -c blur"
echo "  python restore_danet.py -i input -o output_danet -c noise"
echo "  streamlit run compare_app.py"
echo ""
echo "Note: you'll still need to place your own input/ images (clean/blur/noise/compress)"
echo "since those are your test dataset, not part of this repo."
