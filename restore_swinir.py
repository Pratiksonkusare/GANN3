"""
restore_swinir.py

Same pattern as restore_categories.py / restore_bsrgan.py, using SwinIR's
real-world super-resolution model -- specifically the GAN-trained variant
(003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth), the only one of SwinIR's
several pretrained models actually trained with adversarial loss.

IMPORTANT HONESTY NOTE: SwinIR itself is a Swin Transformer architecture,
not a GAN. Most of its pretrained checkpoints (denoising, JPEG artifact
removal) are trained with plain L1/perceptual loss, not adversarially.
This script specifically uses the ONE checkpoint that IS GAN-trained (the
real-world SR model, trained the same way as BSRGAN/Real-ESRGAN with an
adversarial + perceptual loss for realistic real-world degradation). Like
BSRGAN, this covers blur + downsampling + noise + compression in a single
blind pass, so it's a reasonable addition alongside your other models --
just know it's "GAN-trained SwinIR", not "SwinIR is a GAN architecture".

--- ONE-TIME SETUP ---

1. Clone the official repo next to this script:

       git clone https://github.com/JingyunLiang/SwinIR.git

   Your folder should look like:

       cctv-restore/
         restore_categories.py
         restore_bsrgan.py
         restore_swinir.py
         SwinIR/                  <- cloned repo
         input/
           blur/
           noise/
           compression/

2. Install requirements:

       pip install -r SwinIR/requirements.txt
       (or, if that file doesn't exist in your clone: pip install torch opencv-python numpy timm)

3. Weights auto-download on first run of this script (direct GitHub
   release, no Google Drive). If your network can't reach GitHub directly,
   download manually from:
   https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
   and place it at:
       SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth

--- USAGE ---

    python restore_swinir.py -i input -o output_swinir
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_REPO_DIR = "SwinIR"
WEIGHTS_REL_PATH = "model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
WEIGHTS_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"

WINDOW_SIZE = 8


def load_model(repo_dir):
    if not os.path.isdir(repo_dir):
        print(f"[error] SwinIR repo not found at '{repo_dir}'.")
        print("        Clone it first: git clone https://github.com/JingyunLiang/SwinIR.git")
        sys.exit(1)

    weights_path = os.path.join(repo_dir, WEIGHTS_REL_PATH)
    if not os.path.isfile(weights_path):
        print("[setup] downloading GAN-trained SwinIR weights (first run only)...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(WEIGHTS_URL, weights_path)
        except Exception as e:
            print(f"[error] auto-download failed: {e}")
            print(f"        Download manually from: {WEIGHTS_URL}")
            print(f"        and place it at: {weights_path}")
            sys.exit(1)

    sys.path.insert(0, os.path.abspath(repo_dir))

    try:
        from models.network_swinir import SwinIR
    except ImportError as e:
        print("[error] Could not import SwinIR from the cloned repo.")
        print(f"        Details: {e}")
        print("        Make sure the repo cloned correctly (models/network_swinir.py should exist).")
        sys.exit(1)

    # Config matches the official 003_realSR_BSRGAN..._GAN checkpoint (SwinIR-M, x4, real-world SR)
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=WINDOW_SIZE,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )

    state = torch.load(weights_path, map_location="cpu")
    state_dict = state.get("params_ema", state.get("params", state))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, device


def infer_single(model, device, img_bgr, tile=None, tile_overlap=32):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    _, _, h, w = tensor.shape
    pad_h = (WINDOW_SIZE - h % WINDOW_SIZE) % WINDOW_SIZE
    pad_w = (WINDOW_SIZE - w % WINDOW_SIZE) % WINDOW_SIZE
    tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out = model(tensor)

    out = out[:, :, : h * 4, : w * 4]  # crop back (x4 upscale), matching original aspect
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out = (out * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def iter_images(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, f)


def process_category(category_name, in_dir, out_dir, model, device, skip_existing):
    os.makedirs(out_dir, exist_ok=True)
    images = sorted(iter_images(in_dir))
    if not images:
        print(f"  [skip] no images found in {in_dir}")
        return 0

    done = 0
    for img_path in images:
        rel = os.path.relpath(img_path, in_dir)
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path) or out_dir, exist_ok=True)

        if skip_existing and os.path.isfile(out_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [warn] could not read {img_path}, skipping")
            continue

        try:
            restored = infer_single(model, device, img)
        except RuntimeError as e:
            print(f"  [error] failed on {img_path} (possibly out of GPU memory on a large frame): {e}")
            continue

        cv2.imwrite(out_path, restored)
        done += 1
        print(f"  [{category_name}] {rel} -> done")

    return done


def main():
    parser = argparse.ArgumentParser(description="Batch restore CCTV frames with GAN-trained SwinIR, by category folder.")
    parser.add_argument("-i", "--input", default="input", help="Root input folder containing category subfolders")
    parser.add_argument("-o", "--output", default="output_swinir", help="Root output folder (mirrors input categories)")
    parser.add_argument("--repo", default=DEFAULT_REPO_DIR, help="Path to the cloned SwinIR repo")
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess images even if output already exists")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input folder not found: {args.input}")
        sys.exit(1)

    categories = [
        d for d in sorted(os.listdir(args.input))
        if os.path.isdir(os.path.join(args.input, d))
    ]

    if not categories:
        print(f"No category subfolders found inside {args.input}. Expected e.g. blur/ noise/ compression/")
        sys.exit(1)

    print(f"Found categories: {categories}")
    print("Loading GAN-trained SwinIR (real-world SR model)...")
    model, device = load_model(args.repo)
    print(f"Model loaded on {device}.")
    print("Note: on large CCTV frames + CPU this can be slow -- SwinIR's transformer")
    print("attention is heavier than the CNN-based models in your other scripts.")

    total = 0
    for cat in categories:
        in_dir = os.path.join(args.input, cat)
        out_dir = os.path.join(args.output, cat)
        print(f"\nProcessing category: {cat}")
        total += process_category(
            cat, in_dir, out_dir, model, device,
            skip_existing=not args.no_skip_existing,
        )

    print(f"\nDone. {total} image(s) restored across {len(categories)} categories.")
    print(f"Output written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
