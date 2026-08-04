"""
restore_bsrgan.py

Same pattern as restore_categories.py, but using BSRGAN instead of Real-ESRGAN.
Walks input/<category>/ folders (blur, noise, compression) and runs every
image through BSRGAN, writing results to output_bsrgan/<same category>/.

BSRGAN (cszn/BSRGAN) is a blind super-resolution GAN trained on a randomly
shuffled degradation pipeline (blur + downsampling + noise + JPEG compression),
so like Real-ESRGAN it's a reasonable single-pass option across all three of
your categories, not just one.

--- ONE-TIME SETUP (required before running this script) ---

1. Clone the official BSRGAN repo next to this script:

       git clone https://github.com/cszn/BSRGAN.git

   Your folder should look like:

       cctv-restore/
         restore_categories.py
         restore_deblurgan.py
         restore_nafnet.py
         restore_bsrgan.py
         BSRGAN/                 <- cloned repo
         input/
           blur/
           noise/
           compression/

2. Install requirements (same env as before is fine):

       pip install -r BSRGAN/requirements.txt

   (If there's no requirements.txt in the version you clone, the essentials
   are: torch, opencv-python, numpy — which you already have from Real-ESRGAN.)

3. Download pretrained weights. BSRGAN actually ships a download script
   for this (nicer than DeblurGAN-v2/NAFNet's manual Google Drive step):

       cd BSRGAN
       python main_download_pretrained_models.py --models "BSRGAN" --model_dir model_zoo
       cd ..

   This places BSRGAN.pth at BSRGAN/model_zoo/BSRGAN.pth automatically.
   If that script errors out (repo structure changes over time), download
   BSRGAN.pth manually from the links in the BSRGAN README instead and put
   it at the same path.

--- USAGE ---

    python restore_bsrgan.py -i input -o output_bsrgan
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_REPO_DIR = "BSRGAN"
DEFAULT_WEIGHTS = "BSRGAN/model_zoo/BSRGAN.pth"


def load_model(repo_dir, weights_path):
    if not os.path.isdir(repo_dir):
        print(f"[error] BSRGAN repo not found at '{repo_dir}'.")
        print("        Clone it first: git clone https://github.com/cszn/BSRGAN.git")
        sys.exit(1)

    if not os.path.isfile(weights_path):
        print(f"[error] Weights not found at '{weights_path}'.")
        print("        Run: cd BSRGAN && python main_download_pretrained_models.py --models \"BSRGAN\" --model_dir model_zoo")
        print("        Or download BSRGAN.pth manually from the BSRGAN README and place it at that path.")
        sys.exit(1)

    sys.path.insert(0, os.path.abspath(repo_dir))

    try:
        from models.network_rrdbnet import RRDBNet
    except ImportError as e:
        print("[error] Could not import RRDBNet from the cloned BSRGAN repo.")
        print(f"        Details: {e}")
        print("        The BSRGAN repo's internal module paths can shift between versions —")
        print("        check BSRGAN/models/ for the correct architecture filename if this fails,")
        print("        and adjust the import above to match.")
        sys.exit(1)

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, device


def infer_single(model, device, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

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
            print(f"  [error] failed on {img_path}: {e}")
            continue

        cv2.imwrite(out_path, restored)
        done += 1
        print(f"  [{category_name}] {rel} -> done")

    return done


def main():
    parser = argparse.ArgumentParser(description="Batch restore CCTV frames with BSRGAN, by category folder.")
    parser.add_argument("-i", "--input", default="input", help="Root input folder containing category subfolders")
    parser.add_argument("-o", "--output", default="output_bsrgan", help="Root output folder (mirrors input categories)")
    parser.add_argument("--repo", default=DEFAULT_REPO_DIR, help="Path to the cloned BSRGAN repo")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to BSRGAN.pth")
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
    print("Loading BSRGAN...")
    model, device = load_model(args.repo, args.weights)
    print(f"Model loaded on {device}.")

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
