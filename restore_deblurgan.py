"""
restore_deblurgan.py

Same idea as restore_categories.py, but using DeblurGAN-v2 instead of Real-ESRGAN.
Walks input/<category>/ folders and runs every image through DeblurGAN-v2,
writing results to output_deblurgan/<same category>/.

DeblurGAN-v2 is specifically a MOTION-DEBLUR model, so it's most useful on
your "blur" category. You can still point it at any/all categories with -c,
but noise/compression images won't benefit much from it.

--- ONE-TIME SETUP (required before running this script) ---

1. Clone the official DeblurGAN-v2 repo next to this script:

       git clone https://github.com/VITA-Group/DeblurGANv2.git

   Your folder should look like:

       cctv-restore/
         restore_categories.py
         restore_deblurgan.py
         DeblurGANv2/            <- cloned repo
         input/
           blur/
           noise/
           compression/

2. Install its requirements (inside the DeblurGANv2 folder, ideally in the
   same venv/conda env you're already using):

       pip install -r DeblurGANv2/requirements.txt

3. Download the pretrained weights manually (Google Drive link, from the
   DeblurGANv2 README): the "best_fpn.h5" checkpoint (FPN-Inception, the
   recommended default). Google Drive can't be auto-downloaded from a script
   reliably, so grab it yourself from the link in the repo's README:
   https://github.com/VITA-Group/DeblurGANv2#pretrained-models

   Place it at:

       cctv-restore/weights_deblurgan/best_fpn.h5

--- USAGE ---

    python restore_deblurgan.py -i input -o output_deblurgan
    python restore_deblurgan.py -i input -o output_deblurgan -c blur
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_REPO_DIR = "DeblurGANv2"
DEFAULT_WEIGHTS = "weights_deblurgan/best_fpn.h5"
DEFAULT_CONFIG = "config/config.yaml"  # relative to repo dir, FPN-Inception config


def load_predictor(repo_dir, weights_path, config_name):
    if not os.path.isdir(repo_dir):
        print(f"[error] DeblurGANv2 repo not found at '{repo_dir}'.")
        print("        Clone it first: git clone https://github.com/VITA-Group/DeblurGANv2.git")
        sys.exit(1)

    if not os.path.isfile(weights_path):
        print(f"[error] Weights not found at '{weights_path}'.")
        print("        Download 'best_fpn.h5' manually from the DeblurGANv2 README (Google Drive)")
        print("        and place it at that path.")
        sys.exit(1)

    # DeblurGANv2's own code lives in the cloned repo, so add it to sys.path
    sys.path.insert(0, os.path.abspath(repo_dir))

    try:
        from predict import Predictor  # provided by the DeblurGANv2 repo
    except ImportError as e:
        print("[error] Could not import Predictor from the cloned DeblurGANv2 repo.")
        print(f"        Details: {e}")
        print("        Make sure DeblurGANv2/predict.py exists and its requirements are installed.")
        sys.exit(1)

    predictor = Predictor(weights_path=weights_path)
    return predictor


def iter_images(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, f)


def process_category(category_name, in_dir, out_dir, predictor, skip_existing):
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

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            restored_rgb = predictor(img_rgb, None)
        except RuntimeError as e:
            print(f"  [error] failed on {img_path}: {e}")
            continue

        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, restored_bgr)
        done += 1
        print(f"  [{category_name}] {rel} -> done")

    return done


def main():
    parser = argparse.ArgumentParser(description="Batch deblur CCTV frames with DeblurGAN-v2, by category folder.")
    parser.add_argument("-i", "--input", default="input", help="Root input folder containing category subfolders")
    parser.add_argument("-o", "--output", default="output_deblurgan", help="Root output folder (mirrors input categories)")
    parser.add_argument("-c", "--categories", nargs="*", default=None,
                         help="Specific category folders to process (default: all found, but 'blur' is where this model helps most)")
    parser.add_argument("--repo", default=DEFAULT_REPO_DIR, help="Path to the cloned DeblurGANv2 repo")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to best_fpn.h5")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config file name inside the repo")
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess images even if output already exists")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input folder not found: {args.input}")
        sys.exit(1)

    all_categories = [
        d for d in sorted(os.listdir(args.input))
        if os.path.isdir(os.path.join(args.input, d))
    ]
    categories = args.categories if args.categories else all_categories

    if not categories:
        print(f"No category subfolders found inside {args.input}. Expected e.g. blur/ noise/ compression/")
        sys.exit(1)

    print(f"Processing categories: {categories}")
    print("Loading DeblurGAN-v2 (this can take a moment)...")
    predictor = load_predictor(args.repo, args.weights, args.config)

    total = 0
    for cat in categories:
        in_dir = os.path.join(args.input, cat)
        if not os.path.isdir(in_dir):
            print(f"[warn] category '{cat}' not found in {args.input}, skipping")
            continue
        out_dir = os.path.join(args.output, cat)
        print(f"\nProcessing category: {cat}")
        total += process_category(
            cat, in_dir, out_dir, predictor,
            skip_existing=not args.no_skip_existing,
        )

    print(f"\nDone. {total} image(s) deblurred across {len(categories)} categories.")
    print(f"Output written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
