"""
restore_categories.py

Walks input/<category>/ folders (e.g. blur, noise, compression),
runs every image through Real-ESRGAN, and writes the result to
output/<same category>/ with the same filename.

Usage:
    python restore_categories.py -i input -o output
    python restore_categories.py -i input -o output --model realesr-general-x4v3
    python restore_categories.py -i input -o output --denoise 0.6

Folder structure expected:
    input/
      blur/
        img1.jpg
        img2.jpg
      noise/
        img3.jpg
      compression/
        img4.jpg

Produces:
    output/
      blur/
        img1.jpg
        img2.jpg
      noise/
        img3.jpg
      compression/
        img4.jpg
"""

import argparse
import os
import sys
import cv2
import torch
import numpy as np

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

MODEL_REGISTRY = {
    "realesr-general-x4v3": {
        "scale": 4,
        "arch": "compact",
        "weights": "weights/realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    },
    "RealESRGAN_x4plus": {
        "scale": 4,
        "arch": "rrdbnet",
        "weights": "weights/RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
}


def build_upsampler(model_name, denoise_strength, gpu_id=None):
    cfg = MODEL_REGISTRY[model_name]

    os.makedirs("weights", exist_ok=True)
    if not os.path.isfile(cfg["weights"]):
        print(f"[setup] downloading weights for {model_name} ...")
        load_file_from_url(
            url=cfg["url"],
            model_dir="weights",
            progress=True,
            file_name=os.path.basename(cfg["weights"]),
        )

    if cfg["arch"] == "compact":
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=32, upscale=4, act_type="prelu",
        )
        dni_weight = None
        model_path = cfg["weights"]

        # realesr-general-x4v3 supports blending with the wdn (denoise) model
        # for adjustable denoise strength, if that companion file is present.
        wdn_path = cfg["weights"].replace(".pth", "_wdn.pth")
        if denoise_strength != 1 and os.path.isfile(wdn_path):
            dni_weight = [denoise_strength, 1 - denoise_strength]
            model_path = [cfg["weights"], wdn_path]

        upsampler = RealESRGANer(
            scale=cfg["scale"],
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            gpu_id=gpu_id,
        )
    else:  # rrdbnet
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        upsampler = RealESRGANer(
            scale=cfg["scale"],
            model_path=cfg["weights"],
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            gpu_id=gpu_id,
        )

    return upsampler


def iter_images(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, f)


def process_category(category_name, in_dir, out_dir, upsampler, outscale, skip_existing):
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

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [warn] could not read {img_path}, skipping")
            continue

        try:
            restored, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as e:
            print(f"  [error] failed on {img_path}: {e}")
            continue

        cv2.imwrite(out_path, restored)
        done += 1
        print(f"  [{category_name}] {rel} -> done")

    return done


def main():
    parser = argparse.ArgumentParser(description="Batch restore CCTV frames with Real-ESRGAN, by category folder.")
    parser.add_argument("-i", "--input", default="input", help="Root input folder containing category subfolders")
    parser.add_argument("-o", "--output", default="output", help="Root output folder (mirrors input categories)")
    parser.add_argument("--model", default="realesr-general-x4v3", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--denoise", type=float, default=0.5, help="Denoise strength 0-1 (only for realesr-general-x4v3)")
    parser.add_argument("--outscale", type=float, default=4, help="Final upsampling scale")
    parser.add_argument("--gpu-id", type=int, default=None)
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
    print(f"Loading Real-ESRGAN model: {args.model}")
    upsampler = build_upsampler(args.model, args.denoise, gpu_id=args.gpu_id)

    total = 0
    for cat in categories:
        in_dir = os.path.join(args.input, cat)
        out_dir = os.path.join(args.output, cat)
        print(f"\nProcessing category: {cat}")
        total += process_category(
            cat, in_dir, out_dir, upsampler,
            outscale=args.outscale,
            skip_existing=not args.no_skip_existing,
        )

    print(f"\nDone. {total} image(s) restored across {len(categories)} categories.")
    print(f"Output written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
