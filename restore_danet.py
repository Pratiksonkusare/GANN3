"""
restore_danet.py

Same input/output pattern as the other restore_*.py scripts, but using
DANet -- a GAN built specifically for REAL-WORLD noise removal (trained
adversarially on real sensor noise from the SIDD dataset, not synthetic
Gaussian noise). This is conceptually the closest match to actual CCTV
low-light sensor noise of any model in your set.

Since DANet is purpose-built for denoising only (it doesn't touch blur or
compression artifacts), this script only really makes sense pointed at
your "noise" folder. It will still walk whatever category folders exist
under input/, but don't expect it to do anything useful for blur/compression
frames -- keep using Real-ESRGAN/DeblurGAN-v2/NAFNet/BSRGAN for those.

--- ONE-TIME SETUP (required before running this script) ---

1. Clone the official DANet repo next to this script:

       git clone https://github.com/zsyOAOA/DANet.git

   Your folder should look like:

       cctv-restore/
         restore_categories.py
         restore_bsrgan.py
         restore_gfpgan.py
         restore_codeformer.py
         restore_danet.py
         DANet/                   <- cloned repo (weights are INSIDE it,
                                      under DANet/model_states/ -- no
                                      separate manual download needed)
         input/
           blur/
           noise/
           compression/

2. Install requirements. DANet's environment.yml lists older pinned
   versions (Python 3.7 / PyTorch 1.3) -- in practice a recent PyTorch +
   opencv + numpy (which you already have installed from the other
   scripts) is usually enough to run inference. If you hit errors tied to
   deprecated PyTorch APIs, that's why -- paste me the error and we'll
   patch the specific call.

       pip install torch opencv-python numpy scipy h5py

--- USAGE ---

    python restore_danet.py -i input -o output_danet
    python restore_danet.py -i input -o output_danet -c noise --model DANet+
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_REPO_DIR = "DANet"

# Filenames as they appear in DANet's model_states/ folder in the repo
# Filenames as they appear in DANet's model_states/ folder in the repo
MODEL_CHECKPOINTS = {
    "DANet": "DANet.pt",
    "DANet+": "DANetPlus.pt",
    "GDANet": "GDANet.pt",
    "GDANet+": "GDANetPlus_fake05.pt",       # moderate synthetic-noise variant
    "GDANet+_light": "GDANetPlus_fake025.pt",  # lighter synthetic-noise variant
}


def load_model(repo_dir, model_name):
    if not os.path.isdir(repo_dir):
        print(f"[error] DANet repo not found at '{repo_dir}'.")
        print("        Clone it first: git clone https://github.com/zsyOAOA/DANet.git")
        sys.exit(1)

    ckpt_name = MODEL_CHECKPOINTS.get(model_name)
    if ckpt_name is None:
        print(f"[error] Unknown model '{model_name}'. Choose from: {list(MODEL_CHECKPOINTS.keys())}")
        sys.exit(1)

    weights_path = os.path.join(repo_dir, "model_states", ckpt_name)
    if not os.path.isfile(weights_path):
        print(f"[error] Weights not found at '{weights_path}'.")
        print("        Check DANet/model_states/ in your cloned copy for the exact filename --")
        print("        it may differ slightly from what this script expects; update MODEL_CHECKPOINTS accordingly.")
        sys.exit(1)

    sys.path.insert(0, os.path.abspath(repo_dir))

    try:
        # DANet's denoiser network -- the exact module/class name has moved around
        # a bit across commits. Try the common location first, fall back with a
        # clear error telling you what to fix if it's changed.
        from networks.UNetD import UNetD
    except ImportError as e:
        print("[error] Could not import the DANet denoiser network from the cloned repo.")
        print(f"        Details: {e}")
        print("        Open DANet/networks/ and check the actual filename/class for the")
        print("        denoiser (commonly called UNetD). Update the import in this script")
        print("        to match -- paste me what you find and I'll fix it for you.")
        sys.exit(1)

    model = UNetD(3)
    state = torch.load(weights_path, map_location="cpu")
    state_dict = state.get("D", state.get("state_dict", state))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, device


def infer_single(model, device, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # DANet's UNetD typically needs H/W divisible by a power of 2 (its
    # downsampling depth); pad defensively to be safe.
    _, _, h, w = tensor.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]

    out = out[:, :, :h, :w]
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
    parser = argparse.ArgumentParser(description="Batch denoise CCTV frames with DANet, by category folder.")
    parser.add_argument("-i", "--input", default="input", help="Root input folder containing category subfolders")
    parser.add_argument("-o", "--output", default="output_danet", help="Root output folder (mirrors input categories)")
    parser.add_argument("-c", "--categories", nargs="*", default=None,
                         help="Specific category folders to process (default: all found, but 'noise' is where this model helps -- it does NOT handle blur or compression)")
    parser.add_argument("--repo", default=DEFAULT_REPO_DIR, help="Path to the cloned DANet repo")
    parser.add_argument("--model", default="DANet+", choices=list(MODEL_CHECKPOINTS.keys()),
                         help="Which DANet variant to use (DANet+ generally denoises best; GDANet/GDANet+ trained on more diverse noise sources)")
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
    print(f"Loading {args.model}...")
    model, device = load_model(args.repo, args.model)
    print(f"Model loaded on {device}.")

    total = 0
    for cat in categories:
        in_dir = os.path.join(args.input, cat)
        if not os.path.isdir(in_dir):
            print(f"[warn] category '{cat}' not found in {args.input}, skipping")
            continue
        out_dir = os.path.join(args.output, cat)
        print(f"\nProcessing category: {cat}")
        total += process_category(
            cat, in_dir, out_dir, model, device,
            skip_existing=not args.no_skip_existing,
        )

    print(f"\nDone. {total} image(s) denoised across {len(categories)} categories.")
    print(f"Output written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
