"""
restore_nafnet.py

Same idea as restore_categories.py / restore_deblurgan.py, but using NAFNet.
Walks input/<category>/ folders and runs every image through NAFNet,
writing results to output_nafnet/<same category>/.

NAFNet has separate pretrained checkpoints per task (denoising vs deblurring
vs deblurring-with-JPEG-artifacts). Point --weights at the checkpoint that
matches the category you're processing -- see step 3 below.

--- ONE-TIME SETUP (required before running this script) ---

1. Clone the official NAFNet repo next to this script:

       git clone https://github.com/megvii-research/NAFNet.git

   Your folder should look like:

       cctv-restore/
         restore_categories.py
         restore_deblurgan.py
         restore_nafnet.py
         DeblurGANv2/
         NAFNet/                 <- cloned repo
         input/
           blur/
           noise/
           compression/

2. Install its requirements (inside the NAFNet folder, same env as before):

       pip install -r NAFNet/requirements.txt
       cd NAFNet && python setup.py develop && cd ..

   (NAFNet is built on BasicSR, and "setup.py develop" registers it so it
   can be imported as a package.)

3. Download pretrained weights manually from the NAFNet README's model zoo
   (Google Drive / OneDrive links, README table under "Results and Pre-trained Models"):
   https://github.com/megvii-research/NAFNet#results-and-pre-trained-models

   Pick the checkpoint that matches what you're restoring:
     - Motion blur          -> NAFNet-GoPro-width64.pth   (best for your "blur" folder)
     - Real noise           -> NAFNet-SIDD-width64.pth    (best for your "noise" folder)
     - JPEG / compression   -> no dedicated NAFNet compression checkpoint;
                                the GoPro or SIDD weights are the closest fallback,
                                Real-ESRGAN remains the better choice for this category.

   Place them at, e.g.:

       cctv-restore/weights_nafnet/NAFNet-GoPro-width64.pth
       cctv-restore/weights_nafnet/NAFNet-SIDD-width64.pth

--- USAGE ---

    python restore_nafnet.py -i input/blur -o output_nafnet/blur --weights weights_nafnet/NAFNet-GoPro-width64.pth
    python restore_nafnet.py -i input/noise -o output_nafnet/noise --weights weights_nafnet/NAFNet-SIDD-width64.pth

Note: unlike the other two scripts, NAFNet is run one category at a time,
since each category needs a DIFFERENT checkpoint file. Run it once per
category with the matching --weights.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_REPO_DIR = "NAFNet"


def load_model(repo_dir, weights_path, width=64, enc_blk_nums=None, middle_blk_num=12, dec_blk_nums=None):
    if not os.path.isdir(repo_dir):
        print(f"[error] NAFNet repo not found at '{repo_dir}'.")
        print("        Clone it first: git clone https://github.com/megvii-research/NAFNet.git")
        sys.exit(1)

    if not os.path.isfile(weights_path):
        print(f"[error] Weights not found at '{weights_path}'.")
        print("        Download the matching .pth from the NAFNet README model zoo (Google Drive/OneDrive)")
        print("        and place it at that path.")
        sys.exit(1)

    sys.path.insert(0, os.path.abspath(repo_dir))

    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except ImportError as e:
        print("[error] Could not import NAFNet from the cloned repo.")
        print(f"        Details: {e}")
        print("        Make sure you ran: cd NAFNet && python setup.py develop")
        sys.exit(1)

    # Default GoPro/SIDD width64 config used by the official pretrained checkpoints
    if enc_blk_nums is None:
        enc_blk_nums = [2, 2, 4, 8]
    if dec_blk_nums is None:
        dec_blk_nums = [2, 2, 2, 2]

    model = NAFNet(
        img_channel=3,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
    )

    state = torch.load(weights_path, map_location="cpu")
    state_dict = state.get("params", state)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, device


def infer_single(model, device, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # NAFNet needs H and W divisible by 8 (its downsampling factor); pad if needed
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out = model(tensor)

    out = out[:, :, :h, :w]  # crop back to original size
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out = (out * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def iter_images(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, f)


def process_folder(in_dir, out_dir, model, device, skip_existing):
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
        print(f"  {rel} -> done")

    return done


def main():
    parser = argparse.ArgumentParser(description="Batch restore images with NAFNet (run once per category, matching checkpoint).")
    parser.add_argument("-i", "--input", required=True, help="Input folder for ONE category, e.g. input/blur")
    parser.add_argument("-o", "--output", required=True, help="Output folder for that category, e.g. output_nafnet/blur")
    parser.add_argument("--repo", default=DEFAULT_REPO_DIR, help="Path to the cloned NAFNet repo")
    parser.add_argument("--weights", required=True, help="Path to the matching .pth checkpoint (GoPro for blur, SIDD for noise)")
    parser.add_argument("--width", type=int, default=64, help="Model width, matches the checkpoint (64 for the standard pretrained models)")
    parser.add_argument("--enc-blk-nums", type=int, nargs="+", default=[2, 2, 4, 8], help="Encoder block counts (SIDD default: 2 2 4 8; GoPro: 1 1 1 28)")
    parser.add_argument("--middle-blk-num", type=int, default=12, help="Middle block count (SIDD default: 12; GoPro: 1)")
    parser.add_argument("--dec-blk-nums", type=int, nargs="+", default=[2, 2, 2, 2], help="Decoder block counts (SIDD default: 2 2 2 2; GoPro: 1 1 1 1)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess images even if output already exists")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input folder not found: {args.input}")
        sys.exit(1)

    print(f"Loading NAFNet from {args.weights} ...")
    model, device = load_model(
        args.repo, args.weights, width=args.width,
        enc_blk_nums=args.enc_blk_nums,
        middle_blk_num=args.middle_blk_num,
        dec_blk_nums=args.dec_blk_nums,
    )
    print(f"Model loaded on {device}.")

    print(f"\nProcessing: {args.input} -> {args.output}")
    total = process_folder(args.input, args.output, model, device, skip_existing=not args.no_skip_existing)

    print(f"\nDone. {total} image(s) restored.")
    print(f"Output written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()