# CCTV Restoration — Multi-Model Comparison Harness

A reproducible benchmark for image restoration models on degraded CCTV footage.
Six architectures across GAN and transformer families, run over the same inputs,
compared side by side in a Streamlit viewer.

**Author:** Pratik Sonkusare
**Context:** Enhancement research for *Smart and Intelligent Video Analysis
System using Agentic AI* — PG-DAI, C-DAC ACTS Pune

---

## Why

The video-analysis pipeline needed a restoration stage to clean degraded CCTV
frames before downstream analysis. Rather than pick a model off a leaderboard,
the practical question was: *which of these actually works on our footage?*
Published benchmarks are run on curated academic datasets (GoPro, SIDD, DIV2K)
that look nothing like real surveillance frames.

So all six were set up, run over the same inputs, and compared directly.

## Models evaluated

| Model | Family | Degradations tested |
|---|---|---|
| Real-ESRGAN | GAN (RRDBNet) | blur, noise, compression |
| SwinIR | Transformer | blur, noise, compression |
| BSRGAN | GAN | blur, noise, compression |
| NAFNet | Attention-free CNN | blur, noise |
| DeblurGAN-v2 | GAN (FPN) | blur |
| DANet | Dual attention CNN | noise |

Each has a dedicated `restore_*.py` runner that loads its weights and writes
outputs to its own folder, so results stay separable and re-runnable.

## Comparison viewer

`compare_app.py` is a Streamlit app that puts the clean original, the degraded
input, and every model's output side by side for the same frame. Outputs across
models don't share filenames, so matching is done on the leading frame id
(`img_001_blur_gaussian_k31.png` ↔ `img_001_clean.png`) rather than exact name.

```bash
streamlit run compare_app.py
```

## What came out of it

Real-ESRGAN handled the broadest range of degradations, but fell short on
blur specifically — its training assumes synthetic degradation profiles that
don't match the blur in real CCTV footage. That gap is what motivated the
follow-up work: Real-ESRGAN was fine-tuned in a fully supervised way on ~210K
real degraded/clean frame pairs from our own dataset, then run back through
this same harness to verify the improvement.

Further work could add dedicated models for other defect types, so the
pipeline routes each frame to whichever restorer suits its degradation.

That finding is what motivated the follow-up work: Real-ESRGAN was fine-tuned in
a fully supervised way on ~210K real degraded/clean frame pairs from our own
dataset, then run back through this same harness to verify the improvement.

**Fine-tuning repo and weights → [GAN4](https://github.com/Pratiksonkusare/GAN4)**

## Setup

```bash
git clone https://github.com/Pratiksonkusare/GANN3.git
cd GANN3
bash setup.sh            # clones upstream model repos, installs dependencies
bash setup_weights.sh    # downloads all 10 pretrained checkpoints from Releases
```

Then place degraded inputs under `input/{blur,noise,compress}/` and clean
references under `input/clean/`, and run whichever `restore_*.py` you need.

## Notes

Upstream model repositories (BSRGAN, DANet, DeblurGANv2, NAFNet, SwinIR) are
cloned by `setup.sh` rather than vendored here. Weights are hosted as release
assets. Inputs and outputs are gitignored — this repo is the harness, not the data.
