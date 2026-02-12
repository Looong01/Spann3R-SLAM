<p align="center">
  <h1 align="center">Spann3R-SLAM</h1>
  <p align="center">
    MASt3R-SLAM style real-time SLAM pipeline integrated with
    <a href="https://github.com/HengyiWang/spann3r">HengyiWang/spann3r</a>
  </p>
</p>

## Overview

This repository is now wired to run **Spann3R + DUSt3R checkpoints** directly.

This branch:
- copies upstream Spann3R source code into this repo (`spann3r_core/`), no submodule;
- loads `checkpoints/spann3r.pth` + `checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` by default;
- defaults to running on `datasets/examples/s00567`.

## Integrated Upstream Code

Upstream source from `HengyiWang/spann3r` is copied into:
- `spann3r_core/spann3r`
- `spann3r_core/dust3r`
- `spann3r_core/croco`
- plus upstream demo/eval/docs/assets files under `spann3r_core/`

No `thirdparty` submodule is used for this integration.

## Installation

### 1. Environment

```bash
conda create -n spann3r-slam python=3.11 -y
conda activate spann3r-slam
```

### 2. PyTorch

Install the CUDA-compatible PyTorch version for your machine, for example:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Dependencies

```bash
pip install -r requirements.txt
pip install -e thirdparty/in3d
pip install --no-build-isolation thirdparty/lietorch
pip install --no-build-isolation -e .
```

## Checkpoints

Create checkpoint folder:

```bash
mkdir -p checkpoints
```

Place these files in `checkpoints/`:
- `spann3r.pth` (from: https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy?usp=sharing)
- `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` (from: https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth)

## Demo Data

Put the example sequence in:

```text
datasets/examples/s00567
```

(downloaded from the same Google Drive link above).

`RGBFiles` loader now supports `png/jpg/jpeg` images.
Default resize resolution is set to `224` to match Spann3R demo settings.

## Run

Default command (already points to the requested checkpoints and sample data):

```bash
python main.py
```

Equivalent explicit command:

```bash
python main.py \
  --dataset datasets/examples/s00567 \
  --checkpoint checkpoints/spann3r.pth \
  --dust3r-checkpoint checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --config config/base.yaml
```

Headless mode:

```bash
python main.py --no-viz
```

## Notes

- SLAM runtime function names are fully switched to `spann3r_*`.
- Retrieval DB now uses a lightweight feature-similarity fallback, so MASt3R retrieval checkpoints are not required.
- Legacy Gaussian rendering path has been removed.

## Key Paths

- Main entry: `main.py`
- Spann3R bridge layer: `spann3r_slam/spann3r_utils.py`
- Path bootstrap: `spann3r_slam/_setup_paths.py`
- Upstream copied source: `spann3r_core/`

## Credits

- [Spann3R](https://github.com/HengyiWang/spann3r)
- [DUSt3R](https://github.com/naver/dust3r)
- [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/)
