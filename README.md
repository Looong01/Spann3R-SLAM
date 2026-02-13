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

## Command-Line Arguments

`main.py` currently supports:

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `datasets/examples/s00567` | Input sequence folder |
| `--config` | `config/base.yaml` | SLAM config YAML |
| `--save-as` | `default` | Output naming for evaluation save path |
| `--no-viz` | off | Disable interactive GUI window |
| `--calib` | `""` | Optional calibration YAML path |
| `--checkpoint` | `checkpoints/spann3r.pth` | Spann3R checkpoint |
| `--dust3r-checkpoint` | `checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | DUSt3R backbone checkpoint |
| `--render-gaussians` | off | Deprecated compatibility flag (rendering is enabled by default) |
| `--no-render-gaussians` | off | Disable Spann3R rendering and PNG export |
| `--render-dir` | `logs/spann3r_renders` | Directory for per-frame rendered PNGs |
| `--max-gaussians` | `4194304` | Max points used by Spann3R renderer |
| `--spatial-stride` | `4` | Per-frame point subsampling stride (`1` = no subsampling) |

Example with explicit rendering-related parameters:

```bash
python main.py \
  --dataset datasets/examples/s00567 \
  --checkpoint checkpoints/spann3r.pth \
  --dust3r-checkpoint checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --config config/base.yaml \
  --spatial-stride 2 \
  --max-gaussians 6000000 \
  --render-dir logs/spann3r_renders
```

## GUI Controls (Interactive Viz)

When GUI is enabled (default, without `--no-viz`), the left panel exposes runtime controls:

| GUI Item | Range / Default | Effect |
|---|---:|---|
| `pause` | bool | Pause frame stepping |
| `C_conf_threshold` | `0.0 .. 5.0` (from config) | Filters low-confidence points before rendering |
| `follow cam` | bool (on) | View follows current tracking camera |
| `spann3r_rendering` | bool (on) | Toggle Spann3R reprojection rendering overlay |
| `render_res_scale` | `0.2 .. 1.0` (default `0.5`) | Rendering resolution scale in viewport |
| `spatial_stride` | `1 .. 16` (default from CLI `--spatial-stride`) | Subsampling density control |
| `max_gaussians` | `20000 .. dynamic upper bound` (default from CLI `--max-gaussians`) | Cap total active points in renderer cache |
| `render_point_radius` | `0 .. 2` (default `1`) | Point splat radius in pixels |
| `cache_refresh` | `1 .. 30` (default `1`) | Current-frame cache refresh interval |
| `show_keyframe_edges` / `show_keyframe` / `show_axis` | bool | Overlay debugging visuals |
| `line_thickness` / `frustum_scale` | drag control | Frustum/edge visualization style |

### CLI vs GUI Priority

- `--spatial-stride` and `--max-gaussians` are **startup defaults** and initialize GUI sliders.
- During GUI run, slider updates are applied live to the interactive viewport.
- For PNG export in `logs/...`, current GUI values of `spatial_stride` and `max_gaussians` are used; other GUI sliders are viewport-only.
- In headless mode (`--no-viz`), only CLI values are used for the whole run.
- If `--no-render-gaussians` is set, Spann3R rendering and PNG export are disabled regardless of GUI state.

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
