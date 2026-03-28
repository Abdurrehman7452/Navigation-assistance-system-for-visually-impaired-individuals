# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**obs-tackle** is a Final Year Project (FYP) implementing an obstacle detection and navigation system. It combines two pre-trained deep learning models in a dual-stream pipeline:

1. **Depth Anything V2** — monocular depth estimation (DINOv2-ViT backbone + DPT decoder)
2. **TopFormer** — semantic segmentation (Token Pyramid Transformer, CVPR 2022)

The output is fused via a 6-step pipeline ending in fuzzy logic that produces navigation commands ("Turn Left", "Go Forward", etc.) for robot/assistive navigation.

Codex will review your output

## Repository Layout

```
FYP/
├── depth_anything_v2_vits.pth          # Deployed depth model weights (95 MB, vits variant)
├── TopFormer-B_512x512_4x8_160k-39.2.pth  # Deployed segmentation weights (59 MB)
├── complete workflow.png / workflow.jpg    # System architecture diagrams
└── obs-tackle/
    └── third_party/
        ├── Depth-Anything-V2/          # Depth estimation submodule
        │   ├── run.py                  # Batch image inference entry point
        │   ├── run_video.py            # Video inference entry point
        │   ├── app.py                  # Gradio web demo
        │   ├── depth_anything_v2/      # Model implementation
        │   │   ├── dpt.py              # DPTHead decoder + DepthAnythingV2 class
        │   │   ├── dinov2.py           # DINOv2 ViT encoder
        │   │   ├── dinov2_layers/      # ViT building blocks
        │   │   └── util/               # Preprocessing transforms, fusion blocks
        │   └── metric_depth/           # Metric (absolute) depth variant + training
        └── TopFormer-main/             # Semantic segmentation submodule
            ├── mmseg/                  # MMSegmentation framework (models, datasets, apis)
            ├── local_configs/topformer/  # TopFormer-specific training configs
            ├── tools/                  # dist_train.sh, dist_test.sh, convert2onnx.py
            └── demo/                   # Jupyter notebooks for inference demos
```

## Setup & Commands

### Depth Anything V2

```bash
cd obs-tackle/third_party/Depth-Anything-V2
pip install -r requirements.txt

# Inference on images (checkpoint must be in ./checkpoints/)
python run.py --encoder vits --img-path <path_or_dir> --outdir ./vis_depth

# Inference on video
python run_video.py --encoder vits --video-path <video> --outdir ./vis_video

# Gradio web demo
python app.py
```

Key `run.py` flags: `--encoder` (vits/vitb/vitl/vitg), `--input-size` (default 518), `--pred-only` (depth only, no side-by-side), `--grayscale`.

**Checkpoint path convention**: scripts expect weights at `checkpoints/depth_anything_v2_<encoder>.pth`. The root-level `depth_anything_v2_vits.pth` needs to be placed/symlinked there.

### TopFormer

```bash
cd obs-tackle/third_party/TopFormer-main
pip install -r requirements.txt
pip install -e .   # installs mmseg package

# Distributed training (4 GPUs)
bash tools/dist_train.sh local_configs/topformer/topformer_base_512x512_160k_4x8_ade20k.py 4

# Distributed evaluation
bash tools/dist_test.sh local_configs/topformer/topformer_base_512x512_160k_4x8_ade20k.py \
    TopFormer-B_512x512_4x8_160k-39.2.pth 4

# ONNX export
python tools/convert2onnx.py <config> <checkpoint>

# Interactive demos
jupyter notebook demo/inference_demo.ipynb
```

## Architecture & Data Flow

```
RGB Frame
   │
   ├──► Depth Anything V2 ──────────────────────────► Depth Map (H×W float)
   │     DINOv2 encoder → 4 intermediate feature maps
   │     DPT decoder → fused upsampled depth
   │
   └──► TopFormer ──────────────────────────────────► Segmentation Mask (H×W class IDs)
         Stem → Token Pyramid (4 stages) → decoder

Depth Map + Segmentation Mask
   │
   ▼  Fusion Pipeline (6 steps):
   1. Extract individual segments (per-class binary masks)
   2. Compute mean depth per segment
   3. Discard segments beyond distance threshold
   4. Retain near-obstacle segments
   5. Fuzzy logic: map obstacle positions → heading angle
   6. Defuzzify → navigation command + voice output
```

### Key Classes

- `DepthAnythingV2` (`dpt.py`) — top-level model; call `model.infer_image(bgr_numpy, input_size)` → depth ndarray
- `DINOv2` (`dinov2.py`) — ViT encoder; returns intermediate features at layers 4, 9, 11, 12 (for vits)
- `DPTHead` (`dpt.py`) — decoder that fuses multi-scale features via `FeatureFusionBlock` → single-channel depth
- TopFormer backbone (`mmseg/models/backbones/topformer.py`) — registered as `'TopFormer'` in MMCV registry; configured via `.py` config files

### Model Config Pattern (Depth Anything V2)

```python
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,   96,  192,  384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192,  384,  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536, 1536, 1536]},
}
depth_model = DepthAnythingV2(**model_configs['vits'])
depth_model.load_state_dict(torch.load('depth_anything_v2_vits.pth', map_location='cpu'))
depth_model.eval()
```

### TopFormer Config System

TopFormer uses MMCV's config inheritance. `local_configs/topformer/` configs inherit from `configs/_base_/{model,dataset,schedule,runtime}.py`. The deployed config is `topformer_base_512x512_160k_4x8_ade20k.py` (39.2 mIoU on ADE20K).

## Navigation Assistance App (obs-tackle root)

Four files live in `D:\PROJECTS\FYP\` for the real-time demo:

| File | Purpose |
|------|---------|
| `setup_venv.bat` | Creates `venv/`, installs torch (CPU), onnxruntime, opencv |
| `convert_topformer_onnx.py` | Standalone (no mmcv/mmseg) TopFormer→ONNX exporter; reads `TopFormer-B_512x512_4x8_160k-39.2.pth`, writes `topformer.onnx` |
| `navigation_app.py` | Real-time camera app: 3-panel display (camera / depth / segmentation) |
| `run_navigation.bat` | One-click launcher (activates venv, runs conversion if needed, starts app) |

### Usage order
```
setup_venv.bat                  # once — creates venv, installs deps
python convert_topformer_onnx.py  # once — ~1 min, produces topformer.onnx
python navigation_app.py          # run each time
# or just: run_navigation.bat
```

### Key design decisions
- **Standalone conversion**: `convert_topformer_onnx.py` re-implements TopFormer classes in pure PyTorch (no mmcv/mmseg). Attribute names match mmcv exactly so checkpoint keys load without remapping.
- **Threading**: Two daemon threads run depth and segmentation inference at their own pace; the display loop runs at camera FPS showing the latest available results.
- **Depth display**: Spectral_r colormap (red = near, blue = far). Bottom row split into LEFT/CENTRE/RIGHT columns; columns where >8% of pixels are in the near quartile get an "OBS {SIDE}" warning overlay.
- **Segmentation display**: ADE20K palette blended 65/35 over camera; rolling top-6 class legend in frame.
- **ONNX input/output**: fixed 512×512 input → `(1, 150, 64, 64)` logits; upsampled to frame size via `F.interpolate` before argmax.

## Dependencies

- Python 3.6+, PyTorch, torchvision
- MMCV + MMSegmentation (for TopFormer)
- OpenCV, NumPy, Matplotlib
- Gradio + gradio_imageslider (for the web demo)
- Both models auto-select CUDA → MPS → CPU at runtime
