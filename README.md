# SAM3 Video Tracker — ROCm / AMD

Mask-level video tracking pipeline built on [SAM3](https://github.com/facebookresearch/sam3),
optimized for AMD ROCm hardware. Achieves **5.72 FPS** (propagation frame) on an
AMD Ryzen AI Max+ 395 with a DAVIS 2017 val Mean J of **81.1%** (504px).

> **Hardware requirement**: AMD gfx1151 (Radeon 8060S / Ryzen AI Max+ 395) with ROCm 7.x.
> Other AMD GPUs supporting ROCm may work but are untested.

---

## How it works

- **Frame 0**: user provides a bounding box → `mask_decoder_init.onnx` produces the initial mask
- **Frames 1+**: memory bank drives `mask_decoder_propagate.onnx` — no prompt needed
- **Backbone** runs as PyTorch on ROCm GPU (FP16); tracking modules run via ONNX Runtime

```
Input frame
  → PyTorch backbone (ROCm GPU FP16)          ~142ms
  → memory_attention_fixed_N7.onnx (MIGraphX)  ~16ms
  → mask_decoder_propagate.onnx (CPU ONNX)      ~7ms
  → memory_encoder.onnx (CPU ONNX)              ~9ms
  ─────────────────────────────────────────────────
  Total propagation frame: ~175ms → 5.72 FPS
```

---

## Setup

> **Important**: PyTorch for gfx1151 (ROCm 7.13) and `onnxruntime-migraphx`
> are **not on standard PyPI**. The correct setup flow is to start from
> AMD's [TheRock](https://github.com/ROCm/TheRock) environment and install
> additional packages on top — not to create a fresh conda env from scratch.

### 1. Install ROCm SDK + PyTorch for gfx1151

AMD provides official nightly wheels for gfx1151 at:
**`https://rocm.nightlies.amd.com/v2/gfx1151/`**

```bash
# Create a fresh conda environment
conda create -n sam3-tracker python=3.12 -y
conda activate sam3-tracker

# Step 1a: Install ROCm runtime Python packages (pin to 20260411 for onnxruntime-migraphx compatibility)
pip install rocm "rocm-sdk-core==7.13.0a20260411" rocm-sdk-libraries-gfx1151 rocm-sdk-devel \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# Step 1b: Install PyTorch matching the same ROCm build date
pip install "torch==2.12.0a0+rocm7.13.0a20260411" \
            "torchvision==0.27.0a0+rocm7.13.0a20260411" \
            triton \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

> **Why pin to `20260411`?** `onnxruntime-migraphx 1.24.2` was compiled against the
> ROCm SDK from that date. Mismatched ROCm versions (even a few days apart) can cause
> MIGraphX kernel compilation to crash at runtime.

Set the following environment variables (add to `~/.bashrc` or your run script):

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.12.0a0+rocm7.13.0a20260411  True
```

### 2. Install onnxruntime-migraphx

The MIGraphX-enabled ONNX Runtime is provided by
[Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build).

```bash
pip install https://github.com/Looong01/onnxruntime-rocm-build/releases/download/v1.24.2/onnxruntime_migraphx-1.24.2-cp312-cp312-manylinux_2_34_x86_64.whl
```

### 3. Install remaining dependencies

```bash
# Install additional packages needed by this project
pip install -r requirements.txt
```

### 4. Download SAM3 model weights

```bash
huggingface-cli download facebook/sam3 --local-dir model/sam3
```

### 5. Export ONNX tracking modules (~5 minutes)

```bash
# 504px — recommended (5.72 FPS, DAVIS J=81.1%)
python export/export_tracker_modules.py --imgsz 504 --output-dir onnx_files

# 1008px — higher quality (1.35 FPS, DAVIS J=85.8%)
python export/export_tracker_modules.py --imgsz 1008 --output-dir onnx_files_1008
```

> `--fixed-slots 7` (default) also exports `memory_attention_fixed_N7.onnx` with static shapes.
> The tracker automatically picks this file and attempts to run it on MIGraphX (falling back to CPU
> if MIGraphX kernel compilation fails, which is known to happen on some builds).

### 6. Run the demo

```bash
python demo.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files \
    --image assets/demo.jpg \
    --box 85,281,1710,850
```

---

## Results

### Single-image segmentation (box prompt)

| truck (demo) | drift-straight (J = 95.2%) | parkour (J = 92.2%) |
|:---:|:---:|:---:|
| ![truck](assets/demo_tracked.jpg) | ![drift-straight](assets/demo_drift-straight.jpg) | ![parkour](assets/demo_parkour.jpg) |

### Video tracking (DAVIS 2017 val, 504px)

| blackswan  (J = 93.0%) | dog  (J = 94.7%) | camel  (J = 96.0%) |
|:---:|:---:|:---:|
| ![blackswan](assets/demo_blackswan.gif) | ![dog](assets/demo_dog.gif) | ![camel](assets/demo_camel.gif) |

---

## Performance

### Video tracking (propagation FPS)

| Resolution | DAVIS 2017 val J | SG val J (50 seqs) | Propagation FPS |
|---|---|---|---|
| **504px** | **81.1%** | **39.6%** | **5.72** |
| 1008px | 85.8% | 44.8% | 1.35 |

*Propagation FPS measured with `memory_attention_fixed_N7.onnx` on MIGraphX, after TunableOp warmup.*

### Single-frame pipeline (Pipeline A: box/point → mask, no tracking)

| Stage | 504px |
|---|---:|
| backbone `[PyTorch ROCm FP16]` | 138.9 ms |
| mask_decoder_init `[ONNX CPU]` | 7.1 ms |
| **Total → FPS** | **159 ms → 6.29 FPS** |

Run `python eval/bench_pipeline.py --checkpoint model/sam3 --onnx-dir onnx_files` to reproduce.

*Measured on AMD Ryzen AI Max+ 395 (gfx1151), after TunableOp warmup.*

---

## Evaluation

```bash
# DAVIS 2017 val (download from https://davischallenge.org)
python eval/eval_davis.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files \
    --davis dataset/DAVIS

# Smartglass SG val (download separately)
python eval/eval_saco_sg.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files \
    --imgsz 504
```

---

## Project structure

```
sam3-tracker-rocm/
├── tracker/
│   ├── tracker.py          # SAM3OnnxTracker class
│   └── __init__.py
├── export/
│   └── export_tracker_modules.py   # Generate ONNX files from model weights
├── eval/
│   ├── eval_davis.py               # DAVIS 2017 evaluation
│   └── eval_saco_sg.py             # Smartglass SG evaluation
├── demo.py                         # Single image / video demo
├── assets/demo.jpg                 # Sample image
├── docs/project_summary.md         # Technical report
└── environment.yml
```

---

## Known limitations

- **MIGraphX cold-start**: first run compiles GPU kernels (~10 min for 1008px backbone,
  ~30s for 504px). Subsequent runs are fast. TunableOp autotuning adds ~8 warmup passes
  at startup.
- **MIGraphX library conflict**: PyTorch and MIGraphX ONNX sessions cannot be loaded in
  the same process with arbitrary ordering — the tracker initializes PyTorch first, then
  loads ONNX sessions.
- **Only `memory_attention_fixed_N7.onnx` runs on MIGraphX**; other tracking modules
  (mask decoder, memory encoder) fall back to CPU ONNX due to MIGraphX compiler bugs on gfx1151.

---

## Acknowledgements

- **SAM3**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3) — model weights
  and architecture. Model weights must be downloaded separately from
  [facebook/sam3](https://huggingface.co/facebook/sam3) on HuggingFace.
- **DART**: the `sam3_tracker_video` model class used in this project originates from the
  [DART](https://arxiv.org/abs/2603.11441) project's custom transformers fork, and has since
  been merged into the official HuggingFace transformers library (≥ 5.7.0).
- **HuggingFace Transformers** ≥ 5.8.0 is required for `Sam3TrackerVideoModel`.
