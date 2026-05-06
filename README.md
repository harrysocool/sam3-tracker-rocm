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

### 1. Install ROCm 7.13 + PyTorch via AMD TheRock

The Ryzen AI Max+ 395 (gfx1151) requires **ROCm 7.13**, which is not yet in the
official stable ROCm release channel. Both ROCm and PyTorch must be installed
through AMD's [TheRock](https://github.com/ROCm/TheRock) project.

Follow the TheRock setup guide for your system, then install the matching
PyTorch nightly wheels from the
[TheRock GitHub Releases](https://github.com/ROCm/TheRock/releases) page:

```bash
# Example — replace filenames with the actual release artifacts
pip install torch-2.12.0a0+rocm7.13.0a20260411-cp312-cp312-linux_x86_64.whl \
            torchvision-0.27.0a0+rocm7.13.0a20260411-cp312-cp312-linux_x86_64.whl
```

Verify:
```bash
rocminfo | grep gfx          # should show gfx1151
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.12.0a0+rocm7.13.0a20260411  True
```

### 3. Create conda environment

```bash
conda env create -f environment.yml
conda activate sam3-tracker-rocm
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

### 6. Run the demo

```bash
python demo.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files \
    --image assets/demo.jpg \
    --box 100,200,800,600
```

---

## Performance

| Resolution | DAVIS 2017 val J | SG val J (50 seqs) | Propagation FPS |
|---|---|---|---|
| **504px** | **81.1%** | **39.6%** | **5.72** |
| 1008px | 85.8% | 44.8% | 1.35 |

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
