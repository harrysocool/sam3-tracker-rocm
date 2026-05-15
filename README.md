# SAM3 Video Tracker — ROCm / AMD

Open-vocabulary video tracking built on [SAM3](https://github.com/facebookresearch/sam3),
optimized for AMD ROCm hardware. A **text prompt** finds the target on frame 0; SAM3
propagates the mask through subsequent frames. **Box-prompt** mode skips detection for
maximum throughput.

Achieves **5.5 FPS** text-prompt and **12.21 FPS** box-prompt (propagation, 504px) on an
AMD Ryzen AI Max+ 395. DAVIS 2017 val Mean J: **81.6%** (504px).

> **Hardware requirement**: AMD gfx1151 (Radeon 8060S / Ryzen AI Max+ 395) with ROCm 7.x.
> Other AMD GPUs supporting ROCm may work but are untested.

## Contents

- [How it works](#how-it-works)
- [Setup](#setup)
- [Run the demo](#run-the-demo)
- [Results](#results)
- [Performance](#performance)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)

---

## How it works

Two pipelines share the same ViT backbone and SAM3 mask decoder. The difference is
how the first-frame mask is obtained:

```
Text-prompt:  "swan" ──► CLIP encoder ──► DETR detector ──► mask init ──► memory propagation
Box-prompt:   [box]                                       ──► mask init ──► memory propagation
```

### Text-prompt pipeline (`demo_text.py`)

Uses `Sam3VideoModel` — the full SAM3 detection + tracking stack:

1. **CLIP text encoder** converts the prompt (e.g. `"swan"`) into query embeddings
2. **ViT backbone** extracts multi-scale visual features from frame 0
3. **DETR encoder + decoder** localises **all matching objects**; presence-aware
   scoring + greedy mask-IoU NMS ranks them by score
4. **SAM3 mask decoder** produces the initial segmentation mask
5. **Memory propagation** (frames 1+): backbone features + memory bank drive the mask
   decoder each frame; object pointers accumulate in the memory bank

All detected objects above `--min-score` are tracked simultaneously — backbone runs
once per frame regardless of object count, so 4 objects costs only ~20% more than 1.

All three heavy modules — backbone, DETR encoder, memory attention — are MIG-accelerated.

### Box-prompt pipeline (`demo.py`)

The user draws a bounding box on frame 0 — detection is skipped entirely.
Uses `SAM3OnnxTracker`: a lightweight ONNX pipeline with MIGraphX-compiled modules for
maximum propagation FPS.

### Shared propagation loop (frames 1+)

```
pixel_values ──► backbone.mxr ──► memory_attention (ORT MIG EP) ──► mask_decoder ──► mask
                                          ▲
                              memory bank (7 spatial frames + object pointers)
```

---

## Setup

### Prerequisites

Have these in place **before** running `./setup.sh`:

| Requirement | Handled by | Notes |
|---|---|---|
| Hardware: AMD Ryzen AI Max+ 395 (gfx1151) | You | Other ROCm-capable AMD GPUs may work but are untested |
| OS: Ubuntu 24.04.4 LTS | You | Other Linux distros with ROCm 7.x support may work |
| Kernel: 6.8+ (tested: 6.18.6) | You | Required for gfx1151 AMDGPU driver support |
| **conda / miniforge** (any recent) | ⚠️ You — install before running setup.sh | `setup.sh` errors out if conda is not found. [Install miniforge](https://github.com/conda-forge/miniforge) |
| **BIOS UMA Frame Buffer Size = 64 GB** | ⚠️ You — set in BIOS | **128 GB systems only** — without this, backbone OOMs at 1008px. See [Finding #7](docs/project_summary.md). |
| **System ROCm 7.2 APT** (`migraphx 2.15.0`) | ✅ setup.sh step 0a | Installs automatically; skip with `--skip-apt` if already done |

> **Why two ROCm stacks?** AMD currently ships gfx1151 PyTorch support only in nightly
> pip wheels (ROCm 7.13), while MIGraphX is only in the stable APT release (ROCm 7.2).
> Both are required; `setup.sh` installs them in the right order.

### Stage 1 — Environment (`./setup.sh`, ~10 min)

```bash
git clone https://github.com/harrysocool/sam3-tracker-rocm.git
cd sam3-tracker-rocm
./setup.sh
```

Useful flags: `--skip-apt`, `--skip-migraphx`, `--env NAME`.
See [setup.sh](setup.sh) for details.

What it does:
1. APT: ROCm 7.2 stack + stock MIGraphX 2.15.0 (`--skip-apt` to bypass)
2. Patched MIGraphX tarball (~2 min, two unreleased fixes for the headline FPS — `--skip-migraphx` to bypass)
3. Conda env (`sam3-tracker` by default; override with `--env`) with Python 3.12
4. ROCm 7.13 nightly SDK + PyTorch (gfx1151 wheels, ~2–5 min)
5. ONNX Runtime MIGraphX EP wheel (1.24.2)
6. Python dependencies from `requirements.txt`
7. Model weights from community mirror `1038lab/sam3` (no HF account needed)

### Stage 2 — Build model artefacts (`export/build.py`)

After `setup.sh`, activate the environment and build artefacts for the pipeline(s) you want:

```bash
conda activate sam3-tracker

# Box-prompt only — demo.py  (~10 min @504px)
python export/build.py --pipeline box --imgsz 504

# Text-prompt MIG — demo_text.py --mig  (~18 min @504px)
python export/build.py --pipeline text --imgsz 504

# Both pipelines at 504px
python export/build.py --pipeline all --imgsz 504

# Both pipelines, both resolutions (~90 min total)
python export/build.py --pipeline all --imgsz 504 1008
```

Each step skips if output already exists — safe to re-run after interruption.
Use `--force` to rebuild from scratch.

> **Which resolution?** 504px runs 3-10× faster with slightly lower mask quality.
> Start with 504 — switch to 1008 if you need higher accuracy.

### Manual / alternative paths

<details>
<summary><b>Patched MIGraphX — build from source</b></summary>

The headline FPS requires two unreleased MIGraphX fixes (a `find_splits` patch +
NHWC output fix). `setup.sh` installs a prebuilt tarball; if you'd rather build:

| Path | FPS (504 / 1008 px) | What you need |
|---|---|---|
| Stock APT 2.15.0 | 5.72 / 1.35 | Checkout tag `v0.1-migraphx-2.15` |
| **Prebuilt tarball** (default) | **8.21 / 2.31** | `setup.sh` downloads + installs |
| Build from source | 8.21 / 2.31 | See [`docs/build_migraphx_patched.md`](docs/build_migraphx_patched.md) |

</details>

<details>
<summary><b>Model weights — download from official HF source</b></summary>

`setup.sh` pulls `1038lab/sam3` (community mirror, no account). To use the
official `facebook/sam3` repo instead (HuggingFace account + accepted terms
required), download manually before running `setup.sh`:

```bash
hf download facebook/sam3 model.safetensors --local-dir model/sam3
```

`setup.sh` skips the download step if `model/sam3/model.safetensors` already exists.

> `hf` is the new CLI in `huggingface_hub ≥ 1.0`. Older versions ship `huggingface-cli` (same arguments).

</details>

<details>
<summary><b>Step-by-step manual install (no setup.sh)</b></summary>

For full control over each step (APT, conda, pip, ONNX export, backbone compile)
see [`docs/manual_setup.md`](docs/manual_setup.md).

</details>

---

## Run the demo

Two demo entry points — pick the one matching your use case:

| Demo | Prompt | Pipeline | Steady-state FPS | Use for |
|---|---|---|---|---|
| `demo.py` | bounding box | Tracking only (no detection) | **12.2 FPS @ 504px** | known target, real-time |
| `demo_text.py --mig --imgsz 504` | text | Detection + tracking @ 504px, N objects | **5.5 FPS** (1 obj) / **4.4 FPS** (4 obj) | open-vocabulary, multi-object |
| `demo_text.py --mig` | text | Detection + tracking @ 1008px | **1.5 FPS @ 1008px** | open-vocabulary, higher quality |
| `demo_text.py` | text | Detection + tracking (pure PyTorch) | 0.5 FPS @ 1008px | open-vocabulary, no MIG setup |

All commands below assume you have activated the conda env (`conda activate sam3-tracker`)
and are in the project root. The MIGraphX text-prompt path requires the `LD_PRELOAD` shown
in the commands below to resolve a dual-ROCm-version conflict; the box-prompt and PyTorch
text-prompt paths do not need it.

### Box-prompt (`demo.py`) — tracking only, fastest

> `assets/truck.jpg` and `assets/blackswan.mp4` are bundled demo files (a truck image and
> a short swan clip). Replace with your own image or video. `--box x1,y1,x2,y2` is
> the bounding box around the target on frame 0, in pixel coordinates.

```bash
# Image — MIGraphX backbone (default, ~115 ms / frame)
python demo.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --image assets/truck.jpg --box 85,281,1710,850   # x1,y1,x2,y2 in pixels

# Image — PyTorch backbone fallback (no MIGraphX needed)
python demo.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --backbone pytorch \
    --image assets/truck.jpg --box 85,281,1710,850

# Video (any mp4) — output written to outputs/box/<stem>_tracked.mp4
python demo.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --video assets/blackswan.mp4 --box 320,170,650,400
```

### Text-prompt (`demo_text.py`) — detection + tracking, open-vocabulary

> `assets/blackswan.mp4` is a bundled swan clip. Replace with your own video.
> MIG commands (`--mig`) require Stage 2 artefacts to be built first.

```bash
# Image — pure PyTorch path (no MIG artifacts needed)
python demo_text.py --checkpoint model/sam3 \
    --image assets/truck.jpg --text "truck"

# Video — pure PyTorch (~0.5 FPS @ 1008px)
python demo_text.py --checkpoint model/sam3 \
    --video assets/blackswan.mp4 --text "swan" --max-frames 60  # omit for full video

# Video — MIG @504 (~5.1 FPS, 10× over PT baseline, best for demos)
LD_PRELOAD=/opt/rocm-7.2.x/lib/libmigraphx_c.so.3:/opt/rocm-7.2.x/lib/migraphx/lib/libmigraphx.so.2016000.0 \
    python demo_text.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --video assets/blackswan.mp4 --text "swan" --imgsz 504 --mig --max-frames 60

# Video — MIG @1008 (~1.5 FPS, highest mask quality)
LD_PRELOAD=/opt/rocm-7.2.x/lib/libmigraphx_c.so.3:/opt/rocm-7.2.x/lib/migraphx/lib/libmigraphx.so.2016000.0 \
    python demo_text.py --checkpoint model/sam3 --onnx-dir onnx_files_1008 \
    --video assets/blackswan.mp4 --text "swan" --mig --max-frames 60
```

Multi-object flags:
- `--min-score 0.5` — only track detections above this confidence (default 0.5)
- `--max-objects 0` — cap by score rank, 0 = all above threshold (default 0 = all)

```bash
# Track all dogs in the scene
python demo_text.py --checkpoint model/sam3 \
    --video assets/blackswan.mp4 --text "dog" \
    --imgsz 504 --mig --onnx-dir onnx_files_504 --min-score 0.4

# Track at most 2 people (highest scoring)
python demo_text.py --checkpoint model/sam3 \
    --video assets/blackswan.mp4 --text "person" \
    --imgsz 504 --mig --onnx-dir onnx_files_504 --max-objects 2
```

Outputs default to `outputs/{box,text}/<input-stem>_{tracked,text}.{jpg,mp4}` (overridable
with `--output`). Try short noun phrases: `"swan"`, `"a person on a bike"`, `"yellow taxi"`.

### Quick checks

```bash
conda activate sam3-tracker  # if not already active
```

| Script | Requires | What it checks | Time |
|---|---|---|---|
| `eval/probes/probe_text_prompt.py`     | Stage 1 only | Text-prompt detection (pure PyTorch) | ~10 s |
| `eval/benchmarks/bench_pipeline.py`    | Stage 2 box  | Per-module latency + total FPS       | ~30 s |
| `eval/probes/probe_text_prompt_mxr.py` | Stage 2 text | Text-prompt with MIGraphX backbone   | ~15 s |
| `eval/benchmarks/profile_text_prompt.py` | Stage 2 text | Per-stage latency of text-prompt   | ~30 s |

```bash
# After Stage 1 only:
python eval/probes/probe_text_prompt.py --checkpoint model/sam3 --image assets/truck.jpg --text "truck"

# After Stage 2 (box):
python eval/benchmarks/bench_pipeline.py --checkpoint model/sam3 --onnx-dir onnx_files_504

# After Stage 2 (text):
python eval/probes/probe_text_prompt_mxr.py --checkpoint model/sam3 --onnx-dir onnx_files_504 --image assets/truck.jpg --text "truck"
python eval/benchmarks/profile_text_prompt.py --checkpoint model/sam3 --image assets/truck.jpg --text "truck"
```

---

## Results

### Text-prompt: detection + tracking (`demo_text.py --mig --imgsz 504`)

| `"swan"` | `"camel"` | `"pig"` (3 objects) |
|:---:|:---:|:---:|
| <img src="docs/images/demo_swan_text_mig.gif" width="260" alt="swan"> | <img src="docs/images/demo_camel_text_mig.gif" width="260" alt="camel"> | <img src="docs/images/demo_pigs_multi_object.gif" width="260" alt="pigs multi-object"> |

### Box-prompt: tracking only (`demo.py`)

| truck — single image | dog-agility — video (8.87 FPS) |
|:---:|:---:|
| <img src="docs/images/demo_tracked.jpg" width="400" alt="truck box-prompt"> | <img src="docs/images/demo_dog_agility_box.gif" width="400" alt="dog-agility box-prompt"> |

---

## Performance

*To reproduce the accuracy numbers below, see the [Evaluation](#evaluation) section.*

### Text-prompt: Detection + Tracking (`demo_text.py`)

| Resolution | Path | Prop FPS | vs PT @1008 |
|---|---|---|---|
| **504px** | **MIG** (backbone + detr_encoder + memory_attention) | **5.5** | **10.6×** |
| 504px | PyTorch | 2.6 | 5.0× |
| **1008px** | **MIG** | **1.5** | **2.9×** |
| 1008px | PyTorch | 0.52 | baseline |

Mask quality: PT vs MIG mean IoU = **0.999** @1008px, **0.994** @504px (verified
frame-by-frame on 20–30 frames). Detection score: truck 0.95, swan 0.93–0.96 across resolutions.

Multi-object scaling @504 MIG (backbone shared across all objects):

| Objects tracked | Prop FPS |
|---|---|
| 1 | 5.5 |
| 4 (estimated) | ~4.4 |
| 8 (estimated) | ~2.9 |

### Box-prompt: Tracking only (`demo.py`)

| Resolution | DAVIS 2017 val J | Prop FPS | Backbone |
|---|---|---|---|
| **504px** | **81.6%** | **12.21** | MIGraphX 2.15+patches + MLIR |
| 1008px | **84.8%** | **3.22** | MIGraphX 2.15+patches + MLIR |
| 504px (PyTorch) | 81.6% | 5.72 | PyTorch ROCm FP16 |
| 1008px (PyTorch) | 84.8% | 1.35 | PyTorch ROCm FP16 |

> **Reference**: SAM2-L (official, GT first-frame mask) achieves **J&F=91.6%** on DAVIS 2017 val.
> Our box-prompt uses a box-derived mask on frame 0 instead of GT — the gap reflects prompt quality, not tracker propagation quality.


### Per-module latency breakdown (504px, MIGraphX backbone)

**Text-prompt propagation** (169 ms/frame → 5.9 FPS, with MLIR attention backbone):

| Stage | Latency | Backend |
|---|---:|---|
| backbone (vision encoder) | ~97 ms | MIGraphX .mxr + MLIR attention ops (FP16) |
| memory_attention | ~20 ms | ORT MIGraphX EP FP16 ¹ |
| detr_encoder | ~11 ms | ORT MIGraphX EP FP16 |
| detr_decoder | ~11 ms | PyTorch |
| tracker_neck + mask_decoder + memory_encoder | ~8 ms | PyTorch |
| **Total propagation frame** | **~169 ms → 5.9 FPS** | |

**Box-prompt propagation** (~82 ms/frame → 12.21 FPS, with MLIR attention backbone):

| Stage | Latency | Backend |
|---|---:|---|
| backbone (`backbone_mxr_tuned.mxr`) | ~67 ms | MIGraphX 2.15+patches + MLIR attention (FP16) |
| memory_attention | ~7 ms | ORT MIGraphX EP FP16 ¹ |
| mask_decoder_propagate (`dec_prop_fp32.mxr`) | ~14 ms | MIGraphX direct API FP32 |
| memory_encoder (`mem_enc_fp32.mxr`) | ~2 ms | MIGraphX direct API FP16 |
| **Total propagation frame** | **~82 ms → 12.21 FPS** | |

¹ `memory_attention` and `detr_encoder` run through ONNX Runtime's MIGraphX EP rather than
a precompiled `.mxr` because the direct MIGraphX FP16 attention kernel produces NaN outputs
(analogous to [ROCm/AMDMIGraphX#3596](https://github.com/ROCm/AMDMIGraphX/issues/3596)).
The ORT EP path uses a different FP16 quantization path that produces correct results.

### Backbone speed comparison (504px)

| Backbone | Latency | Speedup |
|---|---|---|
| MIGraphX 2.15+patches (autotuned) | **92 ms** | **1.5×** |
| PyTorch ROCm FP16 + TunableOp | 139 ms | baseline |
| MIGraphX 2.15.0 (stock, HF ONNX) | ~916 ms | 0.15× |

The 1.5× backbone speedup comes from two patches on top of MIGraphX 2.15:
1. A patch to `find_splits` ([AMDMIGraphX#4256](https://github.com/ROCm/AMDMIGraphX/issues/4256)) enabling fusion of the HF window-attention `Split` ops
2. Kernel autotuning (analogous to PyTorch TunableOp) selecting optimal GEMM kernels

Run `python eval/benchmarks/bench_pipeline.py --checkpoint model/sam3 --onnx-dir onnx_files_504` to reproduce.

*Measured on AMD Ryzen AI Max+ 395 (gfx1151).*

---

## Evaluation

### Download datasets

**DAVIS 2017 val** (semi-supervised, 480p):
```bash
# Download from the official DAVIS challenge site
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip -d dataset/
# Result: dataset/DAVIS/{Annotations,ImageSets,JPEGImages}/
```

> Official page: [davischallenge.org/davis2017/code.html](https://davischallenge.org/davis2017/code.html)

### Run evaluation

```bash
# DAVIS 2017 val (box-prompt)
python eval/datasets/eval_davis.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files_504 \
    --davis dataset/DAVIS \
    --imgsz 504

# PT vs MIG mask regression check
python eval/datasets/mask_diff_pt_vs_mig.py \
    --checkpoint model/sam3 --video assets/blackswan.mp4 \
    --text "swan" --imgsz 504 --max-frames 30 \
    --out results/eval/mask_diff_504.json

# Pipeline latency benchmark (box-prompt)
python eval/benchmarks/bench_pipeline.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files_504

# Per-module profile (text-prompt, full MIG stack)
python eval/benchmarks/profile_full_mig.py \
    --checkpoint model/sam3 --video assets/blackswan.mp4 \
    --text "swan" --imgsz 504 \
    --out results/profile_504_mig.json
```

---

## Project structure

```
sam3-tracker-rocm/
├── tracker/                # Tracker implementations
│   ├── tracker.py          #   SAM3OnnxTracker (box-prompt) + MIGraphXBackbone
│   ├── mig_vision_encoder.py    #   MIG shim: Sam3VideoModel vision_encoder
│   ├── mig_detr_encoder.py      #   MIG shim: Sam3DetrEncoder (ORT MIG EP)
│   └── mig_memory_attention.py  #   MIG shim: memory_attention (ORT MIG EP, padded)
├── export/                      # ONNX export + .mxr compile scripts
│   ├── build.py             # ← unified build entry point (box + text, any resolution)
│   ├── build_text_prompt_mig.py  #   text-prompt sub-script (called by build.py)
│   ├── backbone/            #   ViT backbone: export → simplify → MIGraphX compile
│   ├── detector/            #   DETR encoder export (text-prompt path)
│   └── tracker_modules/     #   mask_decoder_*, memory_*, memory_attention (padded)
├── eval/                   # Benchmarks, dataset evals, regression tools
│   ├── benchmarks/         #   bench_pipeline, profile_full_mig, profile_text_prompt
│   ├── datasets/           #   eval_davis, mask_diff_pt_vs_mig
│   ├── probes/             #   smoke tests for text-prompt + correctness checks
│   └── debug/              #   investigation scripts (FPN diagnostics, ONNX-CPU vs PT)
├── docs/                   # Setup guide, technical report
│   └── images/             #   README/doc visuals
├── model/sam3/             # Config + tokenizer (weights downloaded separately)
├── assets/                 # Demo inputs: demo.jpg, demo.mp4
├── onnx_files_504/         # Generated, gitignored — 504px MIG artefacts
│   ├── backbone_tracker/   #   Box-prompt: tuned.mxr + supporting ONNX
│   ├── backbone_detector/  #   Text-prompt: detector FPN + last_hidden_state
│   ├── detector_modules/   #   detr_encoder_simplified.onnx + ORT cache
│   └── tracker_modules/    #   mask_decoder_*, memory_*, memory_attention + caches
├── onnx_files_1008/        # Generated, gitignored — 1008px (same subdir structure)
├── outputs/                # Demo outputs (gitignored, auto-created)
├── results/                # Eval outputs: JSON metrics, profiles
├── dataset/                # Downloaded datasets (DAVIS)
├── demo.py                 # ← Box-prompt: image / video tracking demo
├── demo_text.py            # ← Text-prompt: open-vocabulary detection + tracking
├── setup.sh                # ← One-command setup
└── environment.yml
```

---

## Known limitations

- **MIGraphX backbone cold-start**: first compile of `backbone_mxr_tuned.mxr` takes
  ~3 min (504px) or ~9 min (1008px) with kernel autotuning. Subsequent runs load in ~3s.
  Run `python export/build.py --pipeline box --imgsz 504` once to pre-build the cache.
- **Text-prompt: vision_encoder dominates** (65% of propagation time at 504px, 57% at 1008px).
  The ViT backbone already runs via MIGraphX, but each frame requires a GPU→CPU→GPU numpy
  round-trip through the MIG bridge (~37 ms overhead at 1008px). Eliminating this round-trip
  (GPU-resident MIG via HIP IPC) is the primary remaining optimization target.
- **Text-prompt: modules under ~30 ms do not benefit from ORT MIG EP** — the CPU↔GPU
  round-trip overhead matches or exceeds the PT runtime. `detr_decoder` (~11–25 ms) was
  investigated and confirmed net-neutral; `mask_decoder` (~5 ms) and `memory_encoder`
  (~6 ms) are too small to MIG-ize profitably.
- **MIG attention modules MUST go through ORT MIG EP.** Direct
  `migraphx.parse_onnx + quantize_fp16` on attention layers (`memory_attention`,
  `detr_encoder`) produces NaN outputs (FP16 attention bug analogous to
  [ROCm/AMDMIGraphX#3596](https://github.com/ROCm/AMDMIGraphX/issues/3596));
  even FP32 produces ~0.05 max-diff that breaks downstream detection thresholds.
  ORT EP with `migraphx_fp16_enable=1` uses a different FP16 quantization path
  that produces correct results.
- **memory_attention K=64 cliff.** MIGraphX picks a 14× slower kernel at
  `num_object_pointer_tokens=64` (791 ms vs 55 ms at K≤32). The shim caps at K=32
  and truncates the oldest pointers — quality impact is invisible for continuous
  tracking; long-video re-identification across long disappearances may degrade slightly.
- **Box-prompt: `dec_propagate` FP16 corrupts results.** ConvTranspose upsampling is
  numerically sensitive — keep it at FP32 (`dec_prop_fp32.mxr`). All other modules run FP16.
- **MIGraphX 2.15+patches required** for box-prompt headline FPS. The stock MIGraphX 2.15.0
  from the ROCm 7.2 APT package produces ~916 ms for the HF backbone (6.6× slower) due to a
  fusion limitation in `find_splits`. See [`analysis/migraphx_backbone_investigation.md`](analysis/migraphx_backbone_investigation.md).
- **Dual LD_PRELOAD required for text-prompt MIG.** The torch ROCm nightly wheels bundle
  their own HIP runtime; loading MIGraphX after torch corrupts `.mxr` deserialization. The
  `LD_PRELOAD` in the demo commands forces `/opt/rocm-7.2.x` libs to load first.

---

## Acknowledgements

- **SAM3**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3) — model weights
  and architecture. Weights must be downloaded separately from
  [facebook/sam3](https://huggingface.co/facebook/sam3) on HuggingFace.
- **DART**: the `sam3_tracker_video` model class originates from the
  [DART](https://arxiv.org/abs/2603.11441) project's transformers fork, since merged
  into HuggingFace Transformers (≥ 5.7.0).
