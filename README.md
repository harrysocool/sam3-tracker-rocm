# SAM3 Video Tracker — ROCm / AMD

Mask-level video tracking pipeline built on [SAM3](https://github.com/facebookresearch/sam3),
optimized for AMD ROCm hardware. Achieves **8.21 FPS** (propagation frame) on an
AMD Ryzen AI Max+ 395 with a DAVIS 2017 val Mean J of **81.5%** (504px).

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

- **Frame 0**: user provides a bounding box → `mask_decoder_init.onnx` produces the initial mask
- **Frames 1+**: memory bank drives `mask_decoder_propagate.onnx` — no prompt needed
- **Backbone** runs via MIGraphX 2.15+patches (ONNX, no PyTorch required); tracking modules run via ONNX Runtime / MIGraphX

```
Input frame
  → backbone_mxr_tuned.mxr  (MIGraphX 2.15+patches)
  → memory_attention        (ORT MIGraphX EP) ¹
  → dec_prop_fp32.mxr       (MIGraphX)
  → mem_enc_fp32.mxr        (MIGraphX)
  ─────────────────────────────────────────────
  Total propagation frame: 8.21 FPS @ 504px (per-module timings: see Performance)
```

¹ `memory_attention` runs through ONNX Runtime's MIGraphX EP rather than a
precompiled `.mxr` because the direct MIGraphX FP16 attention kernel has a
numerical bug that breaks tracking (DAVIS J drops to ~2%). See
[ROCm/AMDMIGraphX#3596](https://github.com/ROCm/AMDMIGraphX/issues/3596).
The ORT EP path costs ~16 ms/frame vs the direct API (≈13% FPS) — mostly
host↔device transfer overhead, not the kernel itself. Recovering it is a known
optimization opportunity.

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

### Quick start (one command)

```bash
./setup.sh
```

Useful flags: `--skip-apt`, `--skip-migraphx`, `--env NAME`, `--imgsz 504/1008`.
See [setup.sh](setup.sh) for details.

### What setup.sh does

1. APT: ROCm 7.2 stack + stock MIGraphX 2.15.0 (`--skip-apt` to bypass)
2. Patched MIGraphX tarball (~2 min, two unreleased fixes for the headline FPS — `--skip-migraphx` to bypass)
3. Conda env (`sam3-tracker` by default; override with `--env`) with Python 3.12
4. ROCm 7.13 nightly SDK + PyTorch (gfx1151 wheels, ~2–5 min)
5. ONNX Runtime MIGraphX EP wheel (1.24.2)
6. Python dependencies from `requirements.txt`
7. Model weights from community mirror `1038lab/sam3` (no HF account needed)
8. Tracker module ONNX export — `memory_attention`, `mask_decoder_*`, `memory_encoder` (~5 min)
9. Backbone `.mxr` compile (single-session export → onnxsim → MIGraphX autotune; ~5 min @ 504px, ~12 min @ 1008px)
10. Pre-warm ORT MIGraphX cache (~1 min)
11. Smoke test (`demo.py` on `assets/demo.jpg`)

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

Two demo entry points — pick the one matching your prompt type:

| Demo | Prompt | Backbone | Latency @504px | Use for |
|---|---|---|---|---|
| `demo.py`      | bounding box  | MIGraphX (fastest)        | ~115 ms (8.21 FPS) | known target, real-time |
| `demo_text.py` | text          | PyTorch (CLIP detector)   | ~2 s init + slow propagation | open-vocabulary, prototyping |

All commands assume you ran `./setup.sh` and are in the project root. The MIGraphX
backbone requires `/opt/rocm-7.2.0/lib` on `PYTHONPATH`; PyTorch path doesn't:
```bash
export PYTHONPATH=/opt/rocm-7.2.0/lib${PYTHONPATH:+:$PYTHONPATH}
```

### Box-prompt (`demo.py`) — fastest

```bash
# Image — MIGraphX backbone (default, ~115 ms / frame)
python demo.py --checkpoint model/sam3 --onnx-dir onnx_files \
    --image assets/demo.jpg --box 85,281,1710,850

# Image — PyTorch backbone fallback (no MIGraphX needed)
python demo.py --checkpoint model/sam3 --onnx-dir onnx_files \
    --backbone pytorch \
    --image assets/demo.jpg --box 85,281,1710,850

# Video (any mp4) — same FPS as image, written to outputs/<stem>_tracked.mp4
# (assets/demo.mp4 is 854x480; box catches the swan in frame 0)
python demo.py --checkpoint model/sam3 --onnx-dir onnx_files \
    --video assets/demo.mp4 --box 320,170,650,400
```

### Text-prompt (`demo_text.py`) — open-vocabulary

```bash
# Image — text → CLIP detection → mask
python demo_text.py --checkpoint model/sam3 \
    --image assets/demo.jpg --text "truck"

# Video — text init on frame 0, then PyTorch tracker propagates
python demo_text.py --checkpoint model/sam3 \
    --video assets/demo.mp4 --text "swan" --max-frames 60
```

Outputs default to `outputs/<input-stem>_{tracked,text}.{jpg,mp4}` (overridable
with `--output`). Try short noun phrases: `"swan"`, `"a person on a bike"`,
`"yellow taxi"`.

### Quick checks (no dataset needed)

After `setup.sh`, four small scripts smoke-test specific aspects of the pipeline
using only `assets/demo.jpg`:

| Script | What it checks | Time |
|---|---|---|
| `eval/bench_pipeline.py`        | Per-module latency + total FPS — does your machine match the headline 8.21 FPS? | ~30 s |
| `eval/probe_text_prompt.py`     | Text-prompt detection works (PyTorch path)             | ~10 s |
| `eval/probe_text_prompt_mxr.py` | Text-prompt with MIGraphX backbone                     | ~15 s |
| `eval/profile_text_prompt.py`   | Per-stage latency of text-prompt path                  | ~30 s |

```bash
python eval/bench_pipeline.py        --checkpoint model/sam3 --onnx-dir onnx_files
python eval/probe_text_prompt.py     --checkpoint model/sam3 --image assets/demo.jpg --text "truck"
python eval/probe_text_prompt_mxr.py --checkpoint model/sam3 --onnx-dir onnx_files --image assets/demo.jpg --text "truck"
python eval/profile_text_prompt.py   --checkpoint model/sam3 --image assets/demo.jpg --text "truck"
```

---

## Results

### Single-image segmentation (box prompt)

| truck (demo) | drift-straight (J = 95.2%) | parkour (J = 92.2%) |
|:---:|:---:|:---:|
| <img src="docs/images/demo_tracked.jpg" width="320" alt="truck"> | <img src="docs/images/demo_drift-straight.jpg" width="320" alt="drift-straight"> | <img src="docs/images/demo_parkour.jpg" width="320" alt="parkour"> |

### Video tracking (DAVIS 2017 val, 504px)

| blackswan  (J = 93.0%) | dog  (J = 94.7%) | camel  (J = 96.0%) |
|:---:|:---:|:---:|
| <img src="docs/images/demo_blackswan.gif" width="320" alt="blackswan"> | <img src="docs/images/demo_dog.gif" width="320" alt="dog"> | <img src="docs/images/demo_camel.gif" width="320" alt="camel"> |

---

## Performance

*To reproduce the accuracy numbers below, see the [Evaluation](#evaluation) section.*

### Video tracking (propagation FPS)

| Resolution | DAVIS 2017 val J | SG val J (50 seqs) ¹ | Propagation FPS | Backbone |
|---|---|---|---|---|
| **504px** | **81.5%** | **40.4%** | **8.21** | MIGraphX 2.15+patches |
| 1008px | 84.8% | 44.0% | **2.31** | MIGraphX 2.15+patches |
| 504px (PyTorch) | 81.5% | 40.4% | 5.72 | PyTorch ROCm FP16 |
| 1008px (PyTorch) | 84.8% | 44.0% | 1.35 | PyTorch ROCm FP16 |

*MIGraphX backbone uses `backbone_mxr_tuned.mxr` (pre-compiled with kernel autotuning).
PyTorch baseline uses TunableOp-autotuned GEMM kernels.*

¹ SG J (IoU) is a proxy metric on a random 50-sequence subset, not the official cgF1/pHOTA
evaluation. Official SG evaluation pending (requires full 1686-annotation run with text prompts).

### Per-module latency breakdown (504px, MIGraphX backbone)

| Stage | Latency | Backend |
|---|---:|---|
| backbone (`backbone_mxr_tuned.mxr`) | ~92 ms | MIGraphX 2.15+patches GPU (FP16 internal) |
| memory_attention (ORT MIGraphX EP FP16) | ~7 ms | ORT MIGraphX EP (direct API has kernel bug) |
| mask_decoder_propagate (`dec_prop_fp32.mxr`) | ~14 ms | MIGraphX direct API FP32 |
| memory_encoder (`mem_enc_fp32.mxr`) | ~2 ms | MIGraphX direct API FP16 |
| **Total propagation frame** | **~115 ms → 8.21 FPS** | |

### Backbone speed comparison (504px)

| Backbone | Latency | Speedup |
|---|---|---|
| MIGraphX 2.15+patches (autotuned) | **92 ms** | **1.5×** |
| PyTorch ROCm FP16 + TunableOp | 139 ms | baseline |
| MIGraphX 2.15.0 (stock, HF ONNX) | ~916 ms | 0.15× |

The 1.5× backbone speedup comes from two patches on top of MIGraphX 2.15:
1. A patch to `find_splits` ([AMDMIGraphX#4256](https://github.com/ROCm/AMDMIGraphX/issues/4256)) enabling fusion of the HF window-attention `Split` ops
2. Kernel autotuning (analogous to PyTorch TunableOp) selecting optimal GEMM kernels

Run `python eval/bench_pipeline.py --checkpoint model/sam3 --onnx-dir onnx_files` to reproduce.

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

**Smartglass SG val** (SA-Co/VEval) — public Roboflow mirror, no account needed:

```bash
mkdir -p dataset
wget https://sa-co.roboflow.com/veval/saco_sg.zip -P dataset/        # ~30 GB frames
wget https://sa-co.roboflow.com/veval/gt-annotations.zip -P dataset/ # ~117 MB annotations
unzip dataset/saco_sg.zip -d dataset/
unzip dataset/gt-annotations.zip -d dataset/gt-annotations/
# Result: dataset/saco_sg/JPEGImages_6fps/ and dataset/gt-annotations/annotation/saco_veval_smartglasses_val.json
```

> Currently only the **SG (SmartGlasses)** subset is set up. The full SA-Co/VEval
> bundle also includes **SAV** and **YT1B** subsets but is significantly larger;
> evaluating on those is a future TODO.
>
> Also available (gated) on HuggingFace: [facebook/SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval). Full mirror bundle (all subsets): [sa-co.roboflow.com/veval/all.zip](https://sa-co.roboflow.com/veval/all.zip).

### Run evaluation

```bash
# DAVIS 2017 val
python eval/eval_davis.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files \
    --davis dataset/DAVIS \
    --imgsz 504

# Smartglass SG val
python eval/eval_saco_sg.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files \
    --gt-json dataset/gt-annotations/saco_veval_smartglasses_val.json \
    --img-root dataset/saco_sg/JPEGImages_6fps \
    --imgsz 504

# Pipeline A vs B latency benchmark
python eval/bench_pipeline.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files
```

---

## Project structure

```
sam3-tracker-rocm/
├── tracker/            # SAM3OnnxTracker — propagation pipeline
├── export/             # ONNX export + .mxr compile + ORT cache prewarm
├── eval/               # DAVIS / SG evaluation, benchmarks, probes
├── analysis/           # optimization deep-dives (markdown)
├── tools/              # patched MIGraphX install helper
├── docs/               # setup guide, technical report
│   └── images/         # README/doc visuals (committed)
├── model/sam3/         # config + tokenizer (weights downloaded separately)
├── assets/             # source inputs for demos: demo.jpg, demo.mp4
├── onnx_files/         # generated, gitignored — 504px ONNX modules
├── onnx_files_1008/    # generated, gitignored — 1008px ONNX modules
├── outputs/            # demo / probe outputs (gitignored, auto-created)
├── results/            # eval outputs (json, plots)
├── dataset/            # downloaded datasets (DAVIS, saco_sg)
├── demo.py             # ← entry point: box-prompt image / video demo
├── demo_text.py        # ← entry point: text-prompt image / video demo
├── setup.sh            # ← entry point: one-command setup
└── environment.yml
```

---

## Known limitations

- **MIGraphX backbone cold-start**: first compile of `backbone_mxr_tuned.mxr` takes
  ~3 min (504px) or ~9 min (1008px) with kernel autotuning. Subsequent runs load in ~3s.
  Run `export/export_backbone_single.py` once per resolution to pre-build the cache.
- **MIGraphX memory_attention cold-start**: first run JIT-compiles
  `memory_attention_fixed_N7.onnx` (~6s at 504px). Subsequent runs use the ORT cache.
- **`dec_propagate` FP16 corrupts results**: ConvTranspose upsampling is numerically
  sensitive — keep it at FP32 (`dec_prop_fp32.mxr`). All other modules run FP16.
- **MIGraphX 2.15+patches required**: the stock MIGraphX 2.15.0 from the ROCm 7.2
  APT package produces ~916ms for the HF backbone (6.6× slower) due to a fusion
  limitation in `find_splits`. See [`analysis/migraphx_backbone_investigation.md`](analysis/migraphx_backbone_investigation.md) for details.
- **Text-prompt path is PyTorch-only** (no MIGraphX acceleration yet). `demo_text.py`
  runs the entire pipeline on PyTorch, so propagation drops from box-prompt's
  8.21 FPS to ~0.5 FPS. Bringing text-prompt up to box-prompt FPS requires
  MIGraphX-izing the detector module — concretely:
    1. Re-export the backbone with `last_hidden_state` (the detector needs the
       1024-d ViT features, not just the 256-d FPN).
    2. Wire the CLIP text encoder + DETR head into a hybrid path
       (PyTorch on frame 0, MIGraphX after).
    3. Port `Sam3VideoModel`'s NMS + presence gating to the OnnxTracker side.
    4. Resolve the torch ROCm 7.13 nightly vs patched MIGraphX 7.2 library
       conflict (LD_PRELOAD workaround already documented in memory).

  Estimated 2–3 days of work. Until then, treat `demo_text.py` as a prototyping
  / open-vocabulary tool, not a real-time path.

---

## Acknowledgements

- **SAM3**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3) — model weights
  and architecture. Weights must be downloaded separately from
  [facebook/sam3](https://huggingface.co/facebook/sam3) on HuggingFace.
- **DART**: the `sam3_tracker_video` model class originates from the
  [DART](https://arxiv.org/abs/2603.11441) project's transformers fork, since merged
  into HuggingFace Transformers (≥ 5.7.0).
