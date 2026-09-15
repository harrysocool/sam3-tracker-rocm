# Historical host runtime and reference results

> **Archive — not current setup instructions.** This page preserves the legacy
> sections formerly carried in the root README at `v0.2.0-rc4` (`afd712a`).
> They span several earlier runtime/optimization iterations. Commands,
> installer flags, artifact layouts, and claims of a "default" or "headline"
> result refer to those historical iterations, not the current checkout.
> Some referenced installer scripts no longer exist. Do not execute these
> instructions against the current release or use them as a host fallback.

Historical weight-download instructions are omitted. Model weights must be
obtained independently under the separate SAM License before use; see the
current [requirements](../../README.md#requirements).

Use the [current Quick start](../../README.md#quick-start) for ROCm 7.14 / MIGraphX
2.17 live deployment. [Current performance records](../../docs/performance.md) separate
live, hybrid, offline, and box workloads. Existing baseline artifacts remain
immutable; never overwrite them while rebuilding for another runtime.

## Historical host installation

The material below documents the original implementation. Its commands no
longer describe the current `setup.sh` and are not a supported fallback.

### Prerequisites

Have these in place **before** running `./setup.sh`:

| Requirement | Handled by | Notes |
|---|---|---|
| Hardware: AMD Ryzen AI Max+ 395 (gfx1151) | You | Other ROCm-capable AMD GPUs may work but are untested |
| OS: Ubuntu 24.04.4 LTS | You | Other Linux distros with ROCm 7.x support may work |
| Kernel: 6.8+ (tested: 6.18.6) | You | Required for gfx1151 AMDGPU driver support |
| **conda / miniforge** (any recent) | ⚠️ You — install before running setup.sh | `setup.sh` errors out if conda is not found. [Install miniforge](https://github.com/conda-forge/miniforge) |
| **BIOS UMA Frame Buffer Size = 64 GB** | ⚠️ You — set in BIOS | **128 GB systems only** — without this, backbone OOMs at 1008px. See [Finding #7](project-summary.md#key-findings). |
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
See [setup.sh](../../setup.sh) for details.

What it does:
1. APT: ROCm 7.2 stack + stock MIGraphX 2.15.0 (`--skip-apt` to bypass)
2. Patched MIGraphX tarball (~2 min, two unreleased fixes for the headline FPS — `--skip-migraphx` to bypass)
3. Conda env (`sam3-tracker` by default; override with `--env`) with Python 3.12
4. ROCm 7.13 nightly SDK + PyTorch (gfx1151 wheels, ~2–5 min)
5. ONNX Runtime MIGraphX EP wheel (1.24.2)
6. Python dependencies from `requirements.txt`

### Stage 2 — Build model artefacts (`export/build.py`)

After `setup.sh`, activate the environment and build artefacts for the pipeline(s) you want:

```bash
conda activate sam3-tracker

# Text-prompt MIG — demo_live.py / tools/text_baseline.py --mig  (~21 min @504px)
python export/build.py --pipeline text --imgsz 504

# Box-prompt only — demo_box.py  (~10 min @504px)
python export/build.py --pipeline box --imgsz 504

# Both pipelines at 504px (recommended)
python export/build.py --pipeline all --imgsz 504
```

Each step skips if output already exists — safe to re-run after interruption.
Use `--force` to rebuild from scratch.

The text build retains `backbone_detector/tuned.mxr` as the host-I/O fallback
and additionally creates `backbone_detector/tuned_gpuio.mxr`. Text inference
prefers the GPU-I/O artifact automatically when it is present.

<details>
<summary><b>1008px (higher mask quality, 3-10× slower)</b></summary>

1008px is supported as an advanced option but isn't the recommended path. Build with:

```bash
python export/build.py --pipeline all --imgsz 1008
# or both resolutions in one run (~90 min total):
python export/build.py --pipeline all --imgsz 504 1008
```

Requires `BIOS UMA Frame Buffer Size = 64 GB` on 128 GB systems to avoid backbone OOM.

</details>

### Manual / alternative paths

<details>
<summary><b>Patched MIGraphX — build from source</b></summary>

The headline FPS requires two unreleased MIGraphX fixes (a `find_splits` patch +
NHWC output fix). `setup.sh` installs a prebuilt tarball; if you'd rather build:

| Path | FPS (504 / 1008 px) | What you need |
|---|---|---|
| Stock APT 2.15.0 | 5.72 / 1.35 | Checkout tag `v0.1-migraphx-2.15` |
| **Prebuilt tarball** (default) | **8.21 / 2.31** | `setup.sh` downloads + installs |
| Build from source | 8.21 / 2.31 | See the [MIGraphX 2.15 patch guide](migraphx-2.15-patches.md) |

</details>

<details>
<summary><b>Step-by-step manual install (no setup.sh)</b></summary>

For full control over each step (APT, conda, pip, ONNX export, backbone compile)
see the [historical manual host setup guide](manual-host-setup.md).

</details>

## Legacy native text usage

The following commands belong to the old host environment above. Some example
videos were local experiment inputs rather than tracked repository assets.
For current defaults and container commands, use [the usage guide](../../docs/usage.md).

> `assets/blackswan.mp4` is a bundled swan clip. Replace with your own video.
> MIG commands (`--mig`) require Stage 2 artefacts to be built first.

```bash
# Image — pure PyTorch path (no MIG artifacts needed)
python tools/text_baseline.py --checkpoint model/sam3 \
    --image assets/truck.jpg --text "truck"

# Video — pure PyTorch baseline
python tools/text_baseline.py --checkpoint model/sam3 \
    --video assets/blackswan.mp4 --text "swan" --max-frames 60

# Video — native compatibility-stack MIG @504 (~7.06 FPS)
LD_PRELOAD=/opt/rocm-7.2.x/lib/libmigraphx_c.so.3:/opt/rocm-7.2.x/lib/migraphx/lib/libmigraphx.so.2016000.0 \
    python tools/text_baseline.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --video assets/blackswan.mp4 --text "swan" --imgsz 504 --mig --max-frames 60
```

`--parallel-tail` overlaps the independent detector and tracker branches after
the shared backbone. It is opt-in and currently validated at 504 px with the
ROCm 7.14 container; add it to the container command documented in
[`docker/rocm714/README.md`](../../docker/rocm714/README.md).

For preloaded videos, add `--pipeline-backbone` as well. It computes frame
N+1's stateless vision backbone while frame N's detector/tracker tail runs.
This improves throughput but adds a one-frame pipeline fill, so it is not used
by the low-latency streaming API. The MIG vision shim also caches its four
fixed sine position encodings locally, preventing the detector and tracker
instances from thrashing Transformers' shared four-entry cache.

**Legacy offline multi-object flags** (`text_baseline.py`; not current live defaults):
- `--min-score 0.5` — only track detections above this confidence (default 0.5)
- `--max-objects 0` — cap by score rank, 0 = all above threshold (default 0 = all)

```bash
# Track every person above 0.4 confidence
python tools/text_baseline.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --video assets/two_person_dog_lawn.mp4 --text "person" \
    --imgsz 504 --mig --min-score 0.4

# Track at most 2 people (highest scoring)
python tools/text_baseline.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --video assets/two_person_dog_lawn.mp4 --text "person" \
    --imgsz 504 --mig --max-objects 2
```

<details>
<summary><b>1008px text-prompt (higher quality, ~1.5 FPS)</b></summary>

```bash
LD_PRELOAD=/opt/rocm-7.2.x/lib/libmigraphx_c.so.3:/opt/rocm-7.2.x/lib/migraphx/lib/libmigraphx.so.2016000.0 \
    python tools/text_baseline.py --checkpoint model/sam3 --onnx-dir onnx_files_1008 \
    --video assets/blackswan.mp4 --text "swan" --mig --max-frames 60
```

</details>

## Box-prompt usage

These commands require the old single-object tracker artifact set. The current
text/live Quick start does not build it, and these timings are not default
live measurements.

> `assets/truck.jpg` and `assets/blackswan.mp4` are bundled demo files. Replace with
> your own image or video. `--box x1,y1,x2,y2` is the bounding box around the target on
> frame 0, in pixel coordinates.

```bash
# Image — MIGraphX backbone (default, ~115 ms / frame)
python demo_box.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --image assets/truck.jpg --box 85,281,1710,850

# Video (any mp4) — output written to demo_out/box/<stem>_tracked.mp4
python demo_box.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --video assets/blackswan.mp4 --box 320,170,650,400
```

Outputs default to `demo_out/{box,text}/<input-stem>_{tracked,text}.{jpg,mp4}` (overridable
with `--output`). Try short noun phrases: `"swan"`, `"a person on a bike"`, `"yellow taxi"`.

## Historical quick checks

```bash
conda activate sam3-tracker  # if not already active
```

| Script | Requires | What it checks | Time |
|---|---|---|---|
| `eval/historical/probes/probe_text_prompt.py` | Stage 1 only | Text-prompt detection (pure PyTorch) | ~10 s |
| `eval/benchmarks/bench_pipeline.py`    | Stage 2 box  | Per-module latency + total FPS       | ~30 s |
| `eval/historical/probes/probe_text_prompt_mxr.py` | Stage 2 text | Text-prompt with MIGraphX backbone | ~15 s |
| `eval/historical/profilers/profile_text_prompt.py` | Stage 2 text | Per-stage latency of text-prompt | ~30 s |

```bash
# After Stage 1 only:
python eval/historical/probes/probe_text_prompt.py --checkpoint model/sam3 --image assets/truck.jpg --text "truck"

# After Stage 2 (box):
python eval/benchmarks/bench_pipeline.py --checkpoint model/sam3 --onnx-dir onnx_files_504

# After Stage 2 (text):
python eval/historical/probes/probe_text_prompt_mxr.py --checkpoint model/sam3 --onnx-dir onnx_files_504 --image assets/truck.jpg --text "truck"
python eval/historical/profilers/profile_text_prompt.py --checkpoint model/sam3 --image assets/truck.jpg --text "truck"
```

## Native and box benchmark snapshots

All numbers below are historical observations on Ryzen AI Max+ 395, not
fresh measurements of the current release. Treat estimates as estimates.

| Historical text configuration | Recorded FPS at 504px |
|---|---:|
| Native compatibility-stack MIG, one object | 7.06 |
| Pure PyTorch reference | approximately 2.6 |

Multi-object scaling @504 MIG (backbone shared across all objects):

| Objects tracked | tools/text_baseline.py prop FPS |
|---|---|
| 1 | **7.06** |
| 4 | ~4.4 (pre-GPU-I/O baseline; re-benchmark pending) |
| 8 (estimated) | ~2.9 (pre-GPU-I/O estimate) |

### Box-prompt (`demo_box.py`) — specialized, fastest

| Backbone | DAVIS 2017 val J | Prop FPS |
|---|---|---|
| **MIGraphX 2.15+patches + MLIR** | **81.6%** | **12.21** |
| PyTorch ROCm FP16 | 81.6% | 5.72 |

> **Reference**: SAM2-L (official, GT first-frame mask) achieves **J&F=91.6%** on DAVIS 2017 val.
> These numbers use different metrics (J versus J&F) and different initial-mask
> protocols. They are not directly comparable, and the gap cannot be attributed
> to prompt quality alone.

<details>
<summary><b>1008px numbers (higher quality, 3-10× slower)</b></summary>

| Demo | Resolution | DAVIS J | Prop FPS |
|---|---|---|---|
| `tools/text_baseline.py --mig` | 1008px | — | 1.5 |
| `tools/text_baseline.py` (no MIG) | 1008px | — | 0.52 |
| `demo_box.py` (MIG) | 1008px | 84.8% | 3.22 |
| `demo_box.py` (PyTorch) | 1008px | 84.8% | 1.35 |

Mask quality (text-prompt): PT vs MIG mean IoU = 0.999 @1008px.

1008px deep-dive: [`docs/historical/1008px_perf_analysis.md`](../../docs/historical/1008px_perf_analysis.md).

</details>


### Per-module latency breakdown (504px, MIGraphX backbone)

**Text-prompt propagation** (137 ms/frame in the 48-frame module profile;
7.06 FPS measured end-to-end over 47 propagation frames) — GPU-resident MLIR
attention backbone plus ORT GPU I/O binding:

| Stage | Latency | Backend |
|---|---:|---|
| backbone (vision encoder) | ~67 ms | MIGraphX `tuned_gpuio.mxr` + MLIR attention ops |
| memory_attention | ~15 ms average ² | ORT MIGraphX EP FP16 + GPU I/O binding ¹ |
| detr_encoder | ~7 ms | ORT MIGraphX EP FP16 + GPU I/O binding |
| detr_decoder | ~11 ms | PyTorch |
| tracker_neck + mask_decoder + memory_encoder | ~8 ms | PyTorch |
| **Total propagation frame** | **~137 ms → 7.3 FPS profile / 7.06 FPS E2E** | |

**Box-prompt propagation** (~82 ms/frame → 12.21 FPS, with MLIR attention backbone):

| Stage | Latency | Backend |
|---|---:|---|
| backbone (`backbone_tracker/tuned.mxr`) | ~67 ms | MIGraphX 2.15+patches + MLIR attention (FP16) |
| memory_attention | ~7 ms | ORT MIGraphX EP FP16 ¹ |
| mask_decoder_propagate (`dec_prop_fp32.mxr`) | ~14 ms | MIGraphX direct API FP32 |
| memory_encoder (`mem_enc_fp32.mxr`) | ~2 ms | MIGraphX direct API FP16 |
| **Total propagation frame** | **~82 ms → 12.21 FPS** | |

¹ `memory_attention` and `detr_encoder` run through ONNX Runtime's MIGraphX EP rather than
a precompiled `.mxr` because the direct MIGraphX FP16 attention kernel produces NaN outputs
(analogous to [ROCm/AMDMIGraphX#3596](https://github.com/ROCm/AMDMIGraphX/issues/3596)).
The ORT EP path uses a different FP16 quantization path that produces correct results.

² Exact S1…S10 graphs avoid dynamic recompilation and PyTorch fallback as the
memory bank and conditioning frames grow. The S7 GPU-I/O-bound ORT call is
~8.4 ms in steady-state microbenchmarks.

The then-experimental ROCm 7.14 Docker stack recorded **111.65 ms/frame (8.96 FPS)** in
the module profile and **8.51 FPS end-to-end**. See
[`docs/rocm714_fullstack_evaluation.md`](../../docs/rocm714_fullstack_evaluation.md).

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

## DAVIS and box evaluation

**DAVIS 2017 val** (semi-supervised, 480p):
```bash
# Download from the official DAVIS challenge site
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip -d dataset/
# Result: dataset/DAVIS/{Annotations,ImageSets,JPEGImages}/
```

> Official page: [davischallenge.org/davis2017/code.html](https://davischallenge.org/davis2017/code.html)

```bash
# Historical box-prompt DAVIS evaluation; requires the matching box artifact set
python eval/datasets/eval_davis.py \
    --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --davis dataset/DAVIS --imgsz 504

# Historical box-pipeline latency benchmark
python eval/benchmarks/bench_pipeline.py \
    --checkpoint model/sam3 --onnx-dir onnx_files_504
```

Tracker regressions still need the DAVIS protocol, but these saved results do
not validate a new runtime or newly compiled artifacts. Current text/live
checks are documented in the [evaluation guide](../../docs/evaluation.md).

## Historical implementation constraints

These findings describe the then-current implementations. In particular,
the old ORT decoder result does not describe the now-accepted direct-MXR
fixed decoder, and MIGraphX 2.15 is no longer the deployment requirement.

| Limitation | Detail / workaround |
|---|---|
| **Backbone cold-start** | First `.mxr` compile takes ~3 min (504px) / ~9 min (1008px) with autotuning; then loads in ~3s. Pre-build once: `python export/build.py --pipeline box --imgsz 504`. |
| **Text-prompt: vision_encoder dominates** | About 49% of the 504px profile after switching to GPU-resident MIGraphX I/O. The 1008px GPU-I/O artifact has not yet been benchmarked and falls back to `tuned.mxr` until built. |
| **Small modules don't gain from ORT MIG EP** | Under ~30 ms the CPU↔GPU round-trip ≥ PT runtime. `detr_decoder` (~11–25 ms) confirmed net-neutral; `mask_decoder` (~5 ms) / `memory_encoder` (~6 ms) too small to MIG-ize. |
| **MIG attention must use ORT MIG EP** | Direct `parse_onnx + quantize_fp16` on `memory_attention` / `detr_encoder` yields NaN ([AMDMIGraphX#3596](https://github.com/ROCm/AMDMIGraphX/issues/3596)); even FP32 has ~0.05 max-diff that breaks detection thresholds. ORT EP with `migraphx_fp16_enable=1` is correct. |
| **memory_attention K=64 cliff at 1008px** | MIGraphX picks a much slower kernel at K=64, so 1008px uses K=48. The 504px path has no K=64 cliff and retains capacity for 16 objects. |
| **Box-prompt `dec_propagate` stays FP32** | ConvTranspose upsampling is numerically sensitive; keep it FP32 (`dec_prop_fp32.mxr`). All other modules run FP16. |
| **MIGraphX 2.15+patches required** | Stock 2.15.0 (ROCm 7.2 APT) runs the HF backbone in ~916 ms (6.6× slower) due to a `find_splits` fusion limit. See [analysis](../../analysis/migraphx_backbone_investigation.md). |
| **Dual LD_PRELOAD for text-prompt MIG** | torch ROCm nightly bundles its own HIP runtime; loading MIGraphX after torch corrupts `.mxr` deserialization. `LD_PRELOAD` forces `/opt/rocm-7.2.x` libs to load first. |

## Further history

- [Manual host installation](manual-host-setup.md)
- [MIGraphX 2.15 patch installation / source build](migraphx-2.15-patches.md)
- [Historical project summary](project-summary.md)
- [August 2026 ROCm 7.14 offline evaluation](../../docs/rocm714_fullstack_evaluation.md)
- [1008px performance analysis](../../docs/historical/1008px_perf_analysis.md)
- [Backbone investigation](../../analysis/migraphx_backbone_investigation.md)
