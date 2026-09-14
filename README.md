# SAM3 Video Tracking on AMD GPUs

Text-prompted video segmentation and tracking on AMD ROCm, built on
[Meta's SAM3](https://github.com/facebookresearch/sam3) and accelerated with
MIGraphX. Describe a target, such as `"swan"` or `"person on a bike"`, to detect
and track its masks through video.

The **streaming API prioritizes fresh observations**: it processes the newest
available frame and runs full text detection on every consumed frame by
default. Includes a video demo, a [ROS 2 integration skeleton](examples/README.md),
and offline text- and box-prompt reference tools.

<img src="docs/images/demo_swan_text_mig.gif" width="480" alt="Text-prompted swan segmentation across video frames">

*Prompt: `"swan"`. Qualitative example from the offline text-prompt path, not a
recording of the live performance benchmark.*

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Performance](#performance)
- [Documentation](#documentation)
- [Known limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Requirements

| Component | Supported / required |
|---|---|
| Validated GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`); other AMD GPUs are untested |
| Host | Linux x86-64, with an AMDGPU driver exposing `/dev/kfd` and `/dev/dri` |
| Tools | Docker with BuildKit, permission to run Docker, Git, and `curl` |
| Network | Access to Docker Hub, GitHub releases, AMD repositories, PyPI, and Hugging Face for the checkpoint |
| Disk | At least 30 GiB free; 40 GiB is recommended for clean rebuilds and validation |
| Runtime | ROCm **7.14**, MIGraphX **2.17**, ONNX Runtime **1.24.2**, installed inside the container |

Use `docker/rocm714/run.sh` for the optimized GPU path. No host conda environment
is needed. The host MIGraphX 2.16 stack is not supported for deployment and
cannot load the current fixed-decoder artifact. See the
[container guide](docker/rocm714/README.md) for exact dependency versions.

## Quick start

These steps use the current supported **`main` branch** and assemble the
versioned **v0.2.0-rc4 runtime** while building the recommended **504px text /
live pipeline**. Run the commands in Bash, in order, from the same shell. An
existing checkout of `main` or a compatible development branch can skip the
clone.

### 1. Get the source and checkpoint

```bash
git clone --branch main --single-branch \
  https://github.com/harrysocool/sam3-tracker-rocm.git
cd sam3-tracker-rocm
export SAM3_MODEL_DIR="$PWD/model/sam3"
```

The checkout includes model configuration and tokenizer files, **not weights**.
Request access and accept the terms at
[facebook/sam3](https://huggingface.co/facebook/sam3). With the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) installed
on the host, download the checkpoint separately:

```bash
hf auth login
hf download facebook/sam3 model.safetensors --local-dir "$SAM3_MODEL_DIR"
```

Already have the checkpoint? Skip the download and set `SAM3_MODEL_DIR` to the
absolute path of your complete model directory, containing `model.safetensors`
alongside the config and tokenizer files. Do not redownload over an existing
weight symlink. The weights remain subject to the separate SAM License.

### 2. Assemble the runtime

```bash
./setup.sh --runtime
```

This downloads checksum-pinned MIGraphX and ORT binaries, installs AMD ROCm and
Torch packages, and assembles a local Docker image. It does **not** compile the
runtime stack or install ROCm on the host. The script checks GPU visibility and
runtime versions after assembly.

Pip may report that the gfx1151 Torch wheel requires the Python
`rocm[libraries]` package. This warning is expected: the image deliberately uses
the system ROCm 7.14 libraries instead, with the included `rocm_sdk`
compatibility module. Treat the final GPU/provider smoke result, not that
resolver warning, as the runtime assembly result.

### 3. Build the model artifacts

Choose a new artifact directory outside the checkout. To resume an interrupted
build, reuse that same directory; do not point it at older runtime artifacts.

```bash
export SAM3_MODEL_BUILD_ROOT="$HOME/sam3-artifacts/gpu/build-0.2.0-rc4"
export SAM3_ONNX_DIR="$SAM3_MODEL_BUILD_ROOT/onnx_files_504"
./setup.sh --models "$SAM3_MODEL_DIR"
```

This exports ONNX and builds the 504px model artifacts inside the container,
including the GPU-I/O backbone and fixed DETR decoder. The initial build is a
one-time compilation step; completed export / compile steps are skipped on
rerun. ORT modules also compile and cache graphs on first use, so the first demo
startup can take longer than later runs.

**Keep both runtime directory variables set when running demos.** The wrapper
mounts host `SAM3_MODEL_DIR` at `/models/sam3` and host `SAM3_ONNX_DIR` at
`/models/onnx_files_504`. In a new shell, re-export those two absolute paths.
This explicit configuration does not depend on the development machine's
`onnx_files_504_mgx217` symlink. ONNX/MXR artifacts and caches are built locally;
no compiled SAM3 model bundle is downloaded.

### 4. Verify the installation and run the demo

```bash
./docker/rocm714/run.sh python -c \
  "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

Expect **1.24.2** and **MIGraphXExecutionProvider** in the provider list. A
VitisAI-only provider list is the NPU environment, not this GPU runtime.

```bash
./docker/rocm714/run.sh python tools/smoke_live_release.py \
  --checkpoint /models/sam3 \
  --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan \
  --frames 12 --mode both \
  --output results/perf/installation-smoke.json
```

Expect `Installation smoke PASS`. This verifies that full and hybrid inference
produce valid, non-empty outputs and that the optimized fixed decoder and
parallel tail load correctly. The smoke runs frames synchronously and is not a
throughput benchmark.

Run the source-paced demo after the smoke passes:

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan \
  --max-frames 60
```

The bundled video simulates live arrivals. Output is saved on the host as
`results/blackswan_live_<timestamp>.mp4`. By default, inference uses MIG,
same-frame detector/tracker overlap, and the fixed decoder, with full detection
on every consumed frame. **This command is a demo, not a benchmark.**

`--max-frames` counts source frames, not output masks: stale waiting frames are
replaced by newer arrivals. The output video contains emitted frames only and
therefore plays faster than wall clock when frames are dropped. There is no
next-frame GPU preprocessing or backbone lookahead in the live path.

For camera / ROS input, use the [integration guide](examples/README.md);
`demo_live.py` itself accepts a video file. See [Usage](#usage)
for optional hybrid detection and the reference tools.

> **Occupancy mapping:** preserve sensor exposure time separately from host
> arrival time and reject stale results. Tracker-only output may add positive
> occupied evidence, but missing masks must not clear free space. Use
> `negative_evidence_valid` together with timestamp and result-age checks.

---

## Usage

The default live command is in [Quick start](#quick-start). Pass multiple
prompts with `--text swan water`, or explicitly opt into hybrid detection:

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan \
  --redetect-interval-ms 1000
```

Hybrid uses periodic full-detection keyframes and native tracking between them.
It changes detection frequency, not the latest-frame scheduling policy.
Full detection on every consumed frame remains the default.

| Entry point | Purpose | Details |
|---|---|---|
| `demo_live.py` / `SAM3Live` | Freshness-first streaming, one or more prompts | [Live usage](docs/usage.md#live-video) |
| `tools/text_baseline.py` | Offline text-prompt reference and regression | [Offline usage](docs/usage.md#offline-text-reference) |
| `demo_box.py` | Specialized single-object tracking; separate artifacts required | [Box reference](docs/usage.md#box-prompt-reference) |

See the [usage guide](docs/usage.md) for parameters, output files, diagnostic
flags, and additional visual examples. Camera / ROS integrations should follow
the [ownership, timestamps, and lifecycle rules](examples/README.md).

## Performance

Recorded live reference runs on **Ryzen AI Max+ 395 / gfx1151**, **504px**,
`blackswan.mp4`, prompt `swan`, **one object**, with ROCm 7.14 / MIGraphX 2.17 /
ORT 1.24.2. The source was paced at 24 FPS for 250 arrivals; not all arrivals
were processed. Measurements exclude overlay and video encoding.

| Detection policy | Output rate | Arrival-to-result age p95 | Emitted / captured |
|---|---:|---:|---:|
| **Full detection on every consumed frame (default)** | **9.13 Hz** | **152.73 ms** | 96 / 250 |
| Hybrid, 1000 ms detection interval (opt-in) | 10.47 Hz | 135.63–137.76 ms | 110 / 250 per run |

These are September 1–2, 2026 reference measurements, **not a new release
benchmark or a paired full-versus-hybrid speedup claim**. The full-mode
statistics exclude the first five outputs; the hybrid statistics include all
outputs after explicit prewarm. Age starts at host arrival, not sensor exposure,
and these rates are not input FPS or a real-time deadline guarantee.

[Performance details](docs/performance.md) record the measurement windows,
multi-object scaling, fixed-decoder A/B, and correctness checks. Offline
throughput and historical box-only results are listed separately.
Use the [evaluation guide](docs/evaluation.md) for checks and measurement scope.

## Documentation

- [Usage and visual examples](docs/usage.md)
- [Container, dependencies, and artifact mounts](docker/rocm714/README.md)
- [Camera / ROS 2 integration](examples/README.md)
- [Performance records and correctness evidence](docs/performance.md)
- [Evaluation and regression commands](docs/evaluation.md)
- [Historical host setup, box benchmarks, and optimization notes](docs/historical/legacy-runtime.md)
- [Release notes](docs/releases/0.2.0-rc4.md)

## Known limitations

- **Validated target:** gfx1151 at 504px. Other GPUs are untested; 1008px is an
  advanced research path, not supported by the current fixed decoder.
- **Cold start:** ONNX/MXR build and first-use ORT compilation take time.
  Build once, retain caches, and measure startup separately from steady state.
- **Object scaling:** more active objects increase tracker cost. Live defaults
  to a five-object cap per prompt; this is not a performance guarantee.
- **Live output drops frames by design.** Use the offline tool when every
  source frame must be processed. The demo accepts files; camera / ROS transport
  and occupancy-grid publication require application integration.
- **Mapping safety:** maintain exposure-time pose alignment and an age budget;
  tracker-only missing masks are not evidence of free space.
- **Long streams:** keep a bounded session-reset policy. Reset prompts or
  tracking only after stopping the active pipeline; see the integration guide.

---

## Acknowledgements

- **SAM3**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3) — model weights
  and architecture. Weights must be downloaded separately from
  [facebook/sam3](https://huggingface.co/facebook/sam3) on HuggingFace.
- **DART**: the `sam3_tracker_video` model class originates from the
  [DART](https://arxiv.org/abs/2603.11441) project's transformers fork, since merged
  into HuggingFace Transformers (≥ 5.7.0).

---

## License

Unless otherwise noted, project-authored source code and documentation in this
repository are licensed under the Apache License, Version 2.0; see
[LICENSE](LICENSE). Required upstream attribution is recorded in
[NOTICE](NOTICE).

The SAM model metadata and tokenizer files under `model/sam3/` are SAM
Materials and remain subject to the separate
[SAM License](model/sam3/LICENSE). Model checkpoints are not included and must
be obtained separately under their applicable license.

Media and generated or evaluation artifacts under `assets/`, `docs/images/`,
and `results/` are not licensed under Apache-2.0 unless a file is explicitly
marked otherwise; they retain their respective source copyrights and terms.
External dependencies, downloaded binaries, datasets, model weights, and
locally generated ONNX/MXR artifacts retain their own licenses and are not
relicensed by this repository.
