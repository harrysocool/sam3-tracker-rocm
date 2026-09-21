# SAM3 Video Tracking on AMD GPUs

Text-prompted video segmentation and tracking on AMD ROCm, built on
[Meta's SAM3](https://github.com/facebookresearch/sam3) and accelerated with
MIGraphX. Describe a target, such as `"swan"` or `"person on a bike"`, to detect
and track its masks through video.

The **streaming API prioritizes fresh observations**: it processes the newest
available frame and runs full text detection on every consumed frame by
default. It includes a video demo, a [ROS 2 integration skeleton](examples/README.md),
and an offline text-prompt inference tool.

| `"swan"` | `"camel"` | `"pig"` (3 objects) |
|:---:|:---:|:---:|
| <img src="docs/images/demo_swan_text_mig.gif" width="260" alt="swan text-prompt segmentation"> | <img src="docs/images/demo_camel_text_mig.gif" width="260" alt="camel text-prompt segmentation"> | <img src="docs/images/demo_pigs_multi_object.gif" width="260" alt="three pigs tracked with a text prompt"> |

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
| Network | Access to Docker Hub, Ubuntu package repositories, GitHub releases, AMD repositories, and PyPI for runtime dependencies |
| SAM3 weights | A compatible local `model.safetensors`, obtained separately under the [SAM License](model/sam3/LICENSE) |
| Disk | At least 30 GiB free; 40 GiB is recommended for clean rebuilds and validation |
| Runtime | ROCm **7.14**, MIGraphX **2.17**, ONNX Runtime **1.24.2**, installed inside the container |

Use `docker/rocm714/run.sh` for the optimized GPU path. No host conda environment
is needed. The host MIGraphX 2.16 stack is not supported for deployment and
cannot load the current fixed-decoder artifact. See the
[container guide](docker/rocm714/README.md) for exact dependency versions.

## Quick start

These steps use the current supported **`main` branch** and build the
recommended **504px text / live pipeline**. Run the commands in Bash, in order,
from the same shell. An existing checkout of `main` or a compatible development
branch can skip the clone.

### 1. Get the source and configure local weights

```bash
git clone --branch main --single-branch \
  https://github.com/harrysocool/sam3-tracker-rocm.git
cd sam3-tracker-rocm
export SAM3_MODEL_DIR="$PWD/model/sam3"
```

The checkout includes model configuration and tokenizer files. Supply the
local weights described in [Requirements](#requirements) alongside those files.

If `model.safetensors` is not already present in `SAM3_MODEL_DIR`, link your
existing file using its absolute path:

```bash
ln -s /absolute/path/to/model.safetensors \
  "$SAM3_MODEL_DIR/model.safetensors"
```

Replace the source path with your actual file. You can also copy it into place.
Keep the configuration and tokenizer files supplied by this checkout; the
runtime expects the weight file to be named `model.safetensors`. If you already
have a complete compatible model directory, set `SAM3_MODEL_DIR` to its absolute
path instead.

The container wrapper supports an absolute weight link through intermediate
links, including layouts where the final weight file has a different basename.
A broken absolute weight link fails before the container starts.

To compare your weights with the published validation checkpoint, see the
optional [checkpoint identity check](docs/evaluation.md#checkpoint-identity).

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
export SAM3_MODEL_BUILD_ROOT="$HOME/sam3-artifacts/gpu/build-$(git rev-parse --short HEAD)"
export SAM3_ONNX_DIR="$SAM3_MODEL_BUILD_ROOT/onnx_files_504"
./setup.sh --models "$SAM3_MODEL_DIR"
```

This exports ONNX and builds the 504px model artifacts inside the container,
including the GPU-I/O backbone, fixed DETR decoder, and independently
autotuned S1--S10 memory-attention caches. The validated memory policy uses
attention-specific MLIR tuning for S1--S7/S9--S10 and generic tuning for S8.
The initial build is a one-time compilation step; completed export / compile
steps are skipped on rerun when their recorded memory-cache policy still
matches. A three-frame writable prewarm populates the DETR runtime cache before
the final manifest is written, so later smoke and benchmark runs do not add
unrecorded artifact files.
Each root also receives `BUILD_PROVENANCE.json` before the first export. A
non-empty root is resumable only when its source, checkpoint, image, EC mode,
and shape parameters match. Otherwise the build fails without deleting files;
only an explicit full `--force` build may reset generated artifacts.

`setup.sh --models` is a performance build and requires the current EC power
mode to be `performance`. On the EVO-X2 this is read from
`/sys/class/ec_su_axb35/apu/power_mode`. On another gfx1151 platform without
that interface, verify the equivalent BIOS setting first and pass the explicit
attestation below:

```bash
SAM3_EC_POWER_MODE=performance ./setup.sh --models "$SAM3_MODEL_DIR"
```

#### Optional EC power-mode verification

The `/sys/class/ec_su_axb35/` interface is not provided by a stock Linux
installation. It appears only when the optional third-party Sixunited
AXB35-02 EC driver is installed and loaded. The driver is not a SAM3 runtime
dependency and `setup.sh` never installs a kernel module automatically.

The default path is to select **Performance** in the BIOS and use the explicit
`SAM3_EC_POWER_MODE=performance` attestation above. EVO-X2 users who want Linux
to verify the setting automatically can review and install the driver from:

```text
https://github.com/cmetz/ec-su_axb35-linux
```

The validated upstream revision used during this work was:

```text
f62c2c228959a08683273a26ef3afd8991e69f6d
```

Follow that project's build/install instructions, then verify:

```bash
cat /sys/class/ec_su_axb35/apu/power_mode
```

It must print `performance` before `setup.sh --models` is run. This is an
out-of-tree driver with root-level EC write access; kernel headers and possibly
Secure Boot module signing are required. Install it only on a supported board
and review its source first.

The build always removes `MIGRAPHX_SKIP_BENCHMARKING` and compiles each memory
shape in a separate process. Do not use the attestation to bypass an unknown or
balanced power policy: autotuning is hardware-measured and the selected MXR
kernels can change with the available power budget.

Successful completion also writes `ARTIFACT_MANIFEST.json`, its checksum
sidecar, and `SHA256SUMS` into `onnx_files_504`. The manifest records the source
revision, checkpoint hash, container image identity, runtime versions, EC mode,
GPU name/architecture, optional `SAM3_BUILD_HOST_ID`, memory compile policy,
and every generated artifact hash. It does not record hostname or hardware
serial numbers.

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
  --max-frames 50
```

The bundled video contains 50 frames at 24 FPS and simulates live arrivals.
Output is saved on the host as `results/blackswan_live_<timestamp>.mp4`.
By default, inference uses MIG, same-frame detector/tracker overlap, and the
fixed decoder, with full detection on every consumed frame.
**This command is a demo, not a benchmark.**

`--max-frames` counts source frames, not output masks: stale waiting frames are
replaced by newer arrivals. The output video contains emitted frames only and
therefore plays faster than wall clock when frames are dropped. There is no
next-frame GPU preprocessing or backbone lookahead in the live path.

For camera / ROS input, use the [integration guide](examples/README.md);
`demo_live.py` itself accepts a video file. See [Usage](#usage)
for offline inference and optional hybrid detection.

---

## Usage

Live and offline inference share the optimized **504px FP16 MIG backend** by
default. Choose the entry point by how input frames should be processed:

| Entry point | Frame handling | Details |
|---|---|---|
| `demo_live.py` | Process the newest available frame; stale waiting frames may be dropped | [Live usage](docs/usage.md#live-video) |
| `tools/text_baseline.py` | Process every selected video frame in order, or a single image | [Offline usage](docs/usage.md#offline-text-inference) |

### Live video

Use the live command in [Quick start](#quick-start) for freshness-oriented
processing. Pass multiple prompts with `--text swan water`. For camera / ROS
input, use `SAM3Live` with the latest-frame pipeline described in the
[integration guide](examples/README.md).

### Offline images and videos

For a recorded video where every selected frame should be processed:

```bash
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --max-frames 50
```

Output is saved as `demo_out/text/blackswan_text.mp4` in the host checkout.
Offline video inference uses backbone prefetch for throughput; live does not.
See [offline usage](docs/usage.md#offline-text-inference) for single images,
multiple prompts, output controls, and the explicit PyTorch reference mode.

### Optional hybrid detection

To enable optional hybrid detection:

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --max-frames 50 \
  --redetect-interval-ms 1000
```

Hybrid uses periodic full-detection keyframes and native tracking between them.
It changes detection frequency, not the latest-frame scheduling policy.
Full detection on every consumed frame remains the default.

### Historical box-prompt reference

The legacy `demo_box.py` / `SAM3OnnxTracker` path is retained for historical
box-prompt and DAVIS regression work. It requires a separate artifact set that
`setup.sh --models` does not build, and it is outside the supported release
smoke. See the [historical box reference](docs/usage.md#historical-box-prompt-reference).

See the [usage guide](docs/usage.md) for parameters, output files, diagnostic
flags, and additional visual examples.

## Performance

Recorded live reference runs on **Ryzen AI Max+ 395 / gfx1151**, **504px**,
`blackswan.mp4`, prompt `swan`, **one object**, with ROCm 7.14 / MIGraphX 2.17 /
ORT 1.24.2. The source was paced at 24 FPS for 250 arrivals; not all arrivals
were processed. Measurements exclude overlay and video encoding.

Both modes ran with the fixed 504px DETR decoder and same-frame
detector/tracker parallel tail enabled. These are the optimized live defaults
with the complete MIG artifacts built by Quick start.

| Detection policy | Mean service time | Output rate | Emitted / captured |
|---|---:|---:|---:|
| **Full detection on every consumed frame (default)** | **109.53 ms** | **9.13 Hz** | 96 / 250 |
| Hybrid, 1000 ms detection interval (opt-in) | 95.43–95.48 ms | 10.47 Hz | 110 / 250 per run |

Service time measures processing of a selected frame, including preprocessing,
model inference, and output postprocessing. Output rate counts completed results.

These are September 1–2, 2026 reference measurements, **not a new release
benchmark or a paired full-versus-hybrid speedup claim**. The full-mode
statistics exclude the first five outputs; the hybrid statistics include all
outputs after explicit prewarm. These rates are not input FPS or a real-time
deadline guarantee.

[Performance details](docs/performance.md) record the measurement windows,
multi-object scaling, fixed-decoder A/B, and correctness checks. Offline
throughput and historical results are listed separately.
Use the [evaluation guide](docs/evaluation.md) for checks and measurement scope.

## Documentation

- [Usage and visual examples](docs/usage.md)
- [Container, dependencies, and artifact mounts](docker/rocm714/README.md)
- [Camera / ROS 2 integration](examples/README.md)
- [Performance records and correctness evidence](docs/performance.md)
- [Evaluation and regression commands](docs/evaluation.md)
- [Historical runtime and optimization notes](docs/historical/legacy-runtime.md)
- [Release notes](docs/releases/)

## Known limitations

- **Validated target:** gfx1151 at 504px. Other GPUs are untested; 1008px is an
  advanced research path, not supported by the current fixed decoder.
- **Cold start:** ONNX/MXR build and first-use ORT compilation take time.
  Build once, retain caches, and measure startup separately from steady state.
- **Object scaling:** more active objects increase tracker cost. Live and
  offline default to a five-object cap per prompt; this is not a performance guarantee.
- **Live output drops frames by design.** Use the offline tool when every
  source frame must be processed. The demo accepts files; camera / ROS input
  requires application integration.
- **Long streams:** keep a bounded session-reset policy. Reset prompts or
  tracking only after stopping the active pipeline; see the integration guide.

---

## Acknowledgements

- **SAM3**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3) —
  upstream model architecture and materials, subject to the separate SAM License.
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
