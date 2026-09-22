# Docker runtime setup and troubleshooting

This guide covers the ROCm 7.14 / MIGraphX 2.17 container, its directory mounts,
and common setup issues. For a first installation, follow the root
[Quick start](../../README.md#quick-start) through local weight configuration,
runtime assembly, model building, and the installation smoke.

Run all commands below in Bash from the repository root, the directory that
contains `setup.sh`. The image contains Ubuntu 24.04 and the Python/GPU runtime;
the host needs Docker and a working AMDGPU driver.

## Before you start

- Validated hardware: AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`).
- Linux x86-64 with the GPU exposed at `/dev/kfd` and `/dev/dri`.
- Docker with BuildKit/buildx support and permission to run Docker commands.
- Bash, `curl`, and the standard Linux `readlink`, `sha256sum`, and `tar` tools.
- Network access to Docker Hub, Ubuntu package repositories, GitHub releases,
  AMD repositories, and PyPI.
- At least 30 GiB free, with 40 GiB recommended for clean rebuilds. Allow space
  on both the Docker storage filesystem and the model/artifact filesystem.

Useful host checks:

```bash
docker info
docker buildx version
ls -l /dev/kfd /dev/dri/renderD*
```

For model building and inference, supply a compatible local SAM3 checkpoint
obtained independently under the [SAM License](../../model/sam3/LICENSE).
See [local weight configuration](../../README.md#1-get-the-source-and-configure-local-weights)
for the model directory layout. The setup scripts do not download weights.

## Build the runtime

```bash
./setup.sh --runtime
```

This invokes `docker/rocm714/build.sh`. It verifies the downloaded MIGraphX and
ONNX Runtime binaries, installs the pinned ROCm and Torch packages, and builds
a local Docker image. Host ROCm and Conda environments are not used by this
container workflow.

The default build finishes by checking GPU availability, MIGraphX 2.17,
ONNX Runtime 1.24.2, and `MIGraphXExecutionProvider`. On success it prints:

```text
Binary runtime ready: sam3-gpu714-ort1242-mgx217-gfx1151:0.2.0-rc4-local
```

This confirms runtime assembly. The image does not contain the model weights
or compiled SAM3 artifacts; prepare those next. If Quick start has already
completed this stage, continue with its existing image and directories.

## Directories and mounts

The wrapper uses host environment variables to find the model and artifacts.
For a new build, with local weights configured as described in Quick start:

```bash
: "${SAM3_MODEL_DIR:?Set SAM3_MODEL_DIR as described in Quick start}"
export SAM3_MODEL_BUILD_ROOT="$HOME/sam3-artifacts/gpu/build-$(git rev-parse --short HEAD)"
export SAM3_ONNX_DIR="$SAM3_MODEL_BUILD_ROOT/onnx_files_504"
./setup.sh --models "$SAM3_MODEL_DIR"
```

Keep the model directory already selected in Quick start, including an
external directory if that is where your complete checkpoint is stored.
The command above checks that the variable is set; it does not replace it.

For an existing build, reuse its actual paths instead of choosing a new
directory. The model command exports ONNX and compiles the 504px artifacts;
successful completion reports `ALL OK`. It independently autotunes the S1--S10
memory-attention caches with full benchmarking enabled; S8 uses the validated
generic policy and the remaining shapes use attention-specific MLIR tuning.

The standard model command requires EC `performance` mode because build-time
kernel measurements affect the selected MXR programs. The EVO-X2 mode is read
from `/sys/class/ec_su_axb35/apu/power_mode`. If another gfx1151 platform has
no compatible sysfs interface, verify its BIOS performance setting and run:

```bash
SAM3_EC_POWER_MODE=performance ./setup.sh --models "$SAM3_MODEL_DIR"
```

The completed root contains `ARTIFACT_MANIFEST.json`,
`ARTIFACT_MANIFEST.sha256`, and `SHA256SUMS`. The clean-environment runner
refreshes this manifest after writable ORT prewarm and before strict smoke.

| Host location or setting | Container location / purpose |
|---|---|
| Repository checkout | `/workspace`, the container working directory |
| `SAM3_MODEL_DIR` | Complete checkpoint directory, mounted read-only at `/models/sam3` |
| `SAM3_MODEL_BUILD_ROOT` | Build output parent; its `onnx_files_504` child contains the artifacts |
| `SAM3_ONNX_DIR` | Artifact directory, mounted at `/models/onnx_files_504` |
| `SAM3_OUTPUT_DIR` (optional) | Existing writable directory, mounted at `/output` |

Keep `SAM3_MODEL_DIR` and `SAM3_ONNX_DIR` exported when running commands.
Re-export the same absolute paths in a new terminal. If the Git revision has
changed since the build, use the existing build directory explicitly rather
than recomputing its name from the new revision.

The scripts do not persist these variables into your shell. Without an
override, `run.sh` looks for checkout `model/sam3` and `onnx_files_504_mgx217`.
The latter is a development-machine symlink and is absent from a fresh clone.
Use the explicit exports above for customer installations.

Inputs must also be accessible through a container mount. See
[use your own video](../../docs/usage.md#use-your-own-video) for a complete
host-to-container path example.

## Verify and run

With the model and artifact directories configured, follow the
[installation smoke command](../../docs/evaluation.md#provider-and-installation-checks).
It exercises full and hybrid inference and checks that the fixed 504px decoder
and parallel detector/tracker tail are loaded. Expect:

```text
Installation smoke PASS: results/perf/installation-smoke.json
```

Then run the bundled video demo:

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --max-frames 50
```

Output is saved under `results/` in the host checkout. To open an interactive
shell with the same mounts and runtime:

```bash
./docker/rocm714/run.sh
```

See [usage](../../docs/usage.md) for other prompts, optional hybrid detection,
and offline inference commands. Camera and ROS applications should follow the
[integration guide](../../examples/README.md).

## Troubleshooting

| Symptom | Check or next step |
|---|---|
| Docker permission or connection error | Confirm `docker info` works for the account running setup. |
| Missing `/dev/kfd` or render device | Check the host AMDGPU driver and GPU device access before building the image. |
| `Set SAM3_MODEL_DIR...` | Export the absolute path to the complete local model directory. |
| `Set SAM3_ONNX_DIR...` | Export the artifact directory created by the model build, including its `onnx_files_504` child. |
| Image not found | Run `./setup.sh --runtime`; for a custom image, match the build and run settings below. |
| Fixed DETR decoder artifact not found | Default 504px MIG requires `detr_decoder_fixed/direct_gpuio.mxr`. Check the model build and `SAM3_ONNX_DIR`; use `--no-fixed-detr-decoder` only for diagnosis. |
| Input video cannot be opened | Use a path inside a configured mount; see the own-video example above. |
| `SHA256 mismatch` | Inspect the file named in the error. For the managed download cache, retry with a new `SAM3_BINARY_CACHE`; local override files must match the pinned release checksums. |

During runtime assembly, pip can report that Torch requires
`rocm[libraries]==7.13.0`. This specific resolver warning is expected: the
gfx1151 wheel declares a Python ROCm dependency, while the image supplies
system ROCm 7.14 libraries. The compatibility module described below selects
those libraries. The default build's GPU/provider check must still pass.

MIGraphX MXR files and ORT caches depend on the compiler/runtime version.
After changing runtime versions, rebuild into a new artifact directory.
Host MIGraphX 2.16 is not supported for this deployment path and cannot load
the current fixed-decoder artifact. The installation smoke reports missing
optimized artifacts; check the model-build result and selected artifact path.

## Advanced configuration

### Download cache and local runtime binaries

The default binary cache is
`~/.cache/sam3-runtime-binaries/0.2.0-rc4/`. Set `SAM3_BINARY_CACHE` to use a
different directory. `--no-cache` rebuilds Docker layers; it does not clear
this download cache or Docker's pip cache mount.

To supply already downloaded runtime binaries, use their local paths:

```bash
MIGRAPHX_ARCHIVE=/path/to/migraphx.tar.gz \
ORT_WHEEL_PATH=/path/to/onnxruntime_migraphx.whl \
./setup.sh --runtime
```

These overrides retain checksum verification. The default URLs and SHA256
values are pinned in [build.sh](build.sh).

### Custom image name

The build script reads `RUNTIME_IMAGE`; the run wrapper reads
`SAM3_DOCKER_IMAGE`. Set both to the same tag:

```bash
export SAM3_DOCKER_IMAGE="sam3-local:custom"
RUNTIME_IMAGE="$SAM3_DOCKER_IMAGE" ./setup.sh --runtime
```

### Read-only validation

`SAM3_DOCKER_STRICT=1` disables container networking and makes the checkout,
artifacts, and container root filesystem read-only. Prewarm ORT caches in the
default writable mode first (`SAM3_DOCKER_STRICT=0`), using the installation
smoke above. Then repeat that check with read-only artifacts and an explicit
writable output mount:

```bash
mkdir -p "$PWD/results/strict-validation"
SAM3_DOCKER_STRICT=1 \
SAM3_OUTPUT_DIR="$PWD/results/strict-validation" \
./docker/rocm714/run.sh python tools/smoke_live_release.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan --frames 12 --mode both \
  --output /output/installation-smoke.json
```

The report appears at `results/strict-validation/installation-smoke.json` on
the host. Strict mode is for validation, not model export or cache generation;
if a required cache is missing, populate it in writable mode before retrying.

The wrapper sets `SAM3_DEFAULT_ONNX_DIR=/models/onnx_files_504` for both
`demo_live.py` and `tools/text_baseline.py`. Other evaluation tools may require
explicit `--onnx-dir` and `--imgsz` arguments, as shown in their usage examples.

## Runtime versions

| Component | Pinned version |
|---|---|
| Container base | Ubuntu 24.04 |
| ROCm packages | 7.14, Debian package release `7.14.1-0`, gfx1151 |
| MIGraphX | 2.17, release commit `2e9924db6e6c0c9ff5dcbd61f59b97132598731a` |
| ONNX Runtime | 1.24.2, commit `058787ceead760166e3c50a0a4cba8a833a6f53f` |
| PyTorch | `2.11.0+rocm7.13.0`, gfx1151 wheel |
| torchvision | `0.26.0+rocm7.13.0` |
| Triton | `3.6.0+rocm7.13.0` |

The SAM3-specific MIGraphX archive is published in the
[`harrysocool/AMDMIGraphX` release](https://github.com/harrysocool/AMDMIGraphX/releases/tag/v2.17.0%2Bsam3-fc1sink.20260908.1).
It is based on upstream MIGraphX
`9f1a138e77f4738d82a065d225836b3b337950ce` and rocMLIR FC1-sink commit
`f3404d59b581fdf9d8cd7c1be9aeeb267851af93` (tag
`sam3-fc1-sink-rocm714-v2`). Its SHA256 is
`ed1458c632eb2f0e2cab3c457aee93e39196cbb77d2180e47525e0009563dac1`.

The validated gfx1151 Torch wheel was built against ROCm 7.13. The included
`rocm_sdk_system.py` compatibility module prevents it from preloading
Python-packaged ROCm 7.13 libraries; dynamic libraries resolve from the
container's system ROCm 7.14 installation instead. This combination is covered
by the runtime check and model regression records.

For a complete rebuild from downloaded runtime binaries through local model
compilation, use the [release validation runner](../../docs/evaluation.md#clean-environment-release-validation).
Measurement windows, model identities, and correctness results are recorded in
the [performance guide](../../docs/performance.md).
