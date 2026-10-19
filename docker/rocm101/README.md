# Docker runtime setup and troubleshooting

This guide covers the ROCm 10.1 / MIGraphX 2.18 / ONNX Runtime 1.29 container, its directory mounts,
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

This invokes `docker/rocm101/build.sh`. It verifies the downloaded AMD MIGraphX,
ONNX Runtime plugin EP, and Torch wheels, installs the pinned ROCm and Torch packages, and builds
a local Docker image. Host ROCm and Conda environments are not used by this
container workflow.

The default build finishes by running `smoke_test.py`: GPU availability, the supported
MIGraphX / ONNX Runtime / plugin-EP versions, SDPA, and an ONNX Runtime
session that must be fully assigned to the MIGraphX plugin EP. On success it prints:

```text
Binary runtime ready: sam3-gpu101-ort129-mgx218-gfx1151:0.4.0-rc1-local
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

For an existing compatible build, select its artifact root and proceed to
[Verify and run](#verify-and-run):

```bash
export SAM3_ONNX_DIR="/absolute/path/to/existing-onnx-mxr-directory"
```

Replace the placeholder with the directory containing the runtime module
subdirectories, not a single ONNX file. Keep `SAM3_MODEL_DIR` set to the
compatible checkpoint. Selecting existing artifacts does not require
`setup.sh --models` or `SAM3_MODEL_BUILD_ROOT`.

To resume an interrupted model build, reuse its original
`SAM3_MODEL_BUILD_ROOT`. The model command exports ONNX and compiles the 504px artifacts;
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
| `SAM3_ONNX_DIR` | Host artifact directory, mounted at `/models/onnx_files_504`; the container variable is set to that mount path |
| `SAM3_OUTPUT_DIR` (optional) | Existing writable directory, mounted at `/output` |

Keep `SAM3_MODEL_DIR` and `SAM3_ONNX_DIR` exported when running commands.
Re-export the same absolute paths in a new terminal. If the Git revision has
changed since the build, use the existing build directory explicitly rather
than recomputing its name from the new revision.

The scripts do not persist these variables into your shell. Save your exports
in a Git-ignored `.env.local` file and run `source .env.local` in each new
terminal if you want to reuse them. The file is not loaded automatically.
Use explicit exports for installations; development-machine defaults are
not part of the installation interface.

`setup.sh --models` selects its output using `SAM3_MODEL_BUILD_ROOT` and sets
`SAM3_ONNX_DIR` for the build subprocess only. Changing `SAM3_ONNX_DIR` alone
does not change that setup build destination.

### Select artifacts for an experiment

To use another existing build for one command, set the host directory when
launching the wrapper:

```bash
SAM3_ONNX_DIR="/absolute/path/to/experiment-artifacts" \
./docker/rocm101/run.sh python eval/benchmarks/bench_pipeline.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 --imgsz 504
```

This benchmark needs standalone tracker artifacts, including `backbone_tracker/`
and `tracker_modules/`. Full text inference uses the detector artifact set.
Choose the directory and resolution that match the script.

The example spells out the container path for clarity. Normally omit
`--onnx-dir` and let the script use `SAM3_ONNX_DIR` set by the wrapper. Pass
the option only when you intend to override the selected directory and have
confirmed that the replacement is visible inside the container.

Inside an interactive container, `export SAM3_ONNX_DIR=/workspace/results/my-experiment`
selects another directory already visible through a mount. Changing the variable
does not mount a new host directory. An explicit `--onnx-dir` overrides the
variable. For export or compilation experiments, choose a new writable output
directory rather than an immutable baseline.

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
./docker/rocm101/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --max-frames 50
```

Output is saved under `results/` in the host checkout. To open an interactive
shell with the same mounts and runtime:

```bash
./docker/rocm101/run.sh
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
| `Set SAM3_ONNX_DIR...` | Export the existing artifact root containing the runtime module subdirectories. |
| Image not found | Run `./setup.sh --runtime`; for a custom image, match the build and run settings below. |
| Fixed DETR decoder artifact not found | Default 504px MIG requires `detr_decoder_fixed/direct_gpuio.mxr`. Check the model build and `SAM3_ONNX_DIR`; use `--no-fixed-detr-decoder` only for diagnosis. |
| Input video cannot be opened | Use a path inside a configured mount; see the own-video example above. |
| `SHA256 mismatch` | Inspect the file named in the error. For the managed download cache, retry with a new `SAM3_BINARY_CACHE`; local override files must match the pinned release checksums. |

During runtime assembly, pip can report that Torch requires
`rocm[libraries]~=10.1.0` and `rocm-bootstrap`. These specific resolver
warnings are expected: the Torch wheels declare Python ROCm dependencies,
while the image supplies system ROCm 10.1 libraries. The compatibility module described below selects
those libraries. The default build's GPU/provider check must still pass.

MIGraphX MXR files and ORT caches depend on the compiler/runtime version.
After changing runtime versions, rebuild into a new artifact directory.
Artifacts built with the previous ROCm 7.14 / MIGraphX 2.17 runtime cannot be
loaded and must be rebuilt. The installation smoke reports missing
optimized artifacts; check the model-build result and selected artifact path.

## Advanced configuration

### Download cache and local runtime binaries

The default binary cache is
`~/.cache/sam3-runtime-binaries/0.4.0-rc1/`. Set `SAM3_BINARY_CACHE` to use a
different directory. `--no-cache` rebuilds Docker layers; it does not clear
this download cache or Docker's pip cache mount.

Every wheel is verified against the SHA256 pinned in [build.sh](build.sh) before
the Docker build starts, including files already present in the cache.

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
./docker/rocm101/run.sh python tools/smoke_live_release.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan --frames 12 --mode both \
  --output /output/installation-smoke.json
```

The report appears at `results/strict-validation/installation-smoke.json` on
the host. Strict mode is for validation, not model export or cache generation;
if a required cache is missing, populate it in writable mode before retrying.

The wrapper sets `SAM3_ONNX_DIR=/models/onnx_files_504` inside the container.
Current inference, export, benchmark, and dataset evaluation entry points with
`--onnx-dir` use it as their artifact-directory default. An explicit
`--onnx-dir` overrides it. Current entry points support only 504px artifacts.
The two complete build entry points use 504px without an `--imgsz` option;
lower-level export and evaluation tools still accept explicit `--imgsz 504`.

## Runtime versions

| Component | Pinned version |
|---|---|
| Container base | Ubuntu 24.04 |
| ROCm packages | 10.1, Debian package release `10.1.0-3`, gfx1151 |
| MIGraphX | `2.18.0+rocm10.1.0` (official `migraphx` + `migraphx-libs` wheels) |
| ONNX Runtime | `1.29.0`, with the official `onnxruntime-ep-migraphx` `1.0.0+rocm10.1.0` plugin EP |
| PyTorch | `2.14.0+rocm10.1.0`, with `amd-torch-device-gfx1151` and `amd-torch-device-gfx115x` |
| torchvision | `0.29.0a0+rocm10.1.0` |
| Triton | `3.8.0+git669b31ac.rocm10.1.0` |

MIGraphX, the plugin EP, and Torch come from AMD's package indexes
(`stable.repo.amd.com/rocm/{migraphx,onnxruntime,pytorch}/whl-next`); ONNX
Runtime comes from PyPI. Nothing is compiled during the image build.

The plugin EP is registered from a library and attached with
`SessionOptions.add_provider_for_devices`; `tracker/ort_plugin_ep.py` is the
single session factory and raises if any graph node is not assigned to
MIGraphX. `amd-torch-device-gfx115x` is required in addition to the gfx1151
wheel: without it the default SDPA fails with `hipErrorInvalidImage`.

The Torch wheels declare Python-packaged ROCm dependencies. The included
`rocm_sdk_system.py` compatibility module prevents them from preloading a
second ROCm stack; dynamic libraries resolve from the container's system
ROCm 10.1 installation instead. This combination is covered by the runtime
check and model regression records.

For a complete rebuild from downloaded runtime binaries through local model
compilation, use the [release validation runner](../../docs/evaluation.md#clean-environment-release-validation).
Measurement windows, model identities, and correctness results are recorded in
the [performance guide](../../docs/performance.md).
