# ROCm 7.14 / MIGraphX 2.17 container

This directory builds the tested gfx1151 stack entirely from source or pinned
public wheels without modifying the host ROCm installation.

## Pinned stack

- Ubuntu 24.04
- ROCm 7.14, gfx1151 APT packages
- MIGraphX commit `9f1a138e77f4738d82a065d225836b3b337950ce`
- ONNX Runtime v1.24.2 commit `058787ceead760166e3c50a0a4cba8a833a6f53f`
- PyTorch `2.11.0+rocm7.13.0` gfx1151 wheel, using the system ROCm 7.14 ABI
- torchvision `0.26.0+rocm7.13.0`
- Triton `3.6.0+rocm7.13.0`

AMD's current ROCm 7.14 multi-arch PyTorch wheels contain gfx942 kernels but
not gfx1151 kernels. The gfx1151 wheel is built against ROCm 7.13. The small
`rocm_sdk_system.py` compatibility module prevents that wheel from preloading
Python-packaged ROCm 7.13 libraries; dynamic libraries are resolved from the
container's `/opt/rocm` 7.14 installation instead. This exact combination is
covered by the smoke test and the SAM3 regression described below.

## Requirements

- Linux x86-64 host with Docker and BuildKit
- AMD gfx1151 GPU exposed as `/dev/kfd` and `/dev/dri`
- Approximately 45 GB free disk during a clean build
- Network access to GitHub and `repo.amd.com`

## Build from zero

```bash
./docker/rocm714/build.sh
```

The first build compiles rocMLIR, MIGraphX and ONNX Runtime and can take tens
of minutes. Intermediate source and build files default to:

```text
~/.cache/sam3-rocm714-build/
```

Override this with `SAM3_DOCKER_BUILD_ROOT`. Other useful overrides are
`JOBS`, `GPU_ARCH`, `BUILDER_IMAGE`, and `RUNTIME_IMAGE`.

The final image defaults to:

```text
sam3-gpu714-ort1242-mgx217-gfx1151:torch211
```

## Smoke test

`build.sh` runs this automatically when a GPU is present:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --ipc=host \
  sam3-gpu714-ort1242-mgx217-gfx1151:torch211 \
  python /opt/sam3-tools/smoke_test.py
```

The test executes a Torch GPU kernel and passes Torch GPU allocations directly
through ONNX Runtime's MIGraphX execution provider.

## Run SAM3

Set model and artifact directories if they live outside the checkout:

```bash
export SAM3_MODEL_DIR=/path/to/model/sam3
export SAM3_ONNX_DIR=/path/to/onnx_files_504
```

Open a shell:

```bash
./docker/rocm714/run.sh
```

For a fresh checkout, create the 504 px artifacts inside the container:

```bash
mkdir -p onnx_files_504
./docker/rocm714/run.sh python export/build.py \
  --pipeline text \
  --imgsz 504 \
  --checkpoint /models/sam3
```

Run the text pipeline:

```bash
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 \
  --text swan \
  --imgsz 504 \
  --mig \
  --onnx-dir /models/onnx_files_504 \
  --max-frames 31 \
  --output /workspace/demo_out/text/blackswan_rocm714.mp4
```

MIGraphX `.mxr` files and ORT caches are ABI-specific. Do not reuse artifacts
built by the legacy ROCm 7.2/MIGraphX stack in this image. Build them inside
this image or download artifacts published for the exact stack.

## Validated result

On Ryzen AI Max+ 395 / gfx1151, 504 px, `blackswan.mp4`, prompt `swan`:

- profile mean: 111.65 ms/frame
- propagation: 8.51 FPS
- two consecutive 30-frame regressions: mean IoU 0.9941, min IoU 0.9893

The repository's native setup remains the compatibility path. This container
is the reproducible ROCm 7.14 optimization path.
