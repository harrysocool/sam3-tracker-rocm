# ROCm 7.14 / MIGraphX 2.17 container

This is the single supported runtime for the optimized SAM3 GPU/live path.
The host ROCm 7.2/MIGraphX 2.16 environment may remain installed for unrelated
projects, but it is not a supported SAM3 performance or deployment target and
cannot load the default fixed-decoder MXR.

The supported path assembles the runtime from published binaries: AMD ROCm and
Torch packages, the SAM3-specific MIGraphX tar from the
[`harrysocool/AMDMIGraphX` release](https://github.com/harrysocool/AMDMIGraphX/releases/tag/v2.17.0%2Bsam3-fc1sink.20260908.1),
and the pinned ORT wheel. It does not compile rocMLIR, MIGraphX, ORT, or
PyTorch and does not modify host ROCm.

## Pinned stack

- Ubuntu 24.04
- ROCm 7.14, gfx1151 APT packages
- MIGraphX release commit `2e9924db6e6c0c9ff5dcbd61f59b97132598731a`,
  based on upstream `9f1a138e77f4738d82a065d225836b3b337950ce`
- rocMLIR FC1-sink commit `f3404d59b581fdf9d8cd7c1be9aeeb267851af93`,
  tag `sam3-fc1-sink-rocm714-v2`
- MIGraphX archive SHA256
  `ed1458c632eb2f0e2cab3c457aee93e39196cbb77d2180e47525e0009563dac1`
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
- Network access to the release assets and AMD package repositories

## Assemble from precompiled dependencies

```bash
./docker/rocm714/build.sh
```

`build.sh` downloads and verifies the pinned MIGraphX and ORT binaries, then
installs them into an Ubuntu/ROCm image. Override the release downloads while
testing local files with:

```bash
MIGRAPHX_ARCHIVE=/path/to/migraphx.tar.gz \
ORT_WHEEL_PATH=/path/to/onnxruntime_migraphx.whl \
./docker/rocm714/build.sh
```

The binary download cache defaults to `~/.cache/sam3-runtime-binaries/`.

The final image defaults to:

```text
sam3-gpu714-ort1242-mgx217-gfx1151:0.2.0-rc4-local
```

## Smoke test

`build.sh` checks GPU visibility and the imported runtime versions after image
assembly. Model generation and inference checks are separate steps: follow
[Quick start](../../README.md#quick-start), then the
[installation smoke and regression guide](../../docs/evaluation.md).

## Model artifacts and mounts

For a fresh checkout, use the Quick start's explicit directory exports before
building models. `setup.sh --models` does not create the development machine's
`onnx_files_504_mgx217` symlink or persist variables into your shell.

| Host setting | Container path / purpose |
|---|---|
| `SAM3_MODEL_DIR` | Complete checkpoint directory, mounted read-only at `/models/sam3` |
| `SAM3_MODEL_BUILD_ROOT` | Output parent used by `setup.sh --models`; its `onnx_files_504` child is the runtime artifact directory |
| `SAM3_ONNX_DIR` | Artifact directory mounted at `/models/onnx_files_504`; export the same child used for the model build |
| `SAM3_OUTPUT_DIR` (optional) | An existing writable directory, mounted at `/output` |

Without overrides, `run.sh` uses checkout `model/sam3` and
`onnx_files_504_mgx217`. The latter is a development deployment link, not an
artifact bundle included in a fresh clone. Legacy `onnx_files_504` is not the
optimized default.

The checkout is mounted at `/workspace`. The wrapper sets
`SAM3_DEFAULT_ONNX_DIR=/models/onnx_files_504` for consumers such as live;
offline tools may still require an explicit `--onnx-dir` and `--imgsz`.

MIGraphX `.mxr` files and ORT caches are ABI-specific. Build them inside this
image; never reuse host 2.16 artifacts or overwrite immutable baseline files.
The default writable mode allows first-use ORT compilation beside the graphs.
`SAM3_DOCKER_STRICT=1` disables networking and mounts the checkout/artifacts
read-only, so prewarm caches first and use a writable output mount as needed.

## Run SAM3

With `SAM3_MODEL_DIR` and `SAM3_ONNX_DIR` still exported:

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan
```

The default is latest-frame full detection, with same-frame parallel tails and
the available fixed 504px decoder. A missing or incompatible fixed decoder is
not an accepted optimized deployment configuration; do not silently switch to
the host runtime. Use `--no-fixed-detr-decoder` only for diagnosis.

To open an interactive shell in the same configured environment:

```bash
./docker/rocm714/run.sh
```

See [usage](../../docs/usage.md) for multiple prompts, optional hybrid detection,
offline reference commands, and their different defaults. For occupancy
mapping, preserve exposure timestamps, apply an age budget, and never clear
free space from tracker-only absence; see the
[integration guide](../../examples/README.md).

## Validated result

Measurements, dates, aggregation windows, model identities, and correctness
scope are maintained in one place: [performance records](../../docs/performance.md).
The default-live 9.13 Hz reference, optional-hybrid runs, and earlier offline
FPS values are different workloads, not interchangeable benchmarks.

Use [release validation](../../docs/evaluation.md#clean-environment-release-validation)
to test a new binary assembly and local model build. The host runtime remains
available only for unrelated projects/source diagnostics, not SAM3 deployment.
