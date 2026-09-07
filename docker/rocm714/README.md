# ROCm 7.14 / MIGraphX 2.17 container

This is the single supported runtime for the optimized SAM3 GPU/live path.
The host ROCm 7.2/MIGraphX 2.16 environment may remain installed for unrelated
projects, but it is not a supported SAM3 performance or deployment target and
cannot load the default fixed-decoder MXR.

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

Without an override, `run.sh` uses the checkout symlink
`onnx_files_504_mgx217`, which points at the assembled 2.17 production root.
The legacy `onnx_files_504` directory is not the optimized default.

Open a shell:

```bash
./docker/rocm714/run.sh
```

For a fresh checkout, create the 504 px artifacts inside the container:

```bash
mkdir -p onnx_files_504
SAM3_ONNX_DIR="$PWD/onnx_files_504" \
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
  --parallel-tail \
  --pipeline-backbone \
  --onnx-dir /models/onnx_files_504 \
  --max-frames 31 \
  --output /workspace/demo_out/text/blackswan_rocm714.mp4
```

MIGraphX `.mxr` files and ORT caches are ABI-specific. Do not reuse artifacts
built by the legacy ROCm 7.2/MIGraphX stack in this image. Build them inside
this image or download artifacts published for the exact stack.

Run the default freshness-oriented live path:

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 \
  --text swan
```

For MIG 504px this automatically enables same-frame parallel detector/tracker
tails and loads `detr_decoder_fixed/direct_gpuio.mxr`. A missing or incompatible
fixed decoder is a deployment error for the optimized configuration; do not mix
2.16 and 2.17 MXR artifacts.

The default remains freshness-oriented full detection on every consumed frame.
An explicitly positive interval enables the optional unified hybrid:

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 \
  --text swan \
  --redetect-interval-ms 1000
```

This path uses one `SAM3Live` model and at most one active inference session.
Before every wall-clock keyframe it creates a fresh inner session, reuses the
encoded prompt tensors, and associates the fresh detections with stable public
IDs by same-prompt mask IoU. The same model then performs native detector-skip
tracking between keyframes. It does not instantiate the separate
`SAM3OnnxTracker` backbone and does not load its legacy MIGraphX 2.16 artifacts.
If tracker-only output loses an object, the request for full detection is
sticky until the next consumed frame. Inspect `lost_object_ids` and
`redetect_reason` in the result, and use `negative_evidence_valid` as the
clearing gate. `redetect_reason` is one of `first_frame`, `interval`,
`caller_override`, `inner_forced`, `object_loss`, `reset_prompts`,
`reset_tracking`, `detection_retry`, or `None`.

For occupancy mapping, tracker-only output (`detected=False`) is positive-only
evidence: present masks may mark occupied space, but missing masks must not
clear free space. Apply negative/free-space clearing only from a non-stale full
detection result (`detected=True`).

## Validated result

On Ryzen AI Max+ 395 / gfx1151, 504 px, `blackswan.mp4`, prompt `swan`:

- profile mean: 111.65 ms/frame
- default latest-frame + fixed decoder: 9.1275 Hz, age p95 152.73 ms
- clean-keyframe unified hybrid headless, one object, 250 arrivals at 24 FPS:
  10.4719/10.4724/10.4749 Hz across three runs, 110 outputs per run,
  approximately 95.45 ms mean service time, and 135.63-137.76 ms frame-age p95
- propagation: 8.51 FPS
- offline propagation with `--parallel-tail`: 8.93-9.09 FPS (median 9.03)
- propagation with `--parallel-tail --pipeline-backbone`: 10.21-10.27 FPS
- two consecutive 30-frame regressions: mean IoU 0.9941, min IoU 0.9893

The multi-object unified-hybrid matrix uses
`two_person_dog_lawn.mp4`, 300 arrivals at 25 FPS, and no overlay/video
encoding:

| Prompts | Representative objects/output | Output rate | Service mean | Age p50/p95 |
|---|---:|---:|---:|---:|
| `people` | 2 | 9.4621 Hz | 105.60 ms | 126.91/150.74 ms |
| `people,dog` | 3 | 8.3682 Hz | 119.44 ms | 139.49/172.52 ms |
| `people,dog,lawn,sidewalk` | 6–7 | 6.2287 Hz | 160.47 ms | 178.26/237.65 ms |

Prompt count alone is not the scaling variable; retained object count drives
much of the tracker cost. All three matrix runs reported zero propagation-frame
object-loss events.

A separate 50-frame `office_hallway_two_way` check compared clean hybrid
propagation with full SAM3 on every frame. Floor union IoU mean/min was
0.981233/0.949965 and wall was 0.963659/0.933420; false-free rates were 1.4169%
and 1.7963%, respectively. A tracker loss on frame 47 triggered sticky
recovery on the next frame. These figures are workload-specific bounds.
Tracker-only output remains positive-only evidence regardless of this result.

A 120-keyframe no-GC fresh-session soak showed no sustained growth: Torch
allocated memory changed by about +508 KB, reserved memory by +4 MiB, and
process RSS by +72 KiB, with all metrics flat after iteration 10. See
`/home/amd/project/sam3-artifacts/gpu/experiments/unified-reset-soak/REPORT.md`.

The host setup is retained for source diagnostics and unrelated projects, not
as a second supported SAM3 deployment path. This container is the reproducible
ROCm 7.14 optimization and production path.
