# Evaluation and regression guide

These commands use the [supported ROCm 7.14 / MIGraphX 2.17 container](../docker/rocm714/README.md).
Complete [Quick start](../README.md#quick-start), keep `SAM3_MODEL_DIR` and
`SAM3_ONNX_DIR` exported, and run commands from the checkout root in Bash.
Keep outputs under ignored `results/perf/` or an external artifact directory.

## Choose the right check

| Check | What it establishes | What it does not establish |
|---|---|---|
| Provider / CLI check | Correct imported runtime and accepted arguments | Loaded model correctness or speed |
| Installation smoke | Full/hybrid execution, non-empty masks, default decoder and parallel-tail loading | Mask equivalence or source-paced throughput |
| PT-vs-MIG mask regression | Offline backbone/ORT mask agreement | Fixed-decoder equivalence, hybrid behavior, or map safety |
| Serial/parallel A/B | Offline output equivalence and schedule timing | Default live frame age |
| Source-paced integration check | Latest-frame ownership, drops, service, and age on that input | Original benchmark reproduction or robot-wide latency |
| DAVIS box regression | Tracker quality under a fixed dataset/prompt protocol | Text detection quality or current full-model throughput |

## Provider and installation checks

Before any GPU measurement:

```bash
./docker/rocm714/run.sh python -c \
  "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

Expect ORT **1.24.2** and `MIGraphXExecutionProvider`. A VitisAI-only result
is the NPU environment and must not be used for GPU measurements.

Run the headless installation smoke on the locally built artifacts:

```bash
./docker/rocm714/run.sh python tools/smoke_live_release.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan --frames 12 --mode both \
  --output results/perf/installation-smoke.json
```

The smoke submits frames synchronously, one at a time. It does not pace
arrivals, intentionally drop frames, render overlays, or encode video. Hybrid
detection uses explicit frame indices rather than a wall-clock timer.
Inspect the JSON's `passed` result and loaded-default records; do not report
its timing as live throughput.

The CPU-only release-flow guards can run with host Python and pytest; they
do not load a model or select a GPU/NPU provider:

```bash
python3 -m pytest -q tests/test_binary_release_flow.py
```

## PT-vs-MIG mask regression

```bash
./docker/rocm714/run.sh python eval/datasets/mask_diff_pt_vs_mig.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan --imgsz 504 --max-frames 30 \
  --parallel-tail \
  --out results/perf/mask_diff_504.json
```

Review per-frame IoU, minimum/mean IoU, missing masks, and scores. For the
offline lookahead configuration, add `--pipeline-backbone`; it requires
`--parallel-tail` and does not make this a live test.

This script patches the backbone, DETR encoder, and memory attention but
retains the native DETR decoder. Decoder changes additionally need a
native-versus-fixed full-path comparison and replay of the exact dropped
latest-frame subsequence. The [fixed-decoder evidence](performance.md#correctness-and-resource-checks)
records those checks; the generic PT-vs-MIG script alone does not cover them.

Hybrid changes also need clean-keyframe, object-loss/recovery, public-ID,
negative-evidence, and reset-lifecycle checks, not just a high mean IoU.

## Offline serial-versus-parallel A/B

```bash
./docker/rocm714/run.sh python eval/benchmarks/benchmark_parallel_tail.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan --imgsz 504 \
  --max-frames 50 --repeats 2 \
  --out results/perf/parallel_tail_504.json
```

The harness checks per-frame mask, score, object-ID, and prompt-ownership
equivalence. Add `--pipeline-backbone` only for an offline lookahead experiment.
Match frame windows and exclude or report pipeline fill/drain explicitly.
This harness does not benchmark live's fixed decoder or latest-frame queue.

## Source-paced live integration check

After Quick start, the ROS skeleton can run without ROS installed, using the
bundled file as a paced source and reporting the integration statistics:

```bash
mkdir -p results/perf
set -o pipefail
./docker/rocm714/run.sh python examples/ros_node_skeleton.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan \
  --policy always_full --max-frames 50 \
  --max-result-age-ms 200 \
  | tee results/perf/latest_frame_check.log
```

This includes the skeleton's 200 ms result-age gate. Report publication
rejections separately from inference failures and latest-slot drops.
`--max-frames` caps available source arrivals; it does not extend or loop a
short file.

The [published live reference](performance.md#default-full-detection-live-reference)
used a separate looped, headless 250-arrival harness and a defined warm window.
Neither this check nor a rendered `demo_live.py` run exactly reproduces it.
The original harness is retained in maintainer artifact storage; this checkout
does not provide a one-command reproduction of that historical run.

For new live results, record at least:

- code commit, dirty status, image identity, provider versions, model/MXR hashes;
- input identity, source FPS, resolution, prompts, active object counts;
- detection policy/interval, warmup and reset sequence, aggregation window;
- captured, emitted, dropped, failed, aborted, and age-rejected counts;
- completion cadence, service time, and frame age p50/p95/p99;
- rendering, encoding, transport, and downstream work included or excluded.

## Instrumented offline module profile

The profiler resolves `onnx_files_<imgsz>` relative to its working directory
and has no `--onnx-dir` option. Run it from `/models` so it uses the container's
explicit artifact mount, not the checkout's legacy artifact directory:

```bash
./docker/rocm714/run.sh sh -c '
  cd /models &&
  python /workspace/eval/benchmarks/profile_full_mig.py \
    --checkpoint /models/sam3 --video /workspace/assets/blackswan.mp4 \
    --text swan --imgsz 504 --max-frames 30 \
    --out /workspace/results/perf/profile_504_mig.json
'
```

This profiles 30 propagation frames after initialization, matching the
canonical offline workload shape. It retains the native decoder and uses
device synchronization hooks: it is not a measurement of default live,
parallel-tail overlap, or end-to-end camera latency. Do not sum asynchronous
stage timings to claim a live output rate.

## DAVIS tracker regression

Tracker changes must retain the DAVIS regression in addition to text-path
checks. Use DAVIS 2017 val with the same initial-box protocol, resolution,
and matching box-tracker artifacts. The text/live model build does not supply
the separate box artifact set.

The [historical evaluation instructions](historical/legacy-runtime.md#davis-and-box-evaluation)
preserve the dataset source, commands, and original runtime context. The saved
81.6% figure is mean J for a 504px box-prompt run, not J&F or a text-prompt
result. A new runtime/artifact combination needs its own regression.

## Clean-environment release validation

To test binary download, image assembly, empty-directory model generation,
ORT prewarming, and full/hybrid installation smoke together:

```bash
./tools/docker_test_runner.sh \
  --checkpoint "$SAM3_MODEL_DIR" \
  --output "$HOME/sam3-artifacts/gpu/clean-validation-0.2.0-rc4"
```

Choose a **new** output directory; use `--resume` only for the same interrupted
build. This is a model-building workflow, not a lightweight documentation
check. It downloads precompiled runtime dependencies and compiles SAM3 models
locally; it does not build ROCm/MIGraphX/ORT/Torch from source.

Use `--migraphx-archive` and `--ort-wheel` for local binary release files.
Do not overwrite immutable baseline `tuned.mxr` files or reuse caches from
host MIGraphX 2.16.
