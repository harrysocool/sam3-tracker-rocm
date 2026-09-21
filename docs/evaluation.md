# Evaluation and regression guide

GPU commands use the [supported ROCm 7.14 / MIGraphX 2.17 container](../docker/rocm714/README.md).
Complete [Quick start](../README.md#quick-start), keep `SAM3_MODEL_DIR` and
`SAM3_ONNX_DIR` exported, and run commands from the checkout root in Bash.
Keep outputs under ignored `results/perf/` or an external artifact directory.

## Choose the right check

| Check | What it establishes | What it does not establish |
|---|---|---|
| Provider / CLI check | Correct imported runtime and accepted arguments | Loaded model correctness or speed |
| Installation smoke | Full/hybrid execution, non-empty masks, default decoder and parallel-tail loading | Mask equivalence or source-paced throughput |
| PT-vs-MIG mask regression | Offline backbone/ORT mask agreement | Fixed-decoder equivalence, hybrid behavior, or map safety |
| Canonical latest-frame benchmark | Reproducible 250/1000-arrival service latency and output cadence | Robot-wide latency or an unbounded thermal guarantee |
| Serial/parallel A/B | Offline output equivalence and schedule timing | Default live frame age |
| Source-paced integration check | Latest-frame ownership, drops, service, and age on that input | Original benchmark reproduction or robot-wide latency |
| DAVIS box regression | Tracker quality under a fixed dataset/prompt protocol | Text detection quality or current full-model throughput |

## Checkpoint identity

This optional check identifies whether your locally supplied weights match
those used in the published validation results. Run it on the host with
`SAM3_MODEL_DIR` set to your model directory:

```bash
sha256sum "$SAM3_MODEL_DIR/model.safetensors"
```

The recorded validation checkpoint has SHA256
`6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`.
A matching hash confirms identical file contents. A different hash identifies
a different file and needs its own compatibility and correctness validation.

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
do not load a model or select a GPU/NPU provider. Install pytest separately on
the host if needed: it is an optional developer dependency, not included in
the runtime image or required by Quick start:

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
retains the native DETR decoder. The default live and offline 504px entry points
also use the fixed decoder, which additionally needs a
native-versus-fixed full-path comparison and replay of the exact dropped
latest-frame subsequence. The [fixed-decoder evidence](performance.md#correctness-and-resource-checks)
records those checks; the generic PT-vs-MIG script alone does not cover them.

Hybrid changes also need clean-keyframe, object-loss/recovery, public-ID,
negative-evidence, and reset-lifecycle checks, not just a high mean IoU.

## Canonical latest-frame benchmark

The canonical harness requires a schema-2 `ARTIFACT_MANIFEST.json`, rejects
dirty-source artifacts and tracker-memory environment overrides, verifies the
manifest checksum plus every recorded artifact hash, and rejects unrecorded
files. It also requires ONNX Runtime 1.24.2 with MIGraphX as the primary
provider. It always uses original SAM3 S7/C4, full detection on every consumed
frame, same-frame parallel tail, and no N+1 lookahead. The profile also pins
`assets/blackswan.mp4` by SHA256, prompt `swan`, 24 FPS, five discarded warm
outputs, and the complete aggregation window; custom timing arguments are not
accepted by this harness.

Run three 250-arrival repetitions:

```bash
mkdir -p results/perf/canonical
for run in 1 2 3; do
  ./docker/rocm714/run.sh \
    python eval/benchmarks/benchmark_latest_frame_canonical.py \
      --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
      --profile canonical-250 \
      --out "results/perf/canonical/250-r${run}.json"
done
```

Run the 1000-arrival soak:

```bash
./docker/rocm714/run.sh \
  python eval/benchmarks/benchmark_latest_frame_canonical.py \
    --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
    --profile soak-1000 \
    --out results/perf/canonical/1000.json
```

The output JSON contains every selected source sequence and its queue, service,
and result-age timings. Compare service means only across the same arrival,
warmup, model, power, and artifact conditions.

For release/reference runs, label the machine and attest the measured power
policy before invoking either profile:

```bash
export SAM3_BUILD_HOST_ID=harry-evo-x2
export SAM3_STAPM_LIMIT_W=120
export SAM3_FAST_PPT_LIMIT_W=140
export SAM3_SLOW_PPT_LIMIT_W=120
```

These values are recorded in the result JSON; the benchmark does not request
root access or modify SMU limits. If any value is omitted, the report marks the
power policy as `not_fully_reported` rather than inventing a value.

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
This harness retains the native decoder and does not benchmark the fixed
decoder used by the default entry points or the latest-frame queue.

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
The terminal summary labels all returned inference results as `completed`
and successful publications as `published`, with `age_rejected` and
`superseded` counts reported separately. Compare the `completed` service/age
statistics when assessing inference latency; the `published` statistics exclude
rejected results. Neither includes downstream ROS transport or rendering.
`--max-frames` caps available source arrivals; it does not extend or loop a
short file.

The [published live reference](performance.md#default-full-detection-live-reference)
uses the canonical harness above. Neither this ROS integration check nor a
rendered `demo_live.py` run exactly reproduces it.

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

## External datasets

DAVIS and SA-Co datasets are external prerequisites; this repository does not
distribute them or track a machine-specific `dataset` symlink. The local
`dataset/` path is ignored. Put the actual files there, or pass an explicit path
to host-side tools. Inside the supported container, inputs must be under a
configured mount; an absolute symlink to an unmounted host path is not visible.

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
  --output "$HOME/sam3-artifacts/gpu/clean-validation-0.2.0-rc6"
```

Choose a **new** output directory; use `--resume` only for the same interrupted
build. This is a model-building workflow, not a lightweight documentation
check. It downloads precompiled runtime dependencies and compiles SAM3 models
locally; it does not build ROCm/MIGraphX/ORT/Torch from source.

Use `--migraphx-archive` and `--ort-wheel` for local binary release files.
Do not overwrite immutable baseline `tuned.mxr` files or reuse caches from
host MIGraphX 2.16.
