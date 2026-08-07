# SAM3 ViT XDNA2 benchmark — frozen status

Status: frozen on 2026-08-06

Local release identifier: `sam3-vit-xdna2-benchmark-v1`

## Objective

Compile and run the complete SAM3 ViT detector backbone through two independent
XDNA2 NPU routes, and retain a reproducible benchmark that covers artifacts,
latency, intermediate features, and complete detector masks.

This objective is complete. Real-time robotics, asynchronous keyframe merge,
and further kernel optimization are outside the frozen scope.

## Canonical workload and results

```text
input:        [1, 3, 504, 504]
tokens:       1296
hidden size:  1024
layers:       32
image:        assets/truck.jpg
mask prompt:  truck
reference:    PyTorch FP16
```

| Route | Backbone p50 / p95 | Hidden cosine | Minimum FPN cosine | Complete mask IoU |
|---|---:|---:|---:|---:|
| MLIR-AIE / IRON P14 | 701.449 / 703.845 ms | 0.992889 | 0.996359 | 0.999056 |
| flexml / VitisAI EP | 2489.509 / 2661.741 ms | 0.951607 | 0.962097 | 0.997852 |

Both complete detector gates returned one reference object and one NPU object,
matched all objects, limited the box difference to 1 pixel, and limited the
score difference to 0.001465. The hard mask gate was IoU >= 0.95.

## Frozen source history

The benchmark was finalized by these local-only tracker commits:

```text
76d30f8  feat(npu_bench): add dual-route SAM3 ViT benchmark
8c76584  test(npu_bench): validate flexml with attended TDR profile
ba711a2  test(npu_bench): validate complete detector masks
```

The branch is `perf/iron-sub1s-backbone`. It has no upstream, and no matching
remote branch existed at freeze time. Nothing was pushed, tagged remotely, or
submitted as a PR.

## Reproduction entry points

Run from the tracker repository:

```bash
bash eval/benchmarks/npu_vit/verify_benchmark_artifacts.sh

bash eval/benchmarks/npu_vit/run_benchmark.sh iron \
  --warmup 1 --runs 5 --omp-threads 8

bash eval/benchmarks/npu_vit/run_mask_validation.sh iron \
  --prompt truck \
  --output results/npu_vit_benchmark/iron_mask.json \
  --visual-dir results/npu_vit_benchmark/masks/iron
```

The full flexml NPU job is longer than the normal 2000 ms scheduler timeout.
Only run the attended helper with physical recovery available:

```bash
FLEXML_MASK_ONLY=1 \
  bash eval/benchmarks/npu_vit/run_flexml_attended_tdr.sh
```

It validates the approved module, temporarily uses a 10000 ms timeout, and
restores 2000 ms through an exit trap.

## Compile reproducibility boundary

- flexml: ONNX-to-RAI compilation was historically completed. The closeout
  reran and validated the immutable `.rai`; it did not perform another cold
  20-minute compilation. `compile_flexml_cache.py` is the non-overwriting cold
  compile entry point.
- IRON: the frozen C++ host was rebuilt during closeout and was byte-identical
  to the release binary. XCLBIN, instruction, ATB, and control-overlay artifacts
  were reused from their previously validated staged builds. Their build entry
  points and checksums are documented; the closeout did not cold-rebuild every
  compiler stage.

The project therefore proves that both compile chains were completed and that
their frozen outputs remain runnable and correct. It does not claim a fresh
single-command rebuild of every NPU artifact from an empty workspace.

## Runtime state after validation

```text
amdxdna TDR:       2000 ms
amdxdna use count: 0
D-state tasks:     none
AIE contexts:      none
performance mode:  inactive
xrt-smi:           healthy
```

Validated runtime stack:

```text
kernel:        6.14.0-1020-oem
XRT:           2.21.75 (4eb1f439)
NPU firmware:  1.1.2.65
ONNX Runtime:  1.27.0.dev20260709
```

## Retained evidence

- `README.md`: commands, route definitions, and scheduler constraints.
- `reference_results/`: canonical latency/feature and mask JSON records.
- `reference_results/masks/`: compact difference visualizations.
- `analysis/sam3_vit_npu_benchmark_closeout_20260806.md`: full closeout report.
- `FROZEN_ARTIFACTS.sha256`: canonical external artifact and local evidence
  checksums.
- The external closeout directory contains a Git bundle, bundle verification,
  repository state, and the same manifest without copying multi-gigabyte model
  artifacts.

## Resume policy

Resume this project only for a concrete regression, publication request, or a
new explicitly scoped model/compiler objective. Do not reopen asynchronous
video integration, a 500 ms target, or exploratory kernel work under this
frozen benchmark milestone.
