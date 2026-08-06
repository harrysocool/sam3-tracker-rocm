# SAM3 ViT XDNA2 NPU benchmark closeout

Date: 2026-08-06 on the development machine

## Outcome

The project objective is to compile and run the complete SAM3 ViT detector
backbone through two independent XDNA2 NPU routes and retain a reproducible
correctness/performance benchmark. Real-time robotics performance is not an
acceptance criterion.

The two routes are technically validated:

1. ONNX Runtime VitisAI EP / flexml general compiler route.
2. MLIR-AIE / IRON custom-kernel route.

The new common benchmark harness is under
`eval/benchmarks/npu_vit/`. It uses the same 504 px input, PyTorch FP16
reference, cosine implementation, warm-up policy, timing statistics, and JSON
schema for both routes.

## Canonical artifacts

| Route | Artifact | SHA256 |
|---|---|---|
| flexml input | detector-backbone ONNX | `f0e5f73d254d83b20421e4d40f16c9f1727a02126c99d1fd465d3caf67721003` |
| flexml output | `backbone_detector_504_v1.rai` | `0272945ed2254ac9b6196a1348600024a26023e5d4d340490ede42aea1f9b5b5` |
| IRON output | P14 M1536 release binary | `a53b5b83f77f6c87beadbd33bf52b669729e9d002ba9f361910aa8d13bc98f1a` |

`verify_benchmark_artifacts.sh` currently verifies the ONNX, `.rai`, complete
IRON release manifest, external weight manifest, and frozen IRON binary.

## Compile-chain verification

### flexml

Creating an ONNX Runtime session with `VitisAIExecutionProvider`, a new cache
directory, and a new cache key is the compile operation. The existing 935 MB
`.rai` was produced from the canonical 1.82 GB ONNX and has a fixed checksum.
`compile_flexml_cache.py` exposes this operation without deleting or
overwriting an existing cache key.

The previously validated stack was:

```text
kernel:       6.14.0-1020-oem
XRT:          2.21.75
firmware:     1.1.2.65
ONNX Runtime: 1.27.0.dev20260709
provider:     VitisAIExecutionProvider
```

### IRON

The P14 release host was rebuilt from its frozen source on 2026-08-06. The
rebuilt executable was byte-identical to the release binary:

```text
source SHA256:  b80d37c3bf372f29ad9c4107598960c661427913b436ebfcbf047c32641ead9e
rebuilt SHA256: a53b5b83f77f6c87beadbd33bf52b669729e9d002ba9f361910aa8d13bc98f1a
release SHA256: a53b5b83f77f6c87beadbd33bf52b669729e9d002ba9f361910aa8d13bc98f1a
```

NPU artifacts are stage-built by the MLIR-AIR shared-GEMM generator, the
MLIR-AIE/Chess ATB M1536 builds, and the grouped-Flash/K4864 control-overlay
build. The release manifest binds their outputs to the host executable.

## Common benchmark result: IRON

Protocol: one warm-up, three measured runs, OMP threads 8, rebuilt binary,
`assets/truck.jpg`, PyTorch FP16 reference.

```text
latency min / mean / p50 / p95 / max:
  697.701 / 701.087 / 701.449 / 703.845 / 704.111 ms

NPU component p50:
  692.065 ms

last-hidden cosine:
  0.992889

FPN cosine:
  p2 0.999443
  p3 0.997853
  p4 0.996987
  p5 0.996359
```

Machine-readable result:
`eval/benchmarks/npu_vit/reference_results/iron_20260806.json`.

Pre-run and post-run `xrt-smi` checks passed, with no D-state task.

## Common benchmark result: flexml

Protocol: one pre-GPU NPU warm-up, three measured runs, canonical detector
ONNX/RAI, `assets/truck.jpg`, PyTorch FP16 reference, attended 10000 ms TDR
profile.

```text
latency min / mean / p50 / p95 / max:
  2487.722 / 2552.703 / 2489.509 / 2661.741 / 2680.878 ms

last-hidden cosine:
  0.951607

FPN cosine:
  p2 0.994255
  p3 0.980527
  p4 0.962097
  p5 0.970781
```

Machine-readable result:
`eval/benchmarks/npu_vit/reference_results/flexml_20260806.json`.

The earlier project ledger quoted aggregate flexml FPN cosine around 0.997,
but no per-level source log was retained and that number may refer to the
tracker-backbone cache or an older reference stack. The canonical detector
ONNX result above uses output names rather than positional assumptions and is
the authoritative closeout measurement. Its last-hidden value is consistent
with the earlier overall cosine around 0.961.

## flexml runtime-profile boundary

The currently installed IRON recovery profile enables scheduler TDR at 2000
ms. The complete flexml backbone is submitted as one NPU job longer than two
seconds, so that profile intentionally recovers the context before a valid
flexml job can finish. A current-stack attempt reproduced exactly this boundary:

```text
DRM scheduler timeout at approximately 2 seconds
VitisAI EP: ERT_CMD_STATE_TIMEOUT
device recovered; no D-state; xrt-smi healthy afterward
```

This is a runtime-policy incompatibility, not an invalid ONNX or `.rai`.
`run_benchmark.sh` now detects a TDR timeout below 5000 ms and refuses to launch
flexml. The final run used the guarded attended helper to load a 10000 ms
profile, completed without a scheduler timeout, and restored the normal 2000
ms profile. Post-run checks confirmed use count zero, no D-state, no hardware
context, inactive performance mode, and a healthy `xrt-smi` response.

## Complete detector mask validation

Backbone cosine is an intermediate signal, so both routes were additionally
run through the complete SAM3 detector on `assets/truck.jpg` with prompt
`truck`. The same PyTorch model run was used as the mask reference. Objects
were paired by maximum mask IoU, with a hard gate of equal nonzero object
count, all objects matched, and minimum IoU at least 0.95.

| Route | Reference / candidate / matched | Mask IoU | Box max error | Score max error |
|---|---:|---:|---:|---:|
| IRON P14 M1536 | 1 / 1 / 1 | 0.999056 | 1 px | 0.001465 |
| flexml/VitisAI EP | 1 / 1 / 1 | 0.997852 | 1 px | 0.001465 |

Both routes passed. Reference, candidate, and difference visualizations were
generated and visually inspected. The flexml difference is confined to a very
small number of boundary pixels, consistent with the measured IoU. Canonical
JSON records and difference images are retained under
`eval/benchmarks/npu_vit/reference_results/`.

The flexml mask run used the guarded 10000 ms profile and completed without a
scheduler timeout. The helper then restored 2000 ms; independent post-run
checks found no D-state, use count zero, no hardware context, inactive
performance mode, and healthy `xrt-smi` output.

## Definition-of-done status

| Requirement | Status |
|---|---|
| canonical 504 px workload | complete |
| flexml compile chain and immutable artifact | complete |
| IRON staged compile provenance and immutable artifact | complete |
| common artifact verifier | complete |
| common benchmark/accuracy JSON harness | complete |
| current IRON compile + run + accuracy gate | complete |
| current flexml common-schema rerun | complete; 10000 ms attended profile restored to 2000 ms |
| complete detector mask gate for both routes | complete; IoU 0.999056 / 0.997852 |
| final comparison and reproduction documentation | complete |

## Scope boundary

The benchmark does not include asynchronous video integration, a 500 ms goal,
or further kernel optimization. Those are documented follow-on topics, not
closeout requirements.
