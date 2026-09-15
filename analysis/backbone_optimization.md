# Backbone Optimization

> **Historical MIGraphX 2.15-era optimization record.** The tables and
> "Remaining Opportunities" below describe that stage, not the complete
> current optimization stack. They do not cover the later GPU-I/O, MIGraphX
> 2.17, or FC1-sink work. Current runtime settings and measured results are in
> the [container guide](../docker/rocm714/README.md) and
> [performance records](../docs/performance.md).

## Overview

The SAM3 vision encoder (backbone) was the dominant bottleneck.
This document covers all backbone-related optimizations applied to the tracker.

---

## Why MIGraphX Backbone? Why Not the Original ONNX?

HuggingFace SAM3 backbone exports with 9324 nodes containing `Transpose`, `Shape`,
and `If` ops that MIGraphX cannot efficiently handle. After onnxsim simplification
(9324 → 2202 nodes), profiling showed **Gemm+MatMul = 63% of runtime** — MIGraphX
used untuned generic GEMM kernels vs PyTorch's TunableOp-autotuned hipBLASLt.

Full investigation: [`migraphx_backbone_investigation.md`](migraphx_backbone_investigation.md)

---

## Optimization Steps

### 1. Resolution Reduction (504px)

Reducing input from 1008px to 504px (area ratio 0.25) cuts backbone from ~560ms to ~140ms.
Accuracy cost: DAVIS J 85.8% → 81.1% (−4.7 pp).

### 2. TunableOp (AMD GEMM autotuner)

8 warmup passes trigger per-operation GEMM kernel autotuning.
Saves ~8.7ms on backbone (backbone: 154ms → 142ms after tuning).

### 3. MIGraphX 2.15+patches Backbone (find_splits patch)

**Problem**: 90 `Split` ops from HF window-attention partition blocked MIGraphX graph fusion.
Each Split created a fusion boundary → thousands of separate kernel launches → 916ms.

**Fix**: Extend `find_splits` in `simplify_algebra.cpp` to handle N-arg ops where
all non-split inputs are constants (PR branch: `fix/find-splits-multi-arg`).
Fixes upstream issue [AMDMIGraphX#4256](https://github.com/ROCm/AMDMIGraphX/issues/4256).

**Result**: With kernel autotuning (no `MIGRAPHX_SKIP_BENCHMARKING`):
- Backbone: 916ms (stock MIGraphX) → 88ms (patched) — matching Meta encoder speed
- Full pipeline: 5.72 FPS → 7.10 FPS (+24%)

Profiling comparison (from rocprof):

| Version | Kernel launches | Latency |
|---|---|---|
| Stock MIGraphX 2.15 | ~thousands (Gemm dominated, 45%) | 916ms |
| MIGraphX 2.15+patches + autotuning | 503 compute kernels | 88ms |
| PyTorch FP16 + TunableOp | (not GPU profiled) | 94ms |

### 4. NHWC Output Fix

**Problem**: MIGraphX uses NHWC (channel-last) GPU layout internally. When outputting
to CPU via `offload_copy=True`, the backbone returned non-C-contiguous tensors.
The CPU then needed ~10ms (504px) / ~94ms (1008px) for NHWC→NCHW transpose.

**Fix**: Three coordinated changes to MIGraphX source (PR branch:
`fix/offload-copy-contiguous-output`):
1. `src/targets/gpu/lowering.cpp` — insert `contiguous` via `insert_precompile_op()`
   before `hip::copy_from_gpu` for non-standard outputs
2. `src/eliminate_contiguous.cpp` — protect `gpu::contiguous` before `hip::copy_from_gpu`
   from being removed (add it to the skip-list alongside `@return`)
3. `src/targets/gpu/device/contiguous.cpp` — use `gs_launch` + `standard_shape.multi(i)`
   (flat indexing) instead of `mi_gs_launch` to avoid crash for channel-last strides

Root cause of crash (before fix): `memory_coloring` assigned the `contiguous` output
buffer to the same GPU memory slot as the NHWC input buffer, causing in-place
NHWC→NCHW — a race condition that corrupted the HIP context.

**Result**: GPU `contiguous_kernel` eliminates the CPU layout transpose entirely.
- 504px: ~10ms saved → 9.0 FPS
- 1008px: ~94ms saved → 2.39 FPS (+53% vs state before fix)

Detailed analysis: [`1008px_perf_analysis.md`](../docs/historical/1008px_perf_analysis.md)

---

## MIGraphX Compatibility Notes

| Module | MIGraphX status | Notes |
|---|---|---|
| `sam3_image_encoder.onnx` (Meta format) | ✅ Works — **not used** | Numerically incompatible with HF mask decoder (max_diff=4.89) |
| HF backbone (9324 nodes, dynamic) | ❌ Pre-patch | `Split` ops block fusion → 916ms |
| HF backbone simplified (2202 nodes) | ✅ After patch | find_splits fix + autotuning → 88ms |
| `MIGRAPHX_GPU_HIP_FLAGS` | Required | `-Wno-error -Wno-lifetime-safety-intra-tu-suggestions` (clang `-Werror` fix) |

---

## Remaining Opportunities

| Opportunity | Potential saving | Mechanism |
|---|---|---|
| Flash Attention (fuse Q@K + Softmax) | ~67ms at 1008px | New MIGraphX fusion pass + ROCm ck::flash_attention |
| RoPE-GEMM fusion | ~12ms at 1008px | MIGraphX graph rewrite pass |
| FP8 quantization | 1.5-2× GEMM speedup | Not yet supported on gfx1151 |
