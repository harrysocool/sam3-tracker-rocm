# SAM3 Video Tracking on AMD Ryzen AI Max+ 395 — Project Summary

**Hardware**: AMD Ryzen AI Max+ 395 (gfx1151), 128GB unified memory (UMA=64GB GPU pool)
**Task**: Mask-level video tracking at ≥5 Hz
**Result**: **8.21 FPS** at 504px · **2.31 FPS** at 1008px (regressed from peak 9.46 / 2.39 for correctness — see Finding #8)

---

## Software Environment

| Software | Version | Source |
|---|---|---|
| OS / Kernel | Ubuntu 24.04.4 LTS / 6.18.6 | — |
| ROCm HIP (PyTorch) | 7.13.0a20260411 | AMD nightly pip (`rocm-sdk-core`) |
| PyTorch | 2.12.0a0+rocm7.13.0a20260411 | AMD nightly pip |
| MIGraphX | 2.15+patches | System APT 7.2 + find_splits + NHWC patches |
| ONNX Runtime (MIGraphX EP) | 1.24.2 | [Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build) |
| Python | 3.12.13 | conda |

> **Dual ROCm stack**: nightly pip wheels (ROCm 7.13) provide gfx1151 PyTorch support;
> stable APT (ROCm 7.2) provides MIGraphX. Both coexist without conflict.
> See [README](../README.md) for setup details.

---

## Final Pipeline (propagation frame)

```
Input frame
  → backbone_mxr_tuned.mxr       [MIGraphX 2.15+patches, GPU]    92ms / 342ms
  → memory_attention             [ORT MIGraphX EP, FP16] ⁸        7ms /  60ms
  → mask_decoder_propagate.mxr   [MIGraphX direct API, FP32]     14ms /  98ms
  → memory_encoder.mxr           [MIGraphX direct API, FP16]      2ms /   7ms
  ─────────────────────────────────────────────────────────────────────
  Total:  504px → 115ms → 8.21 FPS   |   1008px → 432ms → 2.31 FPS
```

⁸ memory_attention runs through ORT MIGraphX EP rather than a precompiled `.mxr`
because the direct MIGraphX FP16 attention kernel has a numerical bug (see Finding #8).

---

## Complete Optimization Journey

| Step | 504px FPS | 1008px FPS |
|---|---|---|
| **Initial state** (all modules on CPU ORT) | ~0.5 | ~0.5 |
| Backbone → PyTorch ROCm GPU FP16 | 1.14 | 1.14 |
| memory_attention → MIGraphX fixed N=7 | 1.27 | 1.27 |
| 504px resolution + TunableOp + ORT 8 threads | **5.72** | **1.35** |
| MIGraphX 2.15+patches backbone (find_splits patch + autotuning) | 7.10 | 1.47 |
| dec_prop + mem_enc → MIGraphX GPU (ORT EP, FP32) | 7.71 | 1.56 |
| FP16 enabled for mem_attn + mem_enc (ORT EP, migraphx_fp16_enable) | 8.32 | 1.56 |
| FP16 mem_attn via direct MIGraphX API + autotuned mxr cache | 8.58 | 1.97 |
| NHWC output fix (GPU contiguous_kernel, 3-file MIGraphX patch) — **peak** | **9.46** | **2.39** |
| **Revert mem_attn to ORT EP** (correctness fix, see Finding #8) — **current** | **8.21** | **2.31** |

**vs initial baseline: 504px +16×, 1008px +4.6×**
**vs prior project starting point (5.72 / 1.35 FPS): 504px +43%, 1008px +71%**

---

## Accuracy

| Resolution | DAVIS 2017 val J | SG val J (50 seqs, proxy) | SG HOTA (mask, 300-seq subset) ⁹ |
|---|---|---|---|
| 504px | 81.5% | 40.4% | 0.179 |
| 1008px | 84.8% | 44.0% | 0.183 |

Resolution trade-off: halving resolution costs ~3 pp on DAVIS J, ~3.6 pp on SG proxy J.

⁹ SG HOTA from official SA-Co/VEval evaluator on a 300-sequence subset, **box-prompt**
tracker. cgF1@50 is low (504px=0.047, 1008px=0.024) because cgF1 scores concept
grounding — a box-prompt tracker has no concepts to ground. HOTA reflects pure
tracking quality. Full 1686-seq eval and text-prompt path remain pending.

---

## Hardware Context

| Hardware | Resolution | FPS | Notes |
|---|---|---|---|
| NVIDIA H200 | 1008px | ~5–6 | PyTorch, single object |
| NVIDIA RTX 5090 | 1008px | 30+ | TensorRT + ByteTrack |
| **AMD Ryzen AI Max+ 395 (APU)** | **1008px** | **2.31** | MIGraphX backbone + ORT EP mem_attn |
| **AMD Ryzen AI Max+ 395 (APU)** | **504px** | **8.21** | Half-resolution |

The Ryzen AI Max+ 395 is memory-bandwidth-limited (APU, unified memory).
BIOS UMA=64GB maximizes the fast non-coherent GPU pool (see Finding #7 below).

---

## Key Findings

1. **Two SAM3 implementations are numerically incompatible**: Meta encoder FPN features
   differ from HuggingFace by max_diff=4.89 → mask coverage collapses 29.5% → 0.2%.
   The backbone MUST run as `Sam3TrackerVideoModel.vision_encoder` (PyTorch / MIGraphX ONNX).

2. **memory_attention requires fixed-size ONNX**: dynamic axes trigger MIGraphX
   "Dangling reference" compiler bug. Fixed by exporting with static N=7×HW shape.

3. **MIGRAPHX_GPU_HIP_FLAGS**: newer clang treats `[[lifetimebound]]` as `-Werror`,
   aborting GPU kernel compilation. Set `-Wno-error -Wno-lifetime-safety-intra-tu-suggestions`.

4. **ORT thread tuning**: `intra_op_num_threads=8` reduces decoder+encoder from ~100ms to ~16ms.

5. **TunableOp**: 8 warmup passes enable per-op GEMM kernel autotuning (−8.7ms backbone).

6. **propagate_frame mask bug** (fixed): `binary_mask[0]` extracted row instead of full mask.

7. **UMA BIOS = 64 GB**: gives GPU 64GB fast coarse-grained pool; 128GB starves OS and
   paradoxically reduces GPU bandwidth.

8. **memory_attention direct MIGraphX API has FP16 attention numerical bug**:
   produces garbage attention outputs → dec_propagate emits all-negative logits →
   DAVIS J collapses from ~81% to ~2%. Workaround: route memory_attention through
   ORT MIGraphX EP (loses ~16 ms/frame, ≈13% FPS, vs the direct-API peak). Root
   cause similar to [ROCm/AMDMIGraphX#3596](https://github.com/ROCm/AMDMIGraphX/issues/3596).
   Recovery options: upstream patch, or excluding softmax from FP16 quantization
   when re-exporting the ONNX (estimated 1–2 days).

---

## Analysis Documents

| Topic | File |
|---|---|
| Backbone optimization (find_splits patch, NHWC fix, MIGraphX 2.15+patches) | [`analysis/backbone_optimization.md`](../analysis/backbone_optimization.md) |
| Tracking module optimization (memory_attention, dec/enc, ORT cache) | [`analysis/module_optimization.md`](../analysis/module_optimization.md) |
| MIGraphX backbone investigation (detailed, pre-patch) | [`analysis/migraphx_backbone_investigation.md`](../analysis/migraphx_backbone_investigation.md) |
| 1008px performance deep-dive (NHWC, rocprof, op analysis) | [`analysis/1008px_perf_analysis.md`](../analysis/1008px_perf_analysis.md) |

---

## Pending Work

| Item | Effort | Notes |
|---|---|---|
| Recover 9.46 FPS: fix mem_attn direct-API path | 1–2 days | Exclude softmax from FP16 quantization when re-exporting ONNX, or wait for upstream MIGraphX fix |
| Full SG eval (1686 annotations) + text-prompt path | ~half day runtime | 300-seq subset done (HOTA 0.179 / 0.183); text-prompt requires Sam3VideoModel detector integration |
| Shared backbone for multi-object | Low (Python host only) | O(N×frames) → O(frames + N×decoder) |
| Memory bank N: 7→5 | Low (re-export) | ~42ms saving at 1008px; check accuracy |
| SAM 3.1 (Object Multiplex) | Medium | Up to 7× faster multi-object |
| Q@K + Softmax fusion in backbone window-attention | High (new MIGraphX pass + ROCm `ck::flash_attention` integration) | est. ~30–67ms at 1008px (unverified) |

---

## Reference: Software Stack

```
PyTorch ─── ROCm 7.13 (nightly) ─── AMD GPU (gfx1151)
   │
   └── ONNX export
           │
        [ONNX] ─── ONNX Runtime 1.24.2
                       │
                       ├── MIGraphX EP (GPU)  ← memory_attention (forced via EP, see #8)
                       └── CPU provider
```

**In this project**: backbone, mask_decoder_propagate, and memory_encoder run via
direct MIGraphX Python API (`migraphx.program`) for maximum efficiency (eliminates
ORT EP's NCHW↔NCHWc layout conversion overhead). `memory_attention` is the
exception — it must route through ORT MIGraphX EP because the direct API path has
an FP16 attention numerical bug (Finding #8).
