# SAM3 Video Tracking on AMD Ryzen AI Max+ 395 — Project Summary

**Hardware**: AMD Ryzen AI Max+ 395 (gfx1151), 128GB unified memory (UMA=64GB GPU pool)
**Task**: Mask-level video tracking at ≥5 Hz
**Result**: **9.46 FPS** at 504px · **2.39 FPS** at 1008px

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
  → backbone_mxr_tuned.mxr       [MIGraphX 2.15+patches, GPU]   95ms / 347ms
  → memory_attention_fp16.mxr    [MIGraphX direct API]     7ms  /  58ms
  → mask_decoder_propagate.mxr   [MIGraphX direct API]     2ms  /   5ms
  → memory_encoder.mxr           [MIGraphX direct API]     1ms  /   4ms
  ─────────────────────────────────────────────────────────────────────
  Total:  504px → 106ms → 9.46 FPS   |   1008px → 419ms → 2.39 FPS
```

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
| NHWC output fix (GPU contiguous_kernel, 3-file MIGraphX patch) | **9.46** | **2.39** |

**vs initial baseline: 504px +19×, 1008px +4.8×**
**vs prior project starting point (5.72 / 1.35 FPS): 504px +65%, 1008px +77%**

---

## Accuracy

| Resolution | DAVIS 2017 val J | SG val J (50 seqs, proxy) |
|---|---|---|
| 504px | 81.1% | 39.6% |
| 1008px | 85.8% | 44.8% |

Resolution trade-off: halving resolution costs ~4–5 pp J (DAVIS −4.7 pp, SG −5.2 pp).

> Official SG evaluation (cgF1 / pHOTA) pending — current numbers are proxy metrics only.

---

## Hardware Context

| Hardware | Resolution | FPS | Notes |
|---|---|---|---|
| NVIDIA H200 | 1008px | ~5–6 | PyTorch, single object |
| NVIDIA RTX 5090 | 1008px | 30+ | TensorRT + ByteTrack |
| **AMD Ryzen AI Max+ 395 (APU)** | **1008px** | **2.39** | MIGraphX all modules |
| **AMD Ryzen AI Max+ 395 (APU)** | **504px** | **9.46** | Half-resolution |

The Ryzen AI Max+ 395 is memory-bandwidth-limited (APU, unified memory).
BIOS UMA=64GB maximizes the fast non-coherent GPU pool (see Finding #7 below).

---

## Key Findings (from initial setup phase)

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
| Official SG eval (cgF1 / pHOTA) | ~4h runtime | Run on all 1686 annotations; save YT-VIS JSON |
| Shared backbone for multi-object | Low (Python host only) | O(N×frames) → O(frames + N×decoder) |
| Memory bank N: 7→5 | Low (re-export) | ~42ms saving at 1008px; check accuracy |
| SAM 3.1 (Object Multiplex) | Medium | Up to 7× faster multi-object |
| Flash Attention in backbone | High (MIGraphX compiler) | ~67ms saving at 1008px |

---

## Reference: Software Stack

```
PyTorch ─── ROCm 7.13 (nightly) ─── AMD GPU (gfx1151)
   │
   └── ONNX export
           │
        [ONNX] ─── ONNX Runtime 1.24.2
                       │
                       ├── MIGraphX EP (GPU)  ← tracking modules
                       └── CPU provider
```

**In this project**: backbone runs via direct MIGraphX Python API (`migraphx.program`);
tracking modules (memory_attention, dec, enc) also use direct MIGraphX API for
maximum efficiency (eliminates ORT EP's NCHW↔NCHWc layout conversion overhead).
