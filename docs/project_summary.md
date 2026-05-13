# SAM3 Video Tracking on AMD Ryzen AI Max+ 395 — Project Summary

**Hardware**: AMD Ryzen AI Max+ 395 (gfx1151), 128GB unified memory (UMA=64GB GPU pool)
**Task**: Open-vocabulary video tracking (text-prompt) + mask-level tracking (box-prompt)

| Pipeline | 504px FPS | 1008px FPS |
|---|---|---|
| Box-prompt (`demo.py`) | **12.21** | **3.22** |
| Text-prompt MIG (`demo_text.py --mig`) | **5.5** | **~1.5** |
| Text-prompt PyTorch | 2.6 | 0.52 |

---

## Software Environment

| Software | Version | Source |
|---|---|---|
| OS / Kernel | Ubuntu 24.04.4 LTS / 6.18.6 | — |
| ROCm HIP (PyTorch) | 7.13.0a20260411 | AMD nightly pip (`rocm-sdk-core`) |
| PyTorch | 2.12.0a0+rocm7.13.0a20260411 | AMD nightly pip |
| MIGraphX | **2.15+patches** | System APT 7.2 + find_splits + NHWC patches |
| ONNX Runtime (MIGraphX EP) | 1.24.2 | [Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build) |
| Python | 3.12.13 | conda |

> **Dual ROCm stack**: nightly pip wheels (ROCm 7.13) provide gfx1151 PyTorch support;
> stable APT (ROCm 7.2) provides MIGraphX. Both coexist via `LD_PRELOAD`.

---

## Two Pipelines

### Box-prompt (`demo.py`) — Tracking only

```
Input frame
  → backbone_mxr_tuned.mxr       [MIGraphX 2.15+patches, MLIR attn, FP16]   ~67ms / ~236ms
  → memory_attention             [ORT MIGraphX EP, FP16] ¹                     ~7ms /  ~60ms
  → mask_decoder_propagate.mxr   [MIGraphX direct API, FP32]                  ~14ms /  ~98ms
  → memory_encoder.mxr           [MIGraphX direct API, FP16]                   ~2ms /   ~7ms
  ─────────────────────────────────────────────────────────────────────────────────
  Total:  504px → ~82ms → 12.21 FPS   |   1008px → ~310ms → 3.22 FPS
```

### Text-prompt (`demo_text.py --mig`) — Detection + Tracking

```
Input frame
  → backbone_detector/tuned.mxr  [MIGraphX 2.15+patches, MLIR attn, FP16]   ~97ms / ~378ms
  → tracker_neck                 [PyTorch]                                      ~4ms /  ~23ms
  → detr_encoder                 [ORT MIGraphX EP, FP16] ¹                    ~12ms /  ~64ms
  → detr_decoder                 [PyTorch]                                     ~11ms /  ~25ms
  → memory_attention             [ORT MIGraphX EP, FP16, padded S7×P32] ¹     ~19ms / ~139ms
  → mask_decoder + memory_encoder [PyTorch]                                     ~4ms /  ~10ms
  ─────────────────────────────────────────────────────────────────────────────────
  Total:  504px → ~169ms → 5.9 FPS   |   1008px → ~722ms → ~1.4 FPS
```

¹ Attention modules route through ORT MIGraphX EP (not direct API) due to FP16
attention numerical bug in direct MIGraphX path — see Finding #8.

**Multi-object**: backbone runs once per frame regardless of object count.
Each additional object adds only ~24ms (memory_attention + mask_decoder + memory_encoder).
Frame-0 detection runs DETR; subsequent frames re-detect via hotstart (confirmed mid-video
after ~15 frames). NMS and fill_holes implemented in pure PyTorch/scipy (ROCm-compatible
fallbacks in `tracker/rocm_patches.py`, applied automatically at import).

---

## Box-prompt Optimization Journey

| Step | 504px FPS | 1008px FPS |
|---|---|---|
| Initial (all modules CPU ORT) | ~0.5 | ~0.5 |
| Backbone → PyTorch ROCm GPU FP16 | 1.14 | 1.14 |
| memory_attention → MIGraphX fixed N=7 | 1.27 | 1.27 |
| 504px resolution + TunableOp + ORT 8 threads | 5.72 | 1.35 |
| MIGraphX 2.15+patches backbone (find_splits + autotuning) | 7.10 | 1.47 |
| dec_prop + mem_enc → MIGraphX GPU (ORT EP, FP32) | 7.71 | 1.56 |
| FP16 for mem_attn + mem_enc (migraphx_fp16_enable) | 8.32 | 1.56 |
| FP16 mem_attn via direct MIGraphX API + autotuned mxr | 8.58 | 1.97 |
| NHWC output fix (GPU contiguous_kernel) — **peak** | **9.46** | **2.39** |
| Revert mem_attn to ORT EP (correctness fix — Finding #8) | 8.21 | 2.31 |
| **+MLIR attention backbone** (detector) | 8.21 | 2.31 |
| **+MLIR attention backbone** (tracker) | **12.21** | **3.22** |

## Text-prompt Optimization Journey

| Stage | Change | FPS @1008px | FPS @504px |
|---|---|---|---|
| 0 | Pure PyTorch | 0.52 | 0.52 |
| A | MIG backbone (detector FPN + last_hidden_state) | 0.96 | — |
| B.2 | MIG detr_encoder (ORT MIG EP FP16) | 1.16 | — |
| B.4 | MIG memory_attention (padded S7×P32, ORT MIG EP) | **1.53** | — |
| C | 504px config-based init (image_size setter cascade) | 1.53 | **5.12** |
| D | MLIR attention backbone (`MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention`) | ~1.5 | **5.5** |

---

## Accuracy

### Box-prompt (DAVIS 2017 val, SAM3OnnxTracker)

| Resolution | DAVIS Mean J |
|---|---|
| 504px | **81.6%** |
| 1008px | **84.8%** |

### Text-prompt mask quality (MIG vs PyTorch, frame-by-frame IoU)

| Resolution | Mean IoU | Min IoU |
|---|---|---|
| 504px | 0.994 | 0.989 |
| 1008px | 0.999 | 0.997 |

### MLIR attention backbone vs baseline (backbone swap only)

| | Mean IoU | Min IoU | Score diff |
|---|---|---|---|
| mlir_attention vs baseline | 0.9995 | 0.998 | 0.001 |

---

## Hardware Context

| Hardware | Resolution | FPS | Notes |
|---|---|---|---|
| NVIDIA H200 | 1008px | ~5–6 | PyTorch, single object |
| NVIDIA RTX 5090 | 1008px | 30+ | TensorRT + ByteTrack |
| **AMD Ryzen AI Max+ 395 (APU)** | **504px** | **12.21** (box) / **5.5** (text) | MIGraphX + MLIR |
| **AMD Ryzen AI Max+ 395 (APU)** | **1008px** | **3.22** (box) / **~1.5** (text) | MIGraphX + MLIR |

The Ryzen AI Max+ 395 is memory-bandwidth-limited (APU, unified memory).
BIOS UMA=64GB maximizes the fast non-coherent GPU pool (see Finding #7).

---

## Key Findings

1. **Two SAM3 implementations numerically incompatible**: Meta encoder FPN features
   differ from HuggingFace by max_diff=4.89 → mask coverage collapses 29.5% → 0.2%.
   Backbone must run as `Sam3TrackerVideoModel.vision_encoder` weights.

2. **memory_attention requires fixed-size ONNX**: dynamic axes trigger MIGraphX
   "Dangling reference" compiler bug. Fixed by exporting with static N=7×HW shape.

3. **MIGRAPHX_GPU_HIP_FLAGS**: newer clang treats `[[lifetimebound]]` as `-Werror`,
   aborting GPU kernel compilation. Set `-Wno-error -Wno-lifetime-safety-intra-tu-suggestions`.

4. **ORT thread tuning**: `intra_op_num_threads=8` reduces decoder+encoder from ~100ms to ~16ms.

5. **TunableOp**: 8 warmup passes enable per-op GEMM kernel autotuning (−8.7ms backbone).

6. **propagate_frame mask bug** (fixed): `binary_mask[0]` extracted row instead of full mask.

7. **UMA BIOS = 64 GB**: gives GPU 64GB fast coarse-grained pool; 128GB starves OS and
   paradoxically reduces GPU bandwidth.

8. **memory_attention direct MIGraphX API FP16 attention numerical bug**:
   produces garbage attention outputs → DAVIS J collapses from ~81% to ~2%.
   Workaround: ORT MIGraphX EP (`migraphx_fp16_enable=1`). Same bug affects
   `detr_encoder`. Root cause: [ROCm/AMDMIGraphX#3596](https://github.com/ROCm/AMDMIGraphX/issues/3596).

9. **memory_attention K=64 kernel cliff**: MIGraphX picks 14× slower kernel at
   `num_object_pointer_tokens=64` (791ms vs 55ms at K≤32). Shim caps at K=32,
   truncates oldest pointers. Quality impact invisible for continuous tracking.

10. **Detector and tracker have different FPN projection weights**: `Sam3VideoModel`
    `detector_model.vision_encoder.neck` (detector FPN) ≠ the FPN baked into the
    tracker backbone. Naïvely reusing tracker backbone for detector path produces
    `presence=0` because fpn[0..2] proj weights differ (max diff 0.06). Two separate
    backbones are compiled: `backbone_tracker/` and `backbone_detector/`.

11. **Sam3VideoConfig.image_size setter cascades to all derived sizes**: correct path
    for non-1008px init is rewriting the config BEFORE `from_pretrained` (not post-init
    `retarget_resolution()`). The setter propagates `backbone_feature_sizes`,
    `memory_attention_rope_feat_sizes`, etc. Only `low_res_mask_size` needs manual patch.

12. **MLIR attention ops give +18% on backbone at gfx1151**: `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=
    "attention"` routes ViT attention through rocMLIR-compiled kernels. Not the default
    on RDNA/gfx11 — must be forced explicitly. No accuracy loss (IoU=0.9995 vs baseline).
    Now default in `compile_backbone_mxr.py`.

13. **hipBLASLt broken on gfx1151**: `MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt` produces
    2× slower backbone (ROCm issue #5643, silent fallback to hipBLAS without tensor cores).
    Do not use.

14. **GPU-resident backbone not worth pursuing**: numpy round-trip overhead only 1.5ms
    (1.6% of 97ms backbone). Kernel execution is the bottleneck; data movement is negligible.

15. **cv_utils kernel (NMS, fill_holes) CUDA-only**: `kernels-community/cv-utils` has no
    ROCm build. Fixed via `tracker/rocm_patches.py`: scipy.ndimage.label for connected
    components, pure-PyTorch greedy NMS on GPU IoU matrix. Applied automatically at
    import via `tracker/__init__.py`.

---

## Pending Work

| Item | Effort | Notes |
|---|---|---|
| Recover 9.46 FPS (box-prompt peak) | 1–2 days | Exclude softmax from FP16 quantization in memory_attention direct API, or wait for MIGraphX upstream fix |
| Apply MLIR attention to box-prompt backbone | ~1 hour | Needs re-export + recompile of `backbone_tracker/tuned.mxr` |
| Webcam real-time demo | half day | Add `--device 0` streaming input to demo scripts |
| 756px intermediate resolution | 1–2 hours | Fills gap between 504px (5.5 FPS) and 1008px (~1.5 FPS); ~20 min compile |
| Flash Attention / AOTriton on gfx1151 | — | Broken as of 2026-05 (3.7× regression, flash-attention issue #2392) — skip |

---

## Analysis Documents

| Topic | File |
|---|---|
| Backbone optimization (find_splits patch, NHWC fix, MLIR attn) | [`analysis/backbone_optimization.md`](../analysis/backbone_optimization.md) |
| Backbone optimization research (gfx1151 env vars, community findings) | [`analysis/backbone_optimization_research.md`](backbone_optimization_research.md) |
| Tracking module optimization (memory_attention, dec/enc, ORT cache) | [`analysis/module_optimization.md`](../analysis/module_optimization.md) |
| MIGraphX backbone investigation (detailed, pre-patch) | [`analysis/migraphx_backbone_investigation.md`](../analysis/migraphx_backbone_investigation.md) |
| 1008px performance deep-dive (NHWC, rocprof, op analysis) | [`analysis/1008px_perf_analysis.md`](../analysis/1008px_perf_analysis.md) |
