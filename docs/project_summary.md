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
| **MIGraphX-ize the text-prompt detector path** | 2–3 days | `demo_text.py` is PyTorch-only (~0.5 FPS prop vs 8.21 box). Steps: (a) re-export backbone with `last_hidden_state` — detector needs the 1024-d ViT features, current backbone outputs only 256-d FPN; (b) wire CLIP text encoder + DETR head into a hybrid path (PyTorch on frame 0, MIGraphX after); (c) port `Sam3VideoModel` NMS + presence gating to the OnnxTracker side; (d) resolve torch-ROCm-7.13-nightly vs patched-MIGraphX-7.2 lib conflict (LD_PRELOAD workaround works). Brings text-prompt to box-prompt FPS — unlocks open-vocabulary real-time tracking. |
| Full SG eval on 1686 annotations (text-prompt) | ~half day runtime | 300-seq box-prompt subset done (HOTA 0.179 / 0.183). The official cgF1 / pHOTA numbers reported in the SAM3 paper require text prompts and the full split — depends on text-prompt MIGraphX path above, or accept slow PyTorch eval (~1h per 100 seqs) |
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

---

## Update 2026-05-12 — Text-prompt path: 2.94× over PT baseline

Stage A + B brought the `Sam3VideoModel` text-prompt path from 0.52 FPS pure
PyTorch to **1.5 FPS @ 1008px** (single-object continuous tracking), with mask
output identical to the PT reference (visually verified on 30-frame swan video,
score 0.84 in both paths).

| Stage | What changed | Prop FPS @ 1008px |
|---|---|---|
| 0 | Pure PyTorch baseline | 0.52 |
| A | MIG backbone (`backbone_detector` with `last_hidden_state` output) | 0.96 |
| B.2 | MIG `detr_encoder` (ORT MIG EP, fp16) | 1.16 |
| B.4 | MIG `memory_attention` (padded fixed-shape S7×P32, ORT MIG EP) | **1.53** |

**Architecture choices learned the hard way:**

- **Detector and tracker have *different* FPN proj weights.** `Sam3VideoModel`
  has `detector_model.vision_encoder.neck` (detector FPN) and
  `tracker_neck` (tracker FPN, separate `Sam3VisionNeck`). The original MIG
  backbone was exported from `Sam3TrackerVideoModel.vision_encoder` (tracker
  weights) — works for the box-prompt path; produces `presence=0` when fed
  to `Sam3Model` (detector path). New `--backbone-source detector` exports a
  separate backbone with detector FPN weights, plus `last_hidden_state` output
  so `tracker_neck` can re-run the tracker FPN on the cached ViT tokens.

- **Any module containing attention layers must go through ORT MIG EP** with
  `migraphx_fp16_enable=1`. Direct `migraphx.parse_onnx + quantize_fp16`
  produces NaN outputs for these (memory_attention, detr_encoder); even
  `--no-fp16` produces ~0.05 max-diff that breaks downstream detection
  thresholds. ORT EP uses a different FP16 quantization pipeline that
  produces correct results (this is why `memory_attention` in the box-prompt
  path was already on ORT EP — Finding #8).

- **MIGraphX kernel-selection cliff at K=64.** The padded `memory_attention`
  ONNX with `num_object_pointer_tokens=64` (the architectural max) compiles
  to a 14× slower kernel (791 ms vs 55 ms at K≤32). The shim caps at K=32
  and truncates the oldest pointers when PT would have more. For continuous
  tracking this is invisible; long-video re-identification across long
  disappearances may degrade slightly.

**What is *not* MIG-ized in this path** (and roughly what they cost per
propagation frame, dominating the remaining ~600 ms):

- `vision_encoder` itself (~380 ms — most of it is numpy↔torch + GPU↔CPU
  transfer through the MIG bridge, not raw compute)
- `detr_decoder` (~24 ms)
- `mask_decoder` (~9 ms)
- `dot_product_scoring`, `text_projection`, `presence_head` (<1 ms)
- `tracker_neck` (~24 ms)
- `tracker_model.mask_decoder` propagate (~5 ms)
- Host-side gather/planning/execution (~25 ms)

Going beyond ~1.5 FPS @ 1008px requires either keeping data on GPU through
the MIG bridge (HIP IPC; difficult on this stack) or running the full
pipeline at 504px (planned next: should land around 3–4 FPS).
