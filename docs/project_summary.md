# SAM3 Video Tracking ONNX Pipeline — Project Summary

**Hardware**: AMD Ryzen AI Max+ 395 (gfx1151 / Radeon 8060S), 64GB unified memory (APU)  
**Target**: Mask-level video tracking at ≥5 Hz

## Software Environment

| Software | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | 6.18.6-061806-generic |
| ROCm | 7.13.60980 |
| PyTorch | 2.12.0a0+rocm7.13.0a20260411 |
| MIGraphX | 2.15.0 (20250912) |
| ONNX Runtime | 1.24.2 |
| ONNX | 1.21.0 |
| Python | 3.12.13 |
| NumPy | 1.26.4 |
| conda | 26.1.1 |

---

## Final Pipeline Architecture

```
Input frame (504×504)
  ↓
Sam3TrackerVideoModel.vision_encoder    [PyTorch ROCm GPU FP16]
  → fpn_0 (1,256,144,144)  fpn_1 (1,256,72,72)  fpn_2 (1,256,36,36)
  ↓
memory_attention_fixed_N7.onnx          [MIGraphX]   ← propagation frames only
  ↓
mask_decoder_init / propagate.onnx      [CPU ONNX]
  ↓
memory_encoder.onnx                     [CPU ONNX]
  → memory bank FIFO (max 7 frames)
```

**Prompt**: box prompt on frame 0 only; all subsequent frames use pure memory propagation (no prompt required).

---

## Per-Module Latency and Optimizations

### 504px (Final Configuration)

| Module | Initial Latency | Final Latency | Optimization |
|---|---|---|---|
| **backbone** | 561ms (PyTorch FP16, 1008px) | **142ms** | Resolution reduction to 504px (area ratio 0.25) + TunableOp |
| **memory_attention** | 157ms (CPU, dynamic axes) | **16ms** | Fixed N=7 ONNX → MIGraphX (3.3×); dynamic_axes caused dangling reference compiler bug, bypassed with fixed-size export |
| **mask_decoder_propagate** | 54ms | **7ms** | ORT intra_op_num_threads=8 (4×) |
| **memory_encoder** | 45ms | **9ms** | ORT intra_op_num_threads=8 (5×) |
| **Total (propagation frame)** | ~875ms (1.14 FPS) | **175ms (5.72 FPS)** | |

### 1008px (High-Quality Configuration)

| Module | Latency | Share |
|---|---|---|
| backbone | 528ms | 71% |
| memory_attention | 152ms | 21% |
| mask_decoder_propagate | 27ms | 4% |
| memory_encoder | 33ms | 4% |
| **Total (propagation frame)** | **741ms (1.35 FPS)** | |

---

## Step-by-Step Optimization Progress

| Optimization Step | Propagation Latency | FPS | Description |
|---|---|---|---|
| Baseline (PyTorch CPU, 1008px) | ~2000ms | ~0.5 | All modules on CPU |
| Backbone → GPU FP16 | 875ms | 1.14 | PyTorch ROCm GPU |
| memory_attention → MIGraphX | 786ms | 1.27 | Fixed N=7 export bypasses dangling reference bug |
| Resolution reduction to 504px | 195ms | 5.12 | Area ratio 0.25; backbone 561ms → 154ms |
| TunableOp (AMD GEMM autotuner) | 185ms | 5.41 | Backbone −8.7ms via optimized kernel selection |
| ORT 8-thread CPU modules | **175ms** | **5.72** | CPU decoder + encoder: 34ms → 16ms |

---

## MIGraphX Compatibility Summary

| Module | MIGraphX Status | Root Cause / Resolution |
|---|---|---|
| sam3_image_encoder.onnx (meta format) | ✅ Used in detection pipeline | — |
| memory_attention (dynamic axes) | ❌ | dynamic_axes triggers "Dangling reference in module main" compiler bug |
| **memory_attention (fixed N=7)** | **✅** | Fixed-size ONNX bypasses bug; 3.3× speedup |
| memory_encoder | ❌ CPU only | ConvTranspose layout bug |
| mask_decoder_init / propagate | ❌ CPU only | simplify_reshapes error / Segfault |
| Backbone ONNX FP16 | ❌ | _Float16 vectorization bug in MIGraphX; FP32 ONNX is slower than PyTorch GPU FP16 |

**Key hardware constraint**: gfx1151 is an APU with unified memory. The backbone is **memory-bandwidth-limited**, not compute-limited. As a result, kernel-level optimizations (torch.compile, Flash Attention, SDPA) provide no benefit, while resolution reduction is highly effective.

---

## Resolution Comparison

| Metric | 1008px | 504px |
|---|---|---|
| **DAVIS 2017 val Mean J** | **85.8%** | 81.1% |
| **SG val Mean J** (50 seqs, seed=42) | 44.8% | 39.6% |
| Propagation FPS | 1.35 | **5.72** |
| Init frame FPS | 1.68 | **6.22** |
| Backbone latency | 528ms | 142ms |
| Use case | High-quality offline | **Real-time tracking (≥5 Hz)** |

Accuracy trade-off: halving resolution causes approximately 4–5 pp drop in J (DAVIS −4.7 pp, SG −5.2 pp).

---

## Benchmark Results

### DAVIS 2017 val (30 sequences, standard VOS benchmark)

| Configuration | Mean J | FPS |
|---|---|---|
| **Ours — 1008px** | **85.8%** | 1.35 |
| **Ours — 504px** | **81.1%** | 5.72 |
| SAM2 official (J&F, reference) | ~90.7% | — |

### Smartglass SG val (50 sequences, seed=42, egocentric tracking)

| Configuration | Mean J | FPS |
|---|---|---|
| **Ours — 1008px** | **44.8%** | 1.13 |
| **Ours — 504px** | **39.6%** | 4.06 |

The SG dataset covers first-person (smartglass) viewpoints and is significantly harder than DAVIS, containing many small or fast-moving targets (hands, wires, smartphones).

---

## Key Technical Findings

1. **Meta vs. HF backbone incompatibility**: `sam3_image_encoder.onnx` (Meta format) produces FPN features with max_diff=4.89 vs. `Sam3TrackerVideoModel` (HF format), causing mask coverage to drop from 29.5% to 0.2%. The HF backbone must be run via PyTorch directly.

2. **memory_attention must use fixed-size ONNX**: Dynamic axes trigger a MIGraphX compiler bug ("Dangling reference in module main"). Exporting with fixed N=7×HW shape bypasses this and enables 3.3× speedup on MIGraphX.

3. **propagate_frame mask bug (fixed)**: `binary_mask[0]` at 504px incorrectly extracted the first row (1D) instead of the full 2D mask, causing all propagation frames to output 0% mask coverage. Fixed: `masks.squeeze() > 0`.

4. **TunableOp (AMD)**: 8 warmup passes trigger per-operation GEMM kernel autotuning. Subsequent calls use the optimal kernel, reducing backbone latency by ~8.7ms.

5. **ORT thread tuning**: CPU ONNX sessions default to single-threaded. Setting `intra_op_num_threads=8` reduces combined decoder + encoder latency from ~100ms to ~16ms.

---

## Current Best Configuration

```bash
PYTHONPATH=repo/DART/.local_deps MIGRAPHX_SKIP_BENCHMARKING=1 \
    python scripts/onnx/analysis/track_video_onnx.py \
    --imgsz 504 \
    --image <input.jpg> \
    --box x1,y1,x2,y2
```

- **FPS**: 5.72 (propagation), 6.22 (init frame)
- **DAVIS J**: 81.1% at 504px / 85.8% at 1008px
- **Prompt**: box on frame 0; subsequent frames use pure memory propagation

---

## Reference: Software Stack Relationships

```
Your model code
      │
  [PyTorch] ──── The primary AI framework used for model definition,
      │           training, and inference.
      │
   [ROCm]  ──── AMD's GPU driver layer — the equivalent of NVIDIA CUDA.
      │           Enables PyTorch to run on AMD GPUs.
      │
  [AMD GPU] ──── The hardware (Radeon 8060S). Performs matrix operations
                  in parallel at high speed.


                    Alternative path: ONNX-based inference
  [PyTorch]
      │  export
      ▼
   [ONNX]  ──── A portable model file format (like PDF for neural networks).
      │           Hardware- and framework-agnostic.
      │
  [ONNX Runtime (ORT)] ──── The engine that loads and runs ONNX files.
      │
      ├── CPU provider  ──── Runs on CPU (slower, but universally compatible).
      │        └── [CPU]
      │
      └── MIGraphX provider ── AMD's optimizing compiler for ONNX on AMD GPUs.
               └── [AMD GPU]   Compiles the ONNX graph into AMD-specific kernels.
```

**In this project**: PyTorch + ROCm drives the backbone (vision encoder) directly
on the GPU. The remaining modules (memory attention, mask decoder, memory encoder)
are exported to ONNX and dispatched to either MIGraphX (GPU) or the CPU provider,
depending on compatibility.
