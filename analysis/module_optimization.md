# Tracking Module Optimization

## Overview

The four tracking modules (memory_attention, mask_decoder_propagate,
memory_encoder, mask_decoder_init) were initially all running on CPU ORT.
This document covers all optimizations applied to them.

---

## memory_attention

### Step 1: CPU ORT → MIGraphX ORT EP (fixed N=7)

**Problem 1**: Dynamic axes triggered a MIGraphX compiler bug
("Dangling reference in module main"). Fixed by exporting with static N=7×HW shape.

**Problem 2**: `MIGRAPHX_GPU_HIP_FLAGS` — newer clang treats `[[lifetimebound]]`
as `-Werror`, aborting GPU kernel compilation. Must set:
```
MIGRAPHX_GPU_HIP_FLAGS=-Wno-error -Wno-lifetime-safety-intra-tu-suggestions
```
(set automatically in `tracker.py` via `os.environ.setdefault()`).

**Problem 3**: JIT cache — without `migraphx_model_cache_dir`, MIGraphX recompiles
every process (~120s cold start). Setting this option enables persistent cache.

**Result**: 157ms (CPU) → 16ms (MIGraphX) — **10× speedup**.

### Step 2: FP16 + Direct migraphx API

**FP16**: `migrachx_fp16_enable=1` gives 2.76× speedup (max_diff=0.012, safe).

**Direct API problem**: ORT's MIGraphX EP silently falls back to CPU at 1008px
when the backbone is in GPU memory (FP16 compilation OOMs during inference).
Fixed by using the direct `migraphx` Python API (`MIGraphXSession` class) with
an autotuned `.mxr` cache file.

**Kernel autotuning critical**: without autotuning, memory_attention = 758ms;
with = 56ms at 1008px (14× difference).

**Result**:
- 504px: 16ms → 7ms (FP16 + direct API)
- 1008px: 152ms → 58ms (FP16 + direct API, autotuned .mxr)

### Step 3: ORT Session Cache

Using `prewarm_ort_cache.py` pre-compiles all MIG sessions without the backbone
in GPU memory (avoids OOM during FP16 compilation). Startup: 2+ min → ~5s.

---

## dec_propagate, memory_encoder, dec_init

### CPU ORT → Direct migraphx Python API

**Problem**: ORT's MIGraphX EP adds `ReorderInput`/`ReorderOutput` — CPU-side
NCHW↔NCHWc layout conversion. At 1008px, `ReorderInput` alone = 42ms (37% of total).

**Fix**: Use direct `migraphx.parse_onnx + compile` (same as backbone).
This completely eliminates layout conversion overhead.

**FP16 note**: dec_propagate and dec_init use FP32 — FP16 corrupts ConvTranspose
upsampling (max_diff=15.3 for dec_propagate). memory_encoder uses FP16 safely.

**Result**:

| Module | ORT (CPU/MIG) | Direct MIG | Speedup |
|---|---|---|---|
| dec_propagate | 99ms (MIG FP32 ORT) | **5ms** | **21×** |
| memory_encoder | 7ms (MIG FP16 ORT) | **4ms** | 1.75× |
| dec_init | 118ms (CPU ORT) | **11ms** | **11×** |

---

## Final Module Timing

### 504px

| Module | Latency | Backend |
|---|---|---|
| backbone | 95ms | MIGraphX direct API (NHWC fix) |
| memory_attention | 7ms | MIGraphX direct API FP16 |
| dec_propagate | 2ms | MIGraphX direct API FP32 |
| memory_encoder | 1ms | MIGraphX direct API FP16 |
| **Total** | **106ms → 9.46 FPS** | |

### 1008px

| Module | Latency | Backend |
|---|---|---|
| backbone | 347ms | MIGraphX direct API (NHWC fix) |
| memory_attention | 58ms | MIGraphX direct API FP16 |
| dec_propagate | 5ms | MIGraphX direct API FP32 |
| memory_encoder | 4ms | MIGraphX direct API FP16 |
| **Total** | **419ms → 2.39 FPS** | |

---

## Remaining Opportunities

| Opportunity | Potential gain | Notes |
|---|---|---|
| Memory bank N: 7→5 | ~42ms at 1008px mem_attn | Need to evaluate DAVIS accuracy |
| dec_init on GPU (already done) | 118ms → 11ms | Done |
| Multi-object shared backbone | ~N× for N objects | Python host change only |
