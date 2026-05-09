# SAM3 Tracker 1008px Performance Analysis

**Date**: 2026-05-08  
**Hardware**: AMD Ryzen AI Max+ 395 (gfx1151), 128 GB UMA (64 GB GPU pool)  
**Baseline**: MIGraphX 2.16.0 backbone + MIGraphX ORT sessions

---

## Pipeline Timing (propagation frame)

### Before optimization (MIGraphX backbone, CPU ORT for dec/enc)

| Module | 1008px | 504px | Ratio |
|---|---|---|---|
| backbone (`backbone_mxr_tuned.mxr`) | 341 ms | 94 ms | 3.6× |
| memory_attention (`fixed_N7`, MIGraphX) | 189 ms | 19 ms | 10× |
| mask_decoder_propagate (CPU ORT) | 115 ms | 17 ms | 6.8× |
| memory_encoder (CPU ORT) | 29 ms | 8 ms | 3.6× |
| **Total → FPS** | **678 ms → 1.47 FPS** | **140 ms → 7.10 FPS** | 4.8× |

### After optimization (dec_prop + mem_enc switched to MIGraphX GPU)

| Module | 1008px | 504px | Notes |
|---|---|---|---|
| backbone | 338 ms | 92 ms | unchanged |
| memory_attention | 189 ms | 19 ms | unchanged |
| mask_decoder_propagate | **99 ms** | **15 ms** | MIGraphX GPU |
| memory_encoder | **10 ms** | **2 ms** | MIGraphX GPU |
| **Total → FPS** | **640 ms → 1.56 FPS** | **130 ms → 7.71 FPS** | |

The switch from CPU to MIGraphX for dec_prop and mem_enc saved **38ms at 1008px** (+6% FPS) and **10ms at 504px** (+8% FPS).

---

## Module-Level Root Cause Analysis

### 1. mask_decoder_propagate: CPU bottleneck at 1008px

ORT profiler breakdown at 1008px on CPU (115ms):

| Op | Time | % | Root cause |
|---|---|---|---|
| `ReorderInput` ×2 | 56 ms | **49%** | NCHW→NCHWc layout conversion for SIMD |
| `/conv_s0/Conv` (288×288) | 32 ms | 28% | Large spatial convolution |
| `ConvTranspose` ×2 | 44 ms | 38% | Upsampling to 1008×1008 |
| `/Resize` | 17 ms | 15% | Bilinear resize |
| `ReorderOutput` | 14 ms | 12% | NCHWc→NCHW layout back-conversion |

**Key finding**: `ReorderInput`/`ReorderOutput` (NCHW↔NCHWc blocking for x86 SIMD) accounts for 49% of dec_prop CPU time at 1008px. These conversions don't exist on GPU. Switching to MIGraphX removes this overhead entirely.

After switch: **115ms → 99ms** (MIGraphX runs ConvTranspose + Resize on GPU, no layout conversion overhead).

### 2. memory_encoder: most impactful optimization

| Backend | 1008px | 504px |
|---|---|---|
| CPU ORT | 29 ms | 8 ms |
| MIGraphX GPU | **5 ms** | **2 ms** |
| Speedup | **5.8×** | **4×** |

The memory_encoder has conv + ConvTranspose + elementwise ops. All run natively on GPU. CPU overhead from memory allocation and data layout conversion dominates on CPU.

### 3. memory_attention: quadratic scaling is fundamental

At 1008px, the cross-attention processes:
- Current features: 5184 tokens (72×72 feature map)
- Memory bank: 7 × 5184 = **36,288 memory tokens**
- Attention matrix: 5184 × 36,288 = **188M** multiply-adds per head

vs 504px: 1,296 × 9,072 = **11.8M** multiply-adds per head → **16× more compute**.

Actual slowdown: 189ms / 19ms = **10×** — MIGraphX is roughly 60% efficient vs theoretical maximum due to memory bandwidth saturation at this scale.

---

## Backbone Kernel Profile (rocprof, 1008px)

Measured with `rocprof --stats` on backbone `.mxr` only (no ORT sessions).  
Note: rocprof adds ~2× overhead; ratios are accurate, absolute times are inflated.

| Kernel | % total | Actual ~ms | c/run | us/call | vs 504px |
|---|---|---|---|---|---|
| `mlir_dot_reshape_add_mul_erf_mul_add_mul` | 21.3% | ~73ms | 32 | 4869 | **5.1×** |
| `convert_mul_reduce_max_..._div` (Softmax) | 15.3% | ~52ms | 32 | 3503 | **3.7×** |
| `mlir_transpose_dot` (Q@K / QK@V) | 13.2% | ~45ms | 32 | 3019 | **1.7×** |
| `mlir_dot_reshape` | 12.5% | ~43ms | 27 | 3392 | — |
| `mlir_dot_broadcast_add_add` (Proj) | 8.5% | ~29ms | 27 | 2302 | — |
| `mlir_reshape_slice_..._dot` | 4.6% | ~16ms | 27 | 1249 | — |
| `neg_noop_concat_noop_kernel` (**RoPE**) | 4.0% | ~14ms | 64 | 453 | **10.8×** |
| `mlir_slice_reshape_..._dot` | 3.4% | ~12ms | 5 | 4942 | — |

### Category breakdown (actual ~340ms)

| Category | Time | % | Notes |
|---|---|---|---|
| GELU FFN GEMM | ~73ms | 21% | mlir_dot + erf fused, 32 layers |
| Softmax | ~52ms | 15% | 32 layers × 144 windows |
| Window reshape+GEMM | ~49ms | 14% | 8.2× vs 504px — super-linear |
| Q@K / QK@V attn GEMM | ~45ms | 13% | only 1.7× — MLIR batches efficiently |
| Projection GEMM | ~45ms | 13% | 3.75× — linear with tokens |
| Window slice+GEMM | ~27ms | 8% | 3× |
| **RoPE** | **~14ms** | **4%** | **10.8× — cache miss (see below)** |
| FPN Conv | ~8ms | 2.5% | 4× — linear with area |

### Scaling anomalies

**RoPE is 10.8× slower (expected 4×)**

64 `neg_noop_concat_noop` kernels, each from 19 μs → 452 μs (+23.5×/call).

Cause: 504px RoPE tensors fit in L2 cache (~1.3 MB total). 1008px tensors
are 5.2 MB — a full cache eviction per kernel, with main memory round-trip.
Expected 4× from linear token scaling, actual 23.5×/call due to cold-cache penalty.

**Window reshape+GEMM is 8.2× slower (expected 4×)**

At 1008px, 4× more windows (36→144) but also each GEMM accesses cold working
sets that no longer fit in cache, amplifying the slowdown.

**Q@K GEMM is only 1.7× slower (expected 4×)**

MLIR batches multiple windows into a single large GEMM at 1008px, improving
hardware utilization. The per-call time is similar (3019 vs 3018 μs) but
there are effectively 4× more windows packed into the same 32 kernel calls.

---

## Remaining Optimization Opportunities

### ONNX graph analysis: exporter comparison (2026-05-08)

The 1008px backbone was originally exported with PyTorch 2.9's new `torch.export`-based
ONNX exporter (opset 18), while the 504px used the legacy TorchScript exporter (opset 17).
This produces structurally different graphs:

| | New exporter (1008px) | Legacy exporter |
|---|---|---|
| Split ops | **32** | 90 |
| Cast ops | 132 | 131 |
| MatMul ops | 192 | 204 |
| GELU kernel name | `mlir_dot_reshape_add_mul_erf_mul_add_mul` | `mlir_dot_add_mul_erf_add_mul` |
| GELU time | 73ms | ~74ms |

**Key finding**: The Reshape inside the new exporter's GELU kernel name is absorbed by MLIR
and does NOT cause extra overhead. Re-exporting with the legacy exporter produces **identical
performance** (338ms vs 332ms, within measurement noise). The backbone ONNX graph structure
is already at or near the MIGraphX fusion optimum for the current compiler.

The 132 Cast ops (float32↔float16 type boundaries) are also neutralized by
`migraphx.quantize_fp16()` before compilation — they do not appear as separate GPU kernels
in the final compiled program.

### Tier 1 — Compiler changes to MIGraphX (significant effort)

| Opportunity | Potential saving | Mechanism |
|---|---|---|
| Flash Attention (fuse Q@K + Softmax) | ~67ms (20%) on 1008px backbone | New fusion pass calling ROCm flash-attn kernel |
| RoPE fusion into adjacent GEMM | ~12ms (3.5%) on 1008px backbone | Eliminate 64 independent cache-miss-heavy memory accesses |

These require new MIGraphX passes or op registrations. The upstream issue
[AMDMIGraphX#4256](https://github.com/ROCm/AMDMIGraphX/issues/4256) (our
`find_splits` patch) addresses fusion; flash attention and RoPE fusion would be
separate, larger efforts.

### Tier 2 — Memory bank size reduction (accuracy tradeoff)

Reducing memory slots from N=7 to N=4:
- memory_attention: 189ms × (4/7) ≈ **108ms** (saves ~81ms)
- FPS impact: 640ms → ~560ms → **~1.78 FPS**
- Accuracy impact: needs evaluation on DAVIS 2017 val

To implement: export `memory_attention_fixed_N4.onnx` via
`export/export_tracker_modules.py --fixed-slots 4 --imgsz 1008`.

### Tier 3 — Precision reduction (future)

FP8 quantization of backbone GEMM ops would theoretically halve bandwidth and
compute for the bottleneck GELU FFN / projection kernels. MIGraphX does not
yet support FP8 on gfx1151. No near-term path.

### Tier 4 — Feature map downsampling (major accuracy tradeoff)

Downsampling fpn_2 from 72×72 to 36×36 before memory operations would reduce
memory_attention by 4×. This changes the model's spatial resolution and would
require retraining. Not pursued.

---

## Summary Table

| Config | 1008px total | FPS | Change |
|---|---|---|---|
| PyTorch backbone + CPU ORT (baseline) | ~740ms | 1.35 | — |
| MIGraphX backbone + CPU ORT | 678ms | 1.47 | +9% |
| MIGraphX backbone + MIG dec/enc | **640ms** | **1.56** | +16% |
| + N=4 memory bank (est., no eval) | ~560ms | ~1.78 | +32% |
| + Flash Attention in backbone (est.) | ~510ms | ~1.96 | +45% |

The current best (1.56 FPS) is limited by `memory_attention` (30%) and
`backbone` (53%). Both bottlenecks are fundamentally compute-limited at 1008px
and require compiler-level or architecture-level changes to reduce further.


---

## NHWC Output Issue (2026-05-09)

MIGraphX compiles the backbone with NHWC (channel-last) GPU layout for efficiency.
When offloading to CPU via `offload_copy=True`, the output has non-standard strides
(e.g. stride={5308416, 1, 36864, 256} for shape={1, 256, 144, 144}, channel stride=1).

**Impact**: `np.ascontiguousarray` (NHWC to NCHW layout transpose) costs:
- 504px: ~11ms for 1.7 MB total FPN (absorbed into backbone timing)
- 1008px: ~89ms for 27.6 MB total FPN (limits gains from direct MIG API switch)

**Attempts to fix via ONNX modification** (`export/force_nchw_output.py`):
- Adding Reshape(flatten) then Reshape(original) at ONNX outputs: MIGraphX eliminates as no-op
- NumPy `moveaxis().copy().transpose()`: 5x faster HWC copy but result still NHWC view, not C-contiguous
- `np.ascontiguousarray` directly: only correct option, costs 7.97ms per FPN set at 504px

**Root cause**: MIGraphX `offload_copy=True` does not insert a `hip::contiguous`
instruction before CPU offload. Fixing requires a MIGraphX compiler change or
a new MIGraphX Python API to force NCHW layout on output.

**Current workaround**: `np.ascontiguousarray` in `MIGraphXBackbone.__call__()`
ensures downstream modules receive correct strides. At 1008px, this 89ms cost
roughly offsets the dec_prop gains from direct MIG API (saves 93ms), giving
minimal net FPS improvement for 1008px propagation FPS from the API switch.
