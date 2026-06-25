# SAM3 ViT Backbone NPU Offload — Technical Overview

**Platform**: AMD Strix Halo (Ryzen AI Max 395) — XDNA2 NPU (6×8 AIE2p cores, 50 TOPS)  
**Target audience**: AIE colleagues who want to understand what we did and how  
**Date**: 2026-06-25

---

## Overview

We offload the SAM3 ViT-L backbone (32 transformer blocks, 1296 tokens @ 504px) to
the XDNA2 NPU using custom MLIR-AIE IRON kernels in **BF16 precision**.
The implementation uses a GPU+NPU hybrid: BF16 matrix operations (QKV, O-proj, FFN)
run on NPU xclbins while RoPE, GELU, residuals, and window partitioning run on
GPU/CPU. No INT8 quantization — weights and activations stay in BF16 throughout.

> Note: An INT8 variant exists (`bh_npu_backbone`, cos=0.932) but is not the
> primary pipeline. BF16 gives cos=0.989 at similar latency and is the recommended path.

**Bottom line**: ~36.5 W average for the NPU backbone subprocess alone (measured standalone).
In the full streaming demo (NPU + concurrent GPU tracking), total system power is
39–81 W depending on prompt count, vs 91 W on MIGraphX GPU.
Accuracy: **cos = 0.989** vs PyTorch float32 (BF16, no quantization error). Used as a background async re-detector in a streaming
pipeline where GPU tracking runs at 8–10 FPS continuously.

---

## 1. NPU vs GPU Split Per ViT Block

Each of the 32 ViT blocks is split as follows.

### On NPU (MLIR-AIE IRON xclbins)

| Operation | Precision | Notes |
|---|---|---|
| LayerNorm 1 & 2 | BF16 | Single xclbin kernel, 64 dispatches/frame |
| QKV linear projection | BF16×BF16→BF16 | Full precision, no quantization |
| QKᵀ (attention score) | BF16 | Window: 64 heads×576 tokens; Global: 16 heads×1344 |
| Softmax | BF16 | Separate xclbin per sequence length |
| PV (attention weighted sum) | BF16 | Output: float32 |
| Output projection (O-proj) | BF16×BF16→BF16 | — |
| FFN1 (two halves in parallel) | BF16×BF16→BF16 | Split across cores for parallelism |
| FFN2 | BF16×BF16→BF16 | — |

**Total NPU dispatches**: 288 per frame (9 kernels × 32 blocks).  
NPU wall time: ~990 ms (dispatch overhead dominates — ~3.4 ms per dispatch on XRT).

### On GPU (HIP kernels, `libgpu_kernels.so`)

| Operation | Precision | Notes |
|---|---|---|
| Head reshape (window/global) | FP32 | `win_part` / `win_unpart` |
| RoPE (rotary position embedding) | FP32 | Applied to Q and K before NPU attention |
| FP32→BF16 conversion | — | At NPU boundary before D2H to XRT BO |
| GELU + bias_add | FP32 | In-place, after FFN1 |
| Residual additions | FP32 | 2 per block |

### On CPU (host, OpenMP)

- Window partitioning / unpartitioning (layout rearrangement for windowed attention)
- Head unshuffle after attention (reorder from [G, Sp, d] to [S, C])
- Per-block residual accumulation (can be fused into GPU later)

### Python GPU side (PyTorch)

- `backbone.embeddings(pixel_values)` — patch embedding + tiled position encoding
- `backbone.layer_norm(tokens_spatial)` — pre-norm before ViT layers (applied before passing to the binary)
- `neck(features)` — FPN decoder after all 32 blocks

---

## 2. Kernel Architecture (MLIR-AIE IRON)

### Xclbin organization

```
npu_iron/sam3_attn/
  layernorm/S1296/final.xclbin     # LN for windowed tokens (padded to 1344)
  proj_mc/qkvproj_w/final.xclbin  # QKV BF16 matmul, window heads
  proj_mc/qkvproj_g/final.xclbin  # QKV BF16 matmul, global heads
  proj_mc/oproj_w/final.xclbin    # O-proj BF16, window
  proj_mc/oproj_g/final.xclbin    # O-proj BF16, global
  ffn_mc/ffn1_half/final.xclbin  # FFN1 BF16 multi-core
  ffn_mc/ffn2/final.xclbin       # FFN2 BF16 multi-core
  qkt_S576/final.xclbin            # QKᵀ, window (S=576 per window)
  sm_S576/final.xclbin             # Softmax, window
  qkt_S1296/final.xclbin           # QKᵀ, global (S=1296, padded to 1344)
  sm_S1296/final.xclbin            # Softmax, global
  ffn_mc/ffn1_half/final.xclbin   # FFN1 first half (multi-core)
  ffn_mc/ffn2/final.xclbin        # FFN2 (multi-core)
```

### Attention split (window vs global)

SAM3 ViT alternates between windowed attention (blocks 0–23, window size 6 → 64 windows of
576 tokens, 64 heads each) and global attention (blocks 24–31, full 1296-token sequence,
16 heads). We use separate xclbins for each to match the expected sequence lengths.

### Multi-core FFN (FFN speedup 30×)

`ffn_mc/` uses all 8 columns of the 6×8 AIE array with `whole_array` allocation.
FFN1 is split into two halves processed in parallel across columns, giving ~30× speedup
over single-core. FFN2 uses 30 cores similarly.

### Weight format

Weights are stored as FP32 raw binary in `/tmp/cbb/` (exported by `export_weights_bf16.py`
from the SAM3 model checkpoint) and converted to BF16 at load time via AVX-512
`_mm512_cvtneps_pbh`. No per-tensor or per-token quantization.

---

## 3. GPU+NPU Coexistence on Strix Halo UMA

### The same-process conflict

HIP (via ROCm) and XRT (via XDNA driver) cannot both initialize in the same Python process:
HIP's static constructors reconfigure the IOMMU/MMU, which corrupts XRT's DMA mappings
and causes NPU BO reads to return garbage (constant ~5.6×10¹⁰).

### Solution: subprocess IPC

The NPU binary (`bh_npu_backbone`) runs as a **child subprocess** with no CUDA context.
The Python main process (which has CUDA for GPU tracking) communicates via temp files:

```
Python (CUDA) ──[tokens.bin]──► subprocess (no CUDA) ──[features.bin]──► Python
                                    └── XRT + HIP (dlopen)
```

The subprocess loads `libgpu_kernels.so` via `dlopen` with delayed HIP initialization
(static constructors run at `dlopen` time, not at `main()`, ensuring XRT is initialized
first inside the subprocess).

### Data transfer pattern

Despite UMA (shared physical memory), explicit synchronization is required:

```cpp
// GPU → NPU BO: must go through CPU vector (not direct D2H to BO mapped ptr)
gpu_d2h(cpu_vec.data(), gpu_ptr, size);   // GPU writes to fresh CPU allocation
gpu_sync();
bo.write(cpu_vec.data());                 // updates CPU-side cache
bo.sync(TO_DEVICE);                       // flushes valid cache to DRAM

// NPU → CPU: direct map read is correct after sync
bo.sync(FROM_DEVICE);
float* out = bo.map<float*>();
```

The key insight: `hipMemcpy` D2H bypasses CPU cache (writes directly to DRAM), so a
subsequent `bo.sync(TO_DEVICE)` would flush stale cache data, overwriting the GPU result.
The CPU vector intermediary ensures the cache is populated correctly before sync.

---

## 4. Performance

### Timing breakdown (per frame, ~2350 ms total)

| Stage | Time | % |
|---|---|---|
| NPU dispatch (288 total) | ~990 ms | 42% |
| GPU kernels (BF16 conv, reshape, RoPE, GELU) | ~200 ms | 9% |
| XRT BO write + sync | ~200 ms | 8% |
| CPU host (win_part, residuals) | ~160 ms | 7% |
| Python subprocess overhead | ~700 ms | 30% |

**Dispatch overhead dominates**: 288 × 3.4 ms/dispatch ≈ 979 ms, leaving only ~1 ms
actual compute time per dispatch for many kernels. This is the fundamental bottleneck on
current XRT — not compute throughput.

### Power

| Implementation | Latency | Avg Power | Peak Power | Energy/frame |
|---|---|---|---|---|
| MIGraphX FP16 (GPU only) | 70 ms | 91 W | 118 W | 6.4 J |
| **BF16 GPU+NPU (this work)** | **2290 ms** | **36.5 W*** | **38 W** | **83 J** |

Power sensor: `hwmon5/power1_input` (GFX die, covers both GPU and NPU).

The 60% power reduction comes from shifting matrix operations from GPU SIMD (high
clock frequency, high power) to NPU systolic array (lower frequency, highly parallel).

### Accuracy

Cosine similarity vs PyTorch float32 reference: **cos = 0.989**

BF16 accumulated rounding error across 32 blocks is small (~0.5% loss). No
quantization artifacts. Remaining gap from 1.0 is inherent BF16 precision (7 mantissa
bits vs 23 for float32).

> INT8 variant (future improvement): cos=0.932, 2.35s — available as
> `bh_npu_backbone` binary but not the primary pipeline.

---

## 5. Integration: Async NPU Streaming Pipeline

The backbone runs too slowly (2.35 s) for real-time use. We integrate it as an
**asynchronous background re-detector** in `demo_npu_parallel.py`:

```
Main thread (GPU):                      NPU thread (subprocess):
─────────────────                       ──────────────────────────
Frame 0: SAM3 detect (MIG, ~2.5s) ──► init box trackers
Frame 1: MIGraphX backbone + ORT       │  NPU backbone (2.35s)
         mem_attn → propagate           │  + MIG DETR (~200ms)
Frame 2: propagate  (~120ms/frame) ←──┘ put result queue
Frame 3: propagate                      starts next frame...
...
Frame N: check result queue ──────────► reinit trackers from NPU detections
         propagate (uninterrupted)
```

**GPU tracker throughput**: 8–10 FPS (thing-class), 4–7 FPS (multi-prompt stuff-class)  
**NPU re-detection interval**: ~3.5 s (NPU backbone + MIG DETR)  
**NPU power during re-detection**: ~36.5 W for backbone subprocess alone (BF16);
  full system (NPU + concurrent GPU tracking) measured at 39–81 W depending on prompts

---

## 6. Known Limitations

### Dispatch overhead
288 dispatches × 3.4 ms = 979 ms overhead, regardless of compute. Reducing dispatch
count via kernel fusion would be the single highest-impact optimization. The S×S
attention score matrix passing through DRAM between QKᵀ and Softmax dispatches is
the key fusion challenge — cores cannot stream S×S back to L2 without a DRAM roundtrip
at current grid sizes, causing the Phase1→Phase2 deadlock we investigated.

### Flash Attention dead end
We attempted Flash Attention (online softmax, tile Q×K/V without materializing S×S).
Single-core: 350 ms/head vs 2.7 ms with 3-dispatch. Root cause: NPU bottleneck is
DMA event count (not bandwidth). Flash requires more DMA events per head, not fewer.
**Do not retry this direction.**

### Stuff-class tracking gaps
NPU re-detection at 3.5 s intervals is too infrequent for fast robot motion with
stuff-class prompts (floor, wall). The SAM2-style box tracker loses amorphous regions
within 5–10 frames. The Hybrid MIG pipeline (1 s keyframe interval, GPU DETR) is
better suited for stuff-class.

---

## 7. File Map

| File | Purpose |
|---|---|
| `tracker/npu_backbone_service.py` | Python subprocess wrapper (`NPUIRONVisionEncoder`) |
| `demo_npu_parallel.py` | Full streaming demo (NPU async + GPU tracking) |
| `eval/benchmarks/npu_iron/bh_gpu_v2_fixed_20260619.cpp` | C++ NPU+GPU hybrid binary source |
| `results/npu_iron/gpu_kernels_full_20260619.hip` | HIP GPU kernels (shared lib) |
| `npu_iron/sam3_attn/` | Pre-compiled IRON xclbins |
| `/tmp/cbb/` | Quantized weight files (LN, QKV, FFN scales + biases) |

---

## 8. Reproducing

```bash
# 1. Ensure bh_npu_backbone binary is compiled and at /tmp/bh_npu_backbone
#    (requires: XRT, /tmp/cbb/ weights, npu_iron/sam3_attn/ xclbins)

# 2. Run streaming demo
source /opt/xilinx/xrt/setup.sh
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:...
conda activate rocm7p13-sam3

python demo_npu_parallel.py \
    --video assets/blackswan.mp4 --text swan \
    --onnx-dir onnx_files_504 \
    --output results/demo_npu.mp4
```

The `bh_npu_backbone` binary source is in `eval/benchmarks/npu_iron/` and requires
the pre-compiled xclbins from the IRON fork: `github.com/harrysocool/mlir-aie`
branch `sam3-rope-attention`.
