# MIGraphX Backbone Investigation

**Date**: 2026-05-07  
**Goal**: Replace PyTorch backbone with MIGraphX ONNX to improve inference speed  
**Conclusion**: **PyTorch backbone is irreplaceable** for this model architecture  

---

## Background

The tracker pipeline currently uses:
- **Backbone** (`Sam3TrackerVideoModel.vision_encoder`): PyTorch ROCm FP16 → 139ms at 504px
- **Tracking modules** (memory_attention, mask_decoder, memory_encoder): ONNX (ORT)

The Meta-format `sam3_image_encoder_fp16_iofp16.onnx` achieves **88ms at 512px on MIGraphX**
(1.6× faster than PyTorch), but it is numerically incompatible with the HF tracker modules
(fpn features differ by max_diff=4.89, causing mask coverage to collapse from 29.5% → 0.2%).
The tracker modules only exist in the HF `Sam3TrackerVideoModel` implementation.

---

## Approaches Tried

### 1. ORT session cache discovery

**Finding**: `migraphx_model_cache_dir` provider option enables persistent compilation cache.  
First compile: ~120s. Subsequent loads: ~2s. This resolves the 680s cold-start problem.

```python
MIG = [('MIGraphXExecutionProvider', {
    'migraphx_model_cache_dir': 'mxr_cache',
    'migraphx_fp16_enable': '1',
}), ('CPUExecutionProvider', {})]
```

**Note**: `MIGRAPHX_GPU_HIP_FLAGS='-Wno-error -Wno-lifetime-safety-intra-tu-suggestions'`
must be set before session creation to fix the lifetimebound compiler error.

---

### 2. HF backbone as 3-session split (Part1 + Block31 + FPN)

The original `backbone_split_fp16/` approach splits the backbone across 3 ORT sessions
to work around the MIGraphX Transpose+ConvTranspose layout bug. Each session boundary
requires a CPU↔GPU data transfer.

| Variant | 504px latency | FPS | vs PyTorch |
|---|---|---|---|
| FP32 (3-session) | 1415ms | 0.70 | 10× slower |
| FP16 internal quantize (3-session) | 923ms | 1.08 | 6.6× slower |
| PyTorch FP16 baseline | **139ms** | **7.19** | — |

**Root cause**: 3 CPU↔GPU transfers of intermediate tensors between sessions add
~800ms of overhead. Even with FP16, the session boundary cost dominates.

---

### 3. HF backbone as single ONNX session

Exported `Sam3TrackerVideoModel.vision_encoder` as a single ONNX session
(`backbone_single_fp32.onnx`, 1836MB) using `torch.onnx.export`.

**Accuracy**: max_diff=0.0001 vs PyTorch ✅ (Transpose+ConvTranspose layout bug
fixed by `MIGRAPHX_GPU_HIP_FLAGS`)

**Performance**: 971ms → 1.03 FPS (7× slower than PyTorch)

**Diagnosis**: Graph inspection revealed the HF backbone exports with **9324 nodes**
vs Meta encoder's 2303 nodes. Key extra ops that MIGraphX cannot optimize:

| Op | HF count | Meta | Problem |
|---|---|---|---|
| `Shape` | 521 | ~0 | Dynamic shape extraction at every layer |
| `If` | 2 | 0 | Control flow (conditional branches) |
| `Cast` | many | 0 | Type conversion overhead |
| `Transpose` | many | few | No Transpose+MatMul fusion in MIGraphX |

---

### 4. onnxsim graph simplification

Applied `onnx-simplifier` with fixed 504px input to constant-fold dynamic shape ops:

```
Before: 9324 nodes  (If=2, Shape=521)
After:  2202 nodes  (If=0, Shape=32)   → -76.4%
```

All `If` nodes eliminated, 94% of `Shape` nodes folded. Op types removed:
`Cast`, `ConstantOfShape`, `Identity`, `Mod`, `Constant`.

**Performance after simplification**: **916ms** (vs 971ms before, only -6% improvement)

**Finding**: The 76% node reduction gave almost no speedup. The removed nodes
(`Shape`, `Cast`, `Identity`) are near-zero-cost metadata operations.

---

### 5. Op-level profiling (ORT profiler)

Profiling the simplified backbone (5 inference runs, 9507ms total):

| Op | Time (ms) | Count | Avg/call | % Total |
|---|---|---|---|---|
| **Gemm** | **4320.8** | 912 | 4.74ms | **45.4%** |
| **MatMul** | **1722.2** | 464 | 3.71ms | **18.1%** |
| Add | 546.9 | 1656 | 0.33ms | 5.8% |
| Transpose | 509.3 | 1744 | 0.29ms | 5.4% |
| FusedMatMul | 496.4 | 256 | 1.94ms | 5.2% |
| Split | 294.3 | 720 | 0.41ms | 3.1% |
| Softmax | 285.1 | 256 | 1.11ms | 3.0% |
| ConvTranspose | 159.7 | 24 | **6.65ms** | 1.7% |
| LayerNormalization | 145.1 | 520 | 0.28ms | 1.5% |

**Gemm + MatMul = 63% of total time.** MIGraphX uses generic rocBLAS GEMM kernels
for these, while PyTorch uses TunableOp-autotuned hipBLASLt kernels selected for
the specific matrix shapes used in SAM3's window attention.

---

## Why Meta Encoder is 10× Faster (88ms vs 916ms)

### ORT profiling comparison

**Meta encoder** (88ms/run, 5 runs = 440ms total):

| Op | Total ms | Count | Avg/call | % |
|---|---|---|---|---|
|  | 31.3 | 8 | 3.91ms | **100%** |

MIGraphX **fully fused the entire 2303-node graph into a single GPU kernel**.
One kernel launch, zero intermediate memory transfers, maximum GPU utilization.

**HF backbone simplified** (916ms/run, 5 runs = 9507ms total):

| Op | Total ms | Count | Avg/call | % |
|---|---|---|---|---|
| Gemm | 4320.8 | 912 | 4.74ms | 45.4% |
| MatMul | 1722.2 | 464 | 3.71ms | 18.1% |
| Add | 546.9 | 1656 | 0.33ms | 5.8% |
| Transpose | 509.3 | 1744 | 0.29ms | 5.4% |
| FusedMatMul | 496.4 | 256 | 1.94ms | 5.2% |
| Split | 294.3 | 720 | 0.41ms | 3.1% |
| Softmax | 285.1 | 256 | 1.11ms | 3.0% |
| ... | | | | |

MIGraphX produced **thousands of separate kernel calls** — it could not fuse the HF
backbone's graph, resulting in massive kernel launch and synchronization overhead.

### Root cause: graph fusion eligibility

MIGraphX can fully fuse a graph when:
1. All ops are pure compute (MatMul, Conv, Elementwise) — no control flow, no dynamic shapes
2. Data flow is strictly feed-forward — no branches (), no shape-dependent routing
3. Op sequence maps to known fusion patterns (e.g., Conv+Bias+ReLU, GEMM+Add)

The Meta encoder's clean feed-forward structure (15 op types, no //)
allows complete graph fusion. The HF backbone's  ops around window attention
Q/K/V projections break the fusion boundary — each  forces a separate kernel
and a new data layout in memory.

Both models are ViT transformers — Meta encoder has 156 Transpose ops, HF has 218.
The difference is **not** the presence of Transpose per se.

**Op count comparison (key differences):**

| Op | Meta | HF simplified | Δ | Significance |
|---|---|---|---|---|
| `Split` | **0** | **90** | +90 | Window partition — fusion killer |
| `Neg` | 0 | 64 | +64 | RoPE computation |
| `Gather` | 128 | 20 | −108 | Different indexing strategy |
| `Slice` | 155 | 40 | −115 | Different windowing |
| `Shape` | 0 | 32 | +32 | Remaining dynamic shapes |
| `Transpose` | 156 | 218 | +62 | Both have many |

**Root cause: `Split` ops (90 in HF, 0 in Meta)**

Both models implement window attention, but via different ONNX patterns:

- **Meta**: window partitioning via `Gather`/`Slice` (index-based selection). MIGraphX can fuse these into the surrounding compute.
- **HF**: window partitioning via `Split` (tensor splitting into 90 separate chunks). Each `Split` forces MIGraphX to produce independent output buffers and start a new kernel. 90 Splits = 90 fusion boundaries = hundreds of fragmented kernel calls.

`Split` is a graph fusion "firewall": MIGraphX cannot fuse computation across a Split because each output branch must be materialized separately in memory. This is why despite similar node counts, Meta gets one fused `MGXKernel` while HF produces thousands of separate kernel launches.

**RoPE (`Neg` × 64)** also contributes: HF uses Rotary Position Embedding with 64 negation ops that Meta's implementation avoids.

---

## Hardware Context

The Ryzen AI Max+ 395 is an APU with **memory-bandwidth-limited** compute.
PyTorch ROCm benefits from:
1. **TunableOp**: 8 warmup passes select optimal GEMM kernel for each matrix shape → ~8.7ms saved
2. **Flash Attention variants**: HIP-level optimized kernels for window attention patterns
3. **FP16 throughout**: weights + activations both FP16, no runtime quantization overhead

MIGraphX lacks equivalent auto-tuning for the specific matrix shapes in SAM3's ViT.

---

## Summary

| Approach | 504px latency | FPS | Status |
|---|---|---|---|
| **PyTorch FP16** (current) | **139ms** | **7.19** | ✅ Keep |
| Meta FP16 single session (MIGraphX) | 88ms | 11.28 | ❌ Numerically incompatible |
| HF 3-session FP32 (MIGraphX) | 1415ms | 0.70 | ❌ Transfer overhead |
| HF 3-session FP16 (MIGraphX) | 923ms | 1.08 | ❌ Transfer overhead |
| HF single session FP32→FP16 (MIGraphX) | 971ms | 1.03 | ❌ Slow GEMM kernels |
| HF single session simplified (onnxsim) | 916ms | 1.09 | ❌ Slow GEMM kernels |

---

## Optimization Attempts Summary

### Tools tried

| Tool | Nodes | Change | Key effect | MIGraphX result |
|---|---|---|---|---|
| onnxsim (fixed 504px) | 2202 | -76% | If→0, Shape 521→32 | 916ms (Split still 90) |
| ORT transformer O2 | 1943 | -11% | BiasGelu fusion, Shape→0 | CPU fallback (BiasGelu unsupported by MIGraphX) |

### Community precedent and upstream status

No published work found for AMD/MIGraphX-specific SAM3 window attention optimization.
TensorRT (NVIDIA-only) achieves ~22ms for the SAM3 vision encoder at 512px on RTX 5090 FP16,
as it can fuse window attention patterns natively via its own compiler.

**AMD is aware of this limitation.** A directly relevant open issue exists in the MIGraphX repo:

> **[ROCm/AMDMIGraphX#4256](https://github.com/ROCm/AMDMIGraphX/issues/4256)**  
> "Improve horizontal fusion with multi-used splits" (Aug 22, 2025, open, Perf Improve)  
> 
> "when there are interdependencies in the split groups, we dont fuse it and instead
> fuse after we have run `fuse_pointwise`"  
> Fix: extend `find_splits` to support multiple arguments as long as extra data arguments are constants.

This is exactly the issue causing HF backbone's 90 Split ops to block graph fusion.
Once #4256 is resolved, MIGraphX may be able to fuse the HF window attention pattern,
potentially bringing the HF backbone performance close to the Meta encoder's 88ms.

Other relevant channels:
- MIGraphX GitHub Discussions: [ROCm/AMDMIGraphX/discussions](https://github.com/ROCm/AMDMIGraphX/discussions)
- ROCm Documentation: [rocm.docs.amd.com/projects/AMDMIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)

### Path to fix

Resolving the 90 `Split` ops requires a **custom ONNX graph rewrite pass** that replaces the HF window partition pattern:

```
Reshape → Split(90) → [attention per window] → Concat → Reshape
```

with a Gather/Scatter-based pattern matching Meta's implementation:

```
Reshape → Gather(indices) → [attention per window] → ScatterND → Reshape
```

This is a non-trivial engineering effort targeting the HF `Sam3TrackerVideoModel`
window attention export specifically. No generic tool automates this rewrite.

**To unlock MIGraphX for HF backbone would require:**
1. Custom ONNX graph pass replacing Split-based window partition with Gather/Scatter
2. Or rewrite the HF `vision_encoder.forward()` to avoid Split ops (requires modifying transformers source)
3. Or wait for MIGraphX to support Split-based graph fusion

All options are beyond the current project scope.

---

## Key Technical Artifacts

- `export/export_backbone.py`: HF backbone 3-session ONNX export (256-ch FPN, compatible with mask decoder)
- `tracker/tracker_mxr.py`: ORT MIGraphX tracker reference implementation
- `onnx_files/backbone_single_fp32.onnx`: single-session HF backbone (1836MB)
- `onnx_files/backbone_single_simplified.onnx`: onnxsim-simplified (1817MB, 2202 nodes)
