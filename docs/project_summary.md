# SAM3 Video Tracking ONNX Pipeline — Project Summary

**Hardware**: AMD Ryzen AI Max+ 395 (gfx1151 / Radeon 8060S), 128GB unified memory (APU, UMA=64GB)  
**Target**: Mask-level video tracking at ≥5 Hz

## Software Environment

| Software | Version | Source |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) | — |
| Kernel | 6.18.6-061806-generic | — |
| ROCm HIP runtime | 7.13.0a20260411 | AMD nightly pip (`rocm-sdk-core`) |
| PyTorch | 2.12.0a0+rocm7.13.0a20260411 | AMD nightly pip |
| MIGraphX | 2.15.0 | System APT (`repo.radeon.com/rocm/apt/7.2`) |
| ONNX Runtime (MIGraphX EP) | 1.24.2 | [Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build) |
| ONNX | 1.21.0 | pip |
| Python | 3.12.13 | conda |
| NumPy | 1.26.4 | pip |
| conda | 26.1.1 | miniforge |

> **Dual ROCm stack**: AMD maintains two parallel release tracks. The nightly pip
> wheels (ROCm 7.13) provide gfx1151 PyTorch support but do not include MIGraphX.
> The stable APT release (ROCm 7.2) provides MIGraphX but does not support gfx1151
> for PyTorch. Both are required concurrently — they coexist without conflict because
> PyTorch loads its ROCm runtime libraries explicitly from the conda env's
> `_rocm_sdk_core` paths (via `rocm_sdk.initialize_process()`), while
> `onnxruntime-migraphx` links against the system APT `libmigraphx_c.so.3`.
> This split is expected to resolve when AMD unifies the release tracks (planned
> mid-2026).

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
| **backbone** | 561ms (PyTorch FP16, 1008px) | **139ms** | Resolution reduction to 504px (area ratio 0.25) + TunableOp |
| **memory_attention** | 157ms (CPU, dynamic axes) | **16ms** | Fixed N=7 ONNX → MIGraphX (3.3×); dynamic_axes caused dangling reference compiler bug, bypassed with fixed-size export; requires `MIGRAPHX_GPU_HIP_FLAGS` to suppress lifetimebound compiler error |
| **mask_decoder_propagate** | 54ms | **7ms** | ORT intra_op_num_threads=8 (4×) |
| **memory_encoder** | 45ms | **10ms** | ORT intra_op_num_threads=8 (5×) |
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
| sam3_image_encoder.onnx (Meta format) | ✅ Works on MIGraphX — but **not used in the tracker** | This is the encoder exported from Meta's original codebase. It runs correctly on MIGraphX and was used for single-frame detection benchmarking. It cannot replace the tracker backbone because its FPN outputs are numerically incompatible with `Sam3TrackerVideoModel` (see Key Technical Finding #1). |
| memory_attention (dynamic axes) | ❌ | dynamic_axes triggers "Dangling reference in module main" compiler bug |
| **memory_attention (fixed N=7)** | **✅** | Fixed-size ONNX bypasses dangling-ref bug; requires `MIGRAPHX_GPU_HIP_FLAGS=-Wno-error -Wno-lifetime-safety-intra-tu-suggestions` to pass newer clang `-Werror` check; 3.3× speedup |
| memory_encoder | ❌ CPU only | ConvTranspose layout bug |
| mask_decoder_init / propagate | ❌ CPU only | simplify_reshapes error / Segfault |
| Backbone ONNX FP16 | ❌ (impractical) | Compiles on MIGraphX with `MIGRAPHX_GPU_HIP_FLAGS` fix, but: (1) JIT cache does not persist across Python processes — 680s cold-start every run; (2) existing exports are fixed at 1008px, 504px version would need re-export. PyTorch ROCm GPU FP16 is the practical choice. |

**Key hardware constraint**: gfx1151 is an APU with unified memory. The backbone is **memory-bandwidth-limited**, not compute-limited. As a result, kernel-level optimizations (torch.compile, Flash Attention, SDPA) provide no benefit, while resolution reduction is highly effective.

---

## Resolution Comparison

| Metric | 1008px | 504px |
|---|---|---|
| **DAVIS 2017 val Mean J** | **85.8%** | 81.1% |
| **SG val Mean J** (50 seqs, seed=42) | 44.8% | 39.6% |
| Propagation FPS | 1.35 | **5.72** |
| Init frame FPS | 1.68 | **6.33** |
| Backbone latency | 528ms | 142ms |
| Use case | High-quality offline | **Real-time tracking (≥5 Hz)** |

Accuracy trade-off: halving resolution causes approximately 4–5 pp drop in J (DAVIS −4.7 pp, SG −5.2 pp).

---

## Benchmark Results

### Pipeline A: Single-frame (box/point → mask, no tracking)

| Stage | 504px | 1008px |
|---|---:|---:|
| backbone [PyTorch ROCm FP16] | 139.9 ± 1.8 ms | 525.5 ± 1.4 ms |
| mask_decoder_init [ONNX CPU] | 6.5 ± 0.5 ms | 23.4 ± 0.4 ms |
| **Total → FPS** | **158 ms → 6.33 FPS** | **584 ms → 1.71 FPS** |

### Pipeline B: Propagation per-frame (video tracking)

| Stage | 504px |
|---|---:|
| backbone [PyTorch ROCm FP16] | 138.7 ± 0.7 ms |
| memory_attention [MIGraphX] | 16.1 ± 1.0 ms |
| mask_decoder_propagate [ONNX CPU] | 6.9 ± 0.6 ms |
| memory_encoder [ONNX CPU] | 11.2 ± 3.2 ms |
| **Total → FPS** | **175 ms → 5.72 FPS** |

*n=30 timed runs, GPU exclusive (no concurrent workloads), after TunableOp warmup.*
*Run `python eval/bench_pipeline.py --checkpoint model/sam3 --onnx-dir onnx_files` to reproduce.*

### DAVIS 2017 val (30 sequences, standard VOS benchmark)

| Configuration | Mean J | Propagation FPS |
|---|---|---|
| **Ours — 1008px** | **85.8%** | 1.35 |
| **Ours — 504px** | **81.1%** | 5.72 |
| SAM2 official (J&F, reference) | ~90.7% | — |

*Verified with MIGraphX enabled after `MIGRAPHX_GPU_HIP_FLAGS` fix; accuracy unchanged from pre-fix baseline.*

### Hardware comparison — SAM3 video propagation FPS

SAM3's standard target resolution is **1008px**. The official paper reports 30 ms/image on H200 with 100+ objects, and "near real-time for ∼5 concurrent objects" in video [[SAM3, arXiv 2511.16719](https://arxiv.org/abs/2511.16719)]. The table below compares propagation FPS across hardware at that resolution, followed by our 504px result for context.

| Hardware | Resolution | FPS | Notes | Source |
|---|---|---|---|---|
| NVIDIA H200 (data-centre) | ≈1008px | ~5–6 | PyTorch, single object; paper states "near real-time for ∼5 concurrent objects" | [arXiv 2511.16719](https://arxiv.org/abs/2511.16719); [community report](https://github.com/facebookresearch/sam3/issues/425) |
| NVIDIA RTX 5090 (consumer flagship) | 1008px | ~5 | PyTorch | [Medium — TensorRT guide](https://medium.com/@kynkynkyn/how-to-accelerate-sam3-with-tensorrt-for-real-time-inference-d014ef504383) |
| NVIDIA RTX 5090 | 1008px | 30+ | TensorRT + ByteTrack optimised | [Medium — TensorRT guide](https://medium.com/@kynkynkyn/how-to-accelerate-sam3-with-tensorrt-for-real-time-inference-d014ef504383) |
| **Ours — AMD Ryzen AI Max+ 395 (APU)** | **1008px** | **1.35** | PyTorch backbone + ONNX (MIGraphX) | This work |
| **Ours — AMD Ryzen AI Max+ 395 (APU)** | **504px** | **5.72** | Half-resolution trade-off | This work |

**Fair comparison at 1008px**: our APU achieves 1.35 FPS vs. 5–6 FPS on an H200, a ~4× gap. The 5.72 FPS figure is at 504px (half the standard resolution) and is therefore not a like-for-like comparison against the H200 number.

**Context**: the 504px operating point was a deliberate choice to reach a practical frame rate on an edge APU. The accuracy cost is modest (DAVIS J: 85.8% → 81.1%). Reaching comparable 1008px performance on AMD APU hardware would require either a faster backbone (e.g., MIGraphX backbone ONNX once JIT cache issues are resolved) or model distillation.

### Smartglass SG val (50 sequences, seed=42, egocentric tracking)

> **Note**: The numbers below use J (IoU) averaged over a random 50-sequence subset,
> which is an internal proxy metric only. The official SA-Co/VEval evaluation protocol
> for the SG dataset uses **cgF1** and **pHOTA** (Video Phrase HOTA), computed over
> all 1686 annotations across 334 videos. cgF1 requires full-dataset coverage because
> it includes an image-level MCC term (IL_MCC) that penalises false positives on
> negative video–noun-phrase pairs — a metric that is not meaningful on a 50-seq subset.
> **Official cgF1/pHOTA evaluation is pending** and will be added in a future update.

| Configuration | Mean J (50 seqs, proxy) | Propagation FPS |
|---|---|---|
| **Ours — 1008px** | **44.8%** | 1.13 |
| **Ours — 504px** | **39.6%** | 3.84 |

*SG FPS measured during eval (includes GT mask loading); pure inference FPS matches Pipeline B above.*

The SG dataset covers first-person (smartglass) viewpoints and is significantly harder than DAVIS, containing many small or fast-moving targets (hands, wires, smartphones).

---

## Key Technical Findings

1. **Two SAM3 implementations exist and they are not interchangeable**: SAM3 has two independently maintained codebases — Meta's original and the HuggingFace `transformers` port. Although they nominally share the same weights, their image backbone produces numerically different FPN feature maps (max absolute difference = 4.89 on identical inputs). The mask decoder was trained against the HuggingFace variant's outputs; feeding it Meta-format features causes mask coverage to collapse from 29.5% to 0.2% — effectively no segmentation. Consequence: `sam3_image_encoder.onnx` (exported from Meta's code, already running on MIGraphX) **cannot be dropped in as a replacement** for the tracker backbone. The backbone must be run as PyTorch via `Sam3TrackerVideoModel.vision_encoder`. This is the single most important architectural constraint of the project — it is why the backbone remains PyTorch rather than ONNX/MIGraphX.

2. **memory_attention must use fixed-size ONNX**: Dynamic axes trigger a MIGraphX compiler bug ("Dangling reference in module main"). Exporting with fixed N=7×HW shape bypasses this and enables 3.3× speedup on MIGraphX.

3. **MIGraphX JIT compiler flag** (`MIGRAPHX_GPU_HIP_FLAGS`): Newer versions of the comgr/clang compiler inside ROCm treat the C++ `[[clang::lifetimebound]]` suggestion as `-Werror`, aborting GPU kernel compilation for `memory_attention_fixed_N7.onnx`. Setting `MIGRAPHX_GPU_HIP_FLAGS=-Wno-error -Wno-lifetime-safety-intra-tu-suggestions` suppresses this. Without it, memory_attention silently falls back to CPU (72ms instead of 16ms, propagation drops from 5.74 to 3.82 FPS). This flag is now set automatically in `tracker.py` via `os.environ.setdefault()`.

4. **ORT thread tuning**: CPU ONNX sessions default to single-threaded. Setting `intra_op_num_threads=8` reduces combined decoder + encoder latency from ~100ms to ~16ms.

5. **TunableOp (AMD)**: 8 warmup passes trigger per-operation GEMM kernel autotuning. Subsequent calls use the optimal kernel, reducing backbone latency by ~8.7ms.

6. **propagate_frame mask bug (fixed)**: `binary_mask[0]` at 504px incorrectly extracted the first row (1D) instead of the full 2D mask, causing all propagation frames to output 0% mask coverage. Fixed: `masks.squeeze() > 0`.

7. **UMA BIOS setting (64 GB out of 128 GB)**: The Ryzen AI Max+ 395 is an APU with 128 GB total LPDDR5X shared between CPU and GPU. The BIOS "UMA Frame Buffer Size" carves out a **coarse-grained (non-coherent)** memory pool for the GPU at boot. Memory inside this pool is fast for GPU access; memory outside it is **fine-grained (cache-coherent)**, which is 2–4× slower due to CPU cache coherency overhead. Setting UMA=64 GB gives the GPU a 64 GB fast coarse-grained pool while leaving 64 GB for the OS — the optimal balance. Setting UMA=128 GB (all RAM) starves the OS of memory, causing instability and paradoxically worse GPU performance because OS pressure forces operations into the slow fine-grained region. Measurable impact in early benchmarking: +2.4% on the encoder+decoder chain path, up to +20.7% on the fused FP16 path. The tracker backbone is memory-bandwidth-limited, so this setting matters. The current machine is configured at UMA=64 GB (`rocm-smi` reports 64 GB VRAM).

---

## Current Best Configuration

The pipeline is packaged as the standalone `sam3-tracker-rocm` project (no DART fork required; uses HuggingFace `transformers ≥ 5.8.0`).

```bash
# Export ONNX modules once
python export/export_tracker_modules.py --imgsz 504 --output-dir onnx_files

# Run demo
python demo.py --checkpoint model/sam3 --onnx-dir onnx_files \
               --image assets/demo.jpg --box 85,281,1710,850

# Evaluate on DAVIS 2017 val
python eval/eval_davis.py --checkpoint model/sam3 --onnx-dir onnx_files \
                          --davis dataset/DAVIS --imgsz 504

# Benchmark pipeline A vs B
python eval/bench_pipeline.py --checkpoint model/sam3 --onnx-dir onnx_files
```

Required environment variables (set in `~/.bashrc` or run script):
```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
```

`MIGRAPHX_SKIP_BENCHMARKING=1` and `MIGRAPHX_GPU_HIP_FLAGS` are set automatically by `tracker.py`.

- **FPS**: 5.72 (propagation), 6.33 (single-frame init)
- **DAVIS J**: 81.1% at 504px / 85.8% at 1008px
- **Prompt**: box on frame 0; subsequent frames use pure memory propagation

---

## Next Steps

### 1. Full backbone ONNX → MIGraphX ✗ INVESTIGATED — PyTorch wins

**Status**: Fully investigated. Conclusion: PyTorch backbone is irreplaceable.  
**Full details**: [`docs/migraphx_backbone_investigation.md`](migraphx_backbone_investigation.md)

**Summary**: HF backbone exports 9324 nodes with `Transpose`/`Shape`/`If` ops that MIGraphX
cannot efficiently handle. After onnxsim simplification (9324 → 2202 nodes, -76%), op profiling
showed **Gemm+MatMul = 63% of runtime** — MIGraphX uses untuned generic GEMM kernels vs
PyTorch's TunableOp-autotuned hipBLASLt. Best result: 916ms vs PyTorch 139ms (6.6× slower).

The Meta encoder (2303 nodes, clean feed-forward) runs at 88ms on MIGraphX FP16 but is
numerically incompatible with the HF tracker modules (max_diff=4.89, Finding #1).

---

### 2. Official SG evaluation (cgF1 / pHOTA)

Current SG numbers (J/IoU, 50-seq random subset) are a proxy metric only. The official SA-Co/VEval protocol requires:
- Run tracker on all **1686 annotations** across 334 videos (~3–4 hours at 5 FPS)
- Save predictions in YT-VIS JSON format
- Run DART's `VEvalEvaluator` for **cgF1** and **pHOTA** (Video Phrase HOTA)

The DART evaluator and all dependencies (`iopath`, `sam3.eval.*`) are already available on this machine. Work needed: add `--save-preds` flag to `eval_saco_sg.py` to output YT-VIS format predictions, then call `saco_veval_eval.py`.

---

### 3. memory_encoder and mask_decoder on MIGraphX

Currently CPU-only due to two separate MIGraphX bugs:
- **memory_encoder**: ConvTranspose tensor layout bug (similar to the FPN Transpose+ConvTranspose bug that was worked around in the backbone split)
- **mask_decoder_init / propagate**: `simplify_reshapes` error / segfault

The backbone split approach (inserting a numpy `BHWD→BCHW` permute between ONNX sessions to avoid the layout bug) may be applicable to memory_encoder. If resolved, combined ONNX CPU time (~18ms) would move to MIGraphX.

---

### 4. True simultaneous multi-object tracking

The current implementation tracks multiple objects **sequentially** — the full sequence is run once per object, so the backbone is called N×frames times for N objects. The official SAM3 architecture is designed for **simultaneous** multi-object tracking: the backbone runs **once per frame** (shared across all objects), and each object has its own independent memory bank and mask decoder call.

The fix is straightforward within the existing Python host (no ONNX re-export needed):

```python
# Current (sequential — backbone runs N×frames times):
for obj_id in obj_ids:
    tracker.reset()
    for frame in frames:
        mask = tracker.init_frame(img, box)  # backbone inside

# Proposed (shared backbone — backbone runs frames times):
for frame in frames:
    fpn_0, fpn_1, fpn_2, pos = tracker._backbone(img)   # once
    for obj_id in obj_ids:
        mask = tracker._decode(fpn_0, fpn_1, fpn_2, obj_memory[obj_id])
        obj_memory[obj_id].update(fpn_2, mask)
```

This would reduce total processing time for N objects from O(N×frames) to O(frames + N×decoder_cost). At 504px the decoder is ~7ms per object vs ~140ms for the backbone — so for 3 objects the speedup is roughly 3× vs current sequential approach.

---

### 5. SAM 3.1 upgrade evaluation

Meta released SAM 3.1 with **Object Multiplex** — processes multiple objects in a single forward pass, up to **7× faster** than SAM 3 for multi-object tracking with no accuracy loss. Worth evaluating whether the same ONNX export pipeline applies to SAM 3.1's `Sam3TrackerVideoModel` variant, and measuring the propagation FPS improvement.

---

---

## MIGraphX Backbone Optimization (2026-05-08)

Full investigation and results: [](mig_inv.md) and
[](1008px_perf_analysis.md).

### Optimization journey

| Step | Change | 504px FPS | 1008px FPS |
|---|---|---|---|
| Baseline (PyTorch backbone + CPU ORT) | — | 5.72 | 1.35 |
| MIGraphX 2.16.0 backbone |  multi-arg patch (AMDMIGraphX#4256) + kernel autotuning | 7.10 | 1.47 |
| dec_prop + mem_enc → MIGraphX GPU | Switch CPU ORT → MIGraphX ORT EP (FP32) | 7.71 | 1.56 |
| FP16 for mem_enc + mem_attn (504px) | ORT  | 8.32 | 1.56 |
| FP16 mem_attn via direct migraphx API |  class + autotuned  cache | **8.58** | **1.97** |

**vs PyTorch baseline: +50% at 504px, +46% at 1008px**

### Final per-module timing

| Module | 504px | 1008px | Backend |
|---|---|---|---|
| backbone | 92ms | 342ms | MIGraphX 2.16.0 (FP16 internal, ) |
| memory_attention | 6.5ms | 56ms | MIGraphX direct API FP16 () |
| dec_propagate | 14ms | 98ms | MIGraphX ORT EP FP32 |
| memory_encoder | 2ms | 7ms | MIGraphX ORT EP FP16 |
| **Total → FPS** | **117ms → 8.58** | **507ms → 1.97** | |

### Key technical findings

- ** patch**: the 90  ops in the HF window-attention ONNX blocked
  MIGraphX graph fusion (stock = 916ms). Extending  to handle multi-arg ops
  unlocks fusion, and kernel autotuning brings backbone to 88–94ms (1.5× faster than PyTorch).
- **Kernel autotuning critical for memory_attention**: 
  gives 758ms; autotuning gives 56ms at 1008px (14× difference).
- **ORT MIGraphX EP falls back to CPU at 1008px**: FP16 memory_attention via ORT EP
  silently switches to CPU when backbone is in GPU memory. Fixed by using the direct
   Python API ( class) instead.
- **dec_propagate FP16 corrupts results**: ConvTranspose upsampling is numerically
  sensitive in FP16 (max_diff=15.3). Kept FP32.
- **Startup time**: 2+ minutes (ORT recompile per run) → ~5s (load from ).
  Run  once to populate.

### Remaining opportunities

| Opportunity | Potential gain | Effort |
|---|---|---|
| Flash Attention in backbone (fuse Q@K + Softmax) | ~67ms at 1008px backbone | New MIGraphX fusion pass |
| Memory bank N: 7→5 | ~42ms at 1008px mem_attn | Export fixed_N5 ONNX |
| dec_init on MIGraphX GPU | ~70ms first-frame at 1008px | Test MIGraphX EP |
| RoPE-GEMM fusion in backbone | ~12ms | New MIGraphX graph rewrite |

### Summary table

| Next Step | Effort | Expected FPS gain | Status |
|---|---|---|---|
| MIGraphX backbone (find_splits patch + autotuning) | Done | +50% at 504px, +46% at 1008px (see above) | **Done** |
| MIGraphX JIT cache | Resolved | `migraphx_model_cache_dir` ORT option works | **Closed** |
| Shared backbone for multi-object tracking | Low (Python host only) | ~N× for N objects | Pending |
| Official SG cgF1/pHOTA eval | Medium (~4h runtime) | — (accuracy metric) | Pending |
| memory_encoder on MIGraphX | Done | ~6ms saved (FP16 ORT EP) | **Done** |
| dec_propagate on MIGraphX GPU | Done | ~15ms saved (FP32 ORT EP) | **Done** |
| SAM 3.1 upgrade | Medium | Up to 7× (multi-object) | Pending |

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
