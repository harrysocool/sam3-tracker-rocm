# Backbone Optimization Research — gfx1151 / MIGraphX 2.16

**Context**: SAM3 ViT-H backbone, 93ms/frame @504px (already FP16, autotuned, onnxsimmed).
Copy overhead only 1.5ms — kernel is the real bottleneck. Researched 2026-05-13.

---

## Actionable (ranked)

### 1. `MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention"` — HIGH PRIORITY
Routes attention subgraphs to rocMLIR-compiled kernels.
Default only on gfx942+ (MI300); must be forced on gfx11xx.
Validated by ComfyUI community as key RDNA3 attention performance lever.
```bash
MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention" \
    python export/backbone/compile_backbone_mxr.py --imgsz 504 --backbone-source detector --force
```

### 2. hipBLASLt silent fallback on gfx1151
Known bug ROCm #5643: hipBLASLt may silently degrade to hipBLAS (no tensor cores) on gfx1151.
FP16 GEMMs without tensor cores = large performance loss.
```bash
# Diagnose:
MIGRAPHX_TRACE_GEMM=1 python ... 2>&1 | grep -i 'hipblas\|fallback'
# Force hipBLASLt explicitly:
MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt python export/backbone/compile_backbone_mxr.py ...
```

### 3. ORT attention fusion → MIGraphX compile
MIGraphX 2.13+ supports `com.microsoft.Attention` contrib op natively.
If the exploded Q/K/V MatMul+Softmax pattern can be fused first via ORT optimizer,
MIGraphX may pick a faster kernel path.
```bash
python -m onnxruntime.transformers.optimizer \
    --input onnx_files_504/backbone_detector/single_simplified.onnx \
    --output onnx_files_504/backbone_detector/single_attn_fused.onnx \
    --model_type vit --num_heads 16 --hidden_size 1280
# Then compile single_attn_fused.onnx instead of single_simplified.onnx
```

### 4. `MIGRAPHX_DISABLE_MIOPEN_FUSION=1`
Counterintuitive: disabling some fusion passes fixes broken ones on gfx115x.
Validated by Immich community on gfx1150.
```bash
MIGRAPHX_DISABLE_MIOPEN_FUSION=1 \
    python export/backbone/compile_backbone_mxr.py --imgsz 504 --backbone-source detector --force
```

### 5. `MIGRAPHX_TRACE_MLIR=1`
Diagnostic: verify which MLIR passes are actually firing on the ViT graph.
Run during compile and search for attention-related fusions.

---

## Confirmed NOT worth doing

| Method | Reason |
|---|---|
| GPU-resident backbone (HIP IPC) | Measured: only 1.5ms copy overhead (1.6%), kernel is 93ms |
| AOTriton / Flash Attention | gfx1151 has 3.7× regression, flash-attention issue #2392, unfixed as of 2026-05 |
| ORT MIGraphX EP for backbone | Graph partition overhead + CPU fallback risk, no benefit |
| INT8 quantization | FP8 only on MI300; INT8 has accuracy risk + ROCm 7.2.3 known regression |
| FP8 quantization | Hardware-accelerated only on CDNA3 (MI300), not RDNA |

---

## Key references
- [MIGraphX CHANGELOG](https://github.com/ROCm/AMDMIGraphX/blob/develop/CHANGELOG.md)
- [ROCm #5643: hipBLASLt fallback on gfx1151](https://github.com/ROCm/ROCm/issues/5643)
- [flash-attention #2392: gfx1151 3.7x regression](https://github.com/Dao-AILab/flash-attention/issues/2392)
- [ROCm #5404: AOTriton missing for gfx1151](https://github.com/ROCm/ROCm/issues/5404)
- [ORT Transformers Optimizer](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- [Immich gfx1150 MIGraphX fix](https://gist.github.com/LukaPrebil/590f433b55cfe1bfb1690674c24df05c)
- [ComfyUI RDNA3 MLIR fix](https://github.com/Comfy-Org/ComfyUI/issues/10460)

---

## Benchmark Results (2026-05-13)

### Variant comparison @504px (careful A/B, 3 rounds × 25 frames)
| Variant | median ms | mean±std | FPS | vs baseline |
|---|---|---|---|---|
| baseline (original) | 197 | 200±10 | 4.99 | — |
| **mlir_attention** ✅ | **166** | **169±10** | **5.91** | **+18%** |
| no_miopen | — | ~218 | ~4.6 | -9% |
| hipblaslt | — | ~416 | ~2.4 | **-52% (broken on gfx1151)** |
| mlir_hipblaslt | — | ~484 | ~2.1 | **worst** |

**`MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention"` confirmed: 18% faster, IoU=0.9995 (mask quality preserved)**

### Accuracy check (baseline vs mlir_attention, 30 frames)
- mask IoU: mean=0.9995, min=0.998
- score diff: mean=0.001, max=0.001 (FP16 rounding only)

### Updated baseline (2026-05-13, 504px, MLIR attention backbone)
| Module | ms | % |
|---|---|---|
| vision_encoder (backbone) | 97 | 57% |
| memory_attention (ORT MIG EP) | 19 | 11% |
| detr_encoder (ORT MIG EP) | 12 | 7% |
| detr_decoder (PT) | 11 | 7% |
| tracker_neck (PT) | 4 | 2% |
| others | 4 | 2% |
| **Total** | **~169ms → 5.9 FPS** | |
