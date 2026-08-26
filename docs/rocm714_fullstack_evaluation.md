# ROCm 7.14 Full-Stack Evaluation on gfx1151

**Date:** 2026-08-25
**Hardware:** AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)
**Workload:** `assets/blackswan.mp4`, prompt `swan`, 504 px, 30 propagation frames

## Summary

An isolated ROCm 7.14 stack was built to test newer MIGraphX and rocMLIR without
modifying the stable host installation. The complete container path reduced the
profiled propagation latency from 138.58 ms to 111.65 ms and increased the
end-to-end propagation rate from 7.06 FPS to 8.51 FPS.

The gain comes primarily from the newer ONNX Runtime/MIGraphX combination on
the DETR encoder and memory attention. The newer backbone is only about 4%
faster than the stable backbone.

The Docker path is an additional gfx1151-specific runtime. It does not replace
the native ROCm 7.2/7.13 compatibility path.

## Tested stack

| Component | Version |
|---|---|
| Base OS | Ubuntu 24.04 |
| ROCm runtime | 7.14 |
| GPU target | gfx1151 |
| MIGraphX | 2.17.0, commit `9f1a138e77f4738d82a065d225836b3b337950ce` |
| rocMLIR | `rockCompiler 2.0.0`, revision pinned by the MIGraphX commit |
| ONNX Runtime | 1.24.2, commit `058787ceead760166e3c50a0a4cba8a833a6f53f` |
| PyTorch | `2.11.0+rocm7.13.0` gfx1151 code objects |
| Torch runtime libraries | ROCm 7.14 APT libraries under `/opt/rocm` |
| Transformers | 5.8.1 |

Final local image:

```text
sam3-gpu714-ort1242-mgx217-gfx1151:torch211
sha256:c3101960a75b6ee376bbb59ac071a5c57f51a109e220e3a2ae06bc9e037799b9
```

The full from-source build and runtime instructions are in
`docker/rocm714/README.md`.

## PyTorch gfx1151 compatibility

AMD's ROCm 7.14 `whl-multi-arch` PyTorch wheels report only `gfx942` in
`torch.cuda.get_arch_list()` and fail on gfx1151 with
`hipErrorInvalidImage`. The gfx1151-specific repository currently publishes
PyTorch wheels built against ROCm 7.13.

The validated container uses the gfx1151 code objects from that wheel but
prevents its Python ROCm bootstrap from loading bundled ROCm 7.13 libraries.
The dynamic linker resolves the ABI-compatible ROCm 7.14 APT libraries instead.
Runtime inspection confirmed:

```text
libamdhip64.so -> /opt/rocm/core-7.14/lib/libamdhip64.so.7.14...
librocblas.so  -> /opt/rocm/core-7.14/lib/librocblas.so.5.5
```

Torch elementwise operations, FP16 GEMM, FP16 convolution, and Torch GPU
pointer interoperability with ORT MIGraphX EP all passed.

The newer PyTorch 2.12 nightly is not usable with the ROCm 7.14 system stack:
it requires `librocsolver.so.1`, while ROCm 7.14 provides a different ABI major
(`librocsolver.so.0`). No compatibility symlink is used.

## Artifacts

All generated artifacts remain outside Git:

```text
/home/amd/project/sam3-artifacts/gpu/experiments/fullstack-rocm714/
  backbone_detector/tuned_gpuio.mxr
  detr_cache/
  memory_cache/                  # S1 through S10
  profile_full_synced.json
  profile_stable_synced.json
  mask_diff_synced.json
  mask_diff_synced_rerun.json
  text_fullmodel_synced.mp4
```

New backbone SHA256:

```text
91870b540932e93e51958c1b4af4b0e4a897b52032f42fa10aaeea4e3f3f2c1f
```

The native stable backbone was not modified:

```text
42b82af4dc9b3f146e15105877f5217cd8e4a98a1cdefef8676ddcefa4639508
```

## Performance

### Full-model profile

Both rows use the same source revision, including explicit ORT output
synchronization.

| Stage | Native stable stack | ROCm 7.14 container | Change |
|---|---:|---:|---:|
| Vision encoder | 66.98 ms | 64.30 ms | -4.0% |
| DETR encoder | 6.96 ms | 3.11 ms | -55.3% |
| Memory attention | 13.77 ms | 7.44 ms | -46.0% |
| DETR decoder | 11.2 ms | 11.3 ms | neutral |
| Total propagation | 138.58 ms | **111.65 ms** | **-19.4%** |
| Profile throughput | 7.22 FPS | **8.96 FPS** | **+24.1%** |

### End-to-end path

`tools/text_baseline.py` includes mask processing, rendering, and video output.

| Runtime | Propagation FPS |
|---|---:|
| Native stable stack | 7.06 |
| ROCm 7.14 container | **8.51** |

End-to-end throughput improved by approximately **20.5%**.

## Correctness and synchronization

The first regression run exposed one transient bad frame at frame 24. ORT
I/O binding was returning a GPU-backed Torch tensor without explicitly waiting
for the provider output stream. New MIGraphX uses asynchronous external-stream
execution, making the missing synchronization observable.

`tracker/ort_gpu_io.py` now calls:

```python
session.run_with_iobinding(binding)
binding.synchronize_outputs()
```

The synchronization had no measurable performance penalty. Two consecutive
30-frame regressions after the fix produced:

| Run | Mean IoU | Minimum IoU | Frames below 0.95 |
|---|---:|---:|---:|
| 1 | 0.994174 | 0.989274 | 0 |
| 2 | 0.994109 | 0.989274 | 0 |

The fix is commit `0a03018`.

## Rejected optimization experiments

The accepted backbone uses MLIR attention with all experimental fusion and
tuning switches disabled.

| Experiment | Result |
|---|---:|
| Stable/default new-stack backbone | 61-62 ms p50 |
| MLIR exhaustive tune, limit 16 | 138.57 ms |
| MLIR exhaustive tune, limit 64 | 89.42 ms |
| MLIR exhaustive tune, limit 64 + split-K | Segmentation fault (exit 139) |
| MLIR input fusion | 66.30 ms |
| MLIR reduce fusion | 64.36 ms |
| MLIR GEG fusion | 120.66 ms |
| Input + reduce + GEG fusion | 121.39 ms |
| Force MLP dot/fused-dot from MLIR to hipBLASLt | 119.04 ms |
| Backbone hipBLASLt tuning | 64.82 ms |
| DETR hipBLASLt tuning | 2.394 to 2.377 ms; noise-level |
| Memory-attention hipBLASLt tuning | 6.717 to 6.721 ms; no gain |
| `MIGRAPHX_NSTREAMS=2` | 69.38 ms |
| One CPU/OpenMP thread | 111.65 to 111.35 ms; noise-level |
| Experimental AOTriton attention | 112.94 ms total; net regression |

Rejected model artifacts were deleted.

## Backbone kernel profile

The remaining backbone time is concentrated in already-fused MLIR kernels:

| Kernel family | Share of backbone time |
|---|---:|
| First MLP projection + GELU | 25% |
| Second MLP projection + residual | 18% |
| QKV projection | 18% |
| Attention | 12% |

These four groups account for approximately 73% of backbone time. FP16-to-FP32
input conversion and per-frame output allocation together cost only about
0.08 ms, so Python allocation cleanup is not a useful target.

Forcing the fused MLP kernels through hipBLASLt roughly doubled backbone time.
Further meaningful single-frame gains require a gfx1151-specific fused MLP
kernel or compiler work, not additional environment-variable tuning.

## Remaining directions

1. A custom gfx1151 fused MLP implementation targeting the two dominant MLP
   projections. A 20% improvement to that portion would save roughly 5 ms per
   full frame.
2. Cross-frame backbone pipelining or double buffering. This has a larger
   theoretical ceiling but changes scheduling semantics and was intentionally
   left out of this investigation.

The current ROCm 7.14 Docker configuration is the best validated single-frame
configuration from this evaluation.
