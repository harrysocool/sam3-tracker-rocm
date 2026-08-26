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

A follow-up execution-schedule optimization overlaps the independent detector
and tracker branches after the shared backbone. The opt-in `--parallel-tail`
path reaches 8.93-9.09 FPS end to end (9.03 median) while preserving the
serial path as the default.

For preloaded videos, a second opt-in stage pipelines frame N+1's vision
backbone against frame N's parallel detector/tracker tail. It reaches
10.21-10.27 FPS end to end without changing model outputs.

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

/home/amd/project/sam3-artifacts/gpu/experiments/parallel-tail/
  scoped_fence_50f_ab.json
  scoped_fence_person_dog_50f_ab.json
  backbone_pipeline_final_acceptance.json
  backbone_pipeline_person_dog_final_acceptance.json
  mask_diff_pipeline_run1.json
  mask_diff_pipeline_run2.json
  sg3_pipeline.json
  mask_diff_parallel.json
  sg3_serial.json
  sg3_parallel.json
  davis_val_504_post_parallel.json
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
| ROCm 7.14 + `--parallel-tail` | **8.93-9.09** (median 9.03) |
| ROCm 7.14 + both pipeline flags | **10.21, 10.27** |

The serial ROCm 7.14 path improves on the native stack by approximately 20.5%.
The median parallel-tail run improves on native by 27.9% and on the serial
7.14 result by 6.1%. The median pipelined result improves on native by 45.0%
and on the serial 7.14 path by 20.3%.

### Parallel detector/tracker tail

After the shared vision encoder completes, the detector branch and tracker
propagation branch have no data dependency until mask association. The
experimental scheduler runs them on two persistent worker threads and two HIP
streams, then joins before the unchanged association/update phase.

Two interleaved, hot-process A/B pairs over the complete 50-frame bundled
clip, with the ORT input fence scoped to parallel workers, produced:

| Schedule | Mean propagation latency | Model throughput |
|---|---:|---:|
| Serial | 117.80 ms | 8.49 FPS |
| Parallel tail | **108.87 ms** | **9.18 FPS** |

That is a 7.6% latency reduction and 8.2% model-throughput increase. The
un-instrumented video path, which also includes rendering and encoding, reached
8.93-9.09 FPS (median 9.03 across three runs). The existing per-module profiler
is intentionally not used to measure this optimization because its device-wide
synchronization hooks serialize the two branches.

Correctness checks:

- Two serial/parallel 50-frame pairs had bit-identical raw mask tensors,
  scores, and object IDs.
- The canonical 30-frame PT-vs-MIG regression remained at mean IoU 0.994175,
  minimum IoU 0.989274, with no frame below 0.95.
- A seeded three-sequence SG text subset produced byte-identical prediction
  JSON in serial and parallel modes.
- A two-prompt, three-object 50-frame clip produced bit-identical object masks,
  object IDs/prompt ownership, and scores. Its two-run hot averages were
  173.50 ms serial and 147.89 ms parallel (-14.8%).
- The full DAVIS 2017 validation regression remained at mean J 0.8156, matching
  the saved 504 px baseline. This exercises the unchanged box-tracker path and
  guards against branch-level regressions.

Enable the path with `--parallel-tail`. The implementation also fences the
calling Torch stream before ORT consumes externally bound input pointers; ORT
output synchronization remains enabled.

Use `eval/benchmarks/benchmark_parallel_tail.py` for alternating serial/parallel
runs with per-frame mask, object-ID, prompt-ownership, and score checks.

### Cross-frame backbone pipeline

For a preloaded clip, `--pipeline-backbone` starts frame N+1's stateless
MIGraphX vision encoder while frame N is in the parallel detector/tracker tail.
Tracker state updates remain strictly ordered. The flag requires
`--parallel-tail`; it is intentionally unavailable in `SAM3Live`, where the
next camera frame has not arrived yet.

The corrected steady-state window compares the same frame indices (29-48) and
excludes both pipeline fill and drain:

| Schedule | Output interval | Throughput |
|---|---:|---:|
| Serial | 117.80 ms | 8.49 FPS |
| Parallel tail | 108.87 ms | 9.18 FPS |
| Parallel tail + backbone prefetch | **94.57 ms** | **10.57 FPS** |

Relative to serial, the combined schedule lowers the steady output interval by
19.7% and raises throughput by 24.6%. Prefetch alone, measured on top of the
parallel tail in the same runs, lowers the interval by another 13.1%. This is
throughput optimization rather than single-frame response-time reduction: the
first propagation output includes a roughly 147 ms pipeline fill.
The lookahead retains one extra frame's input and vision outputs (roughly
50 MB at 504 px) until its completion event is consumed.

The two-prompt/three-object 50-frame test measured 173.09 ms serial, 148.73 ms
with parallel tails, and 138.76 ms with both pipeline stages. All 300 compared
raw object-mask tensors, object IDs, prompt ownership, and scores matched the
serial output exactly across two reversed-order runs.

Two canonical PT-vs-MIG runs with both flags retained mean IoU 0.994142 and
minimum IoU 0.989274, with no frame below 0.95. The seeded three-sequence SG
prediction JSON was byte-identical to the serial result.

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
| `torch.compile(max-autotune)` DETR decoder | 5.8 ms microbenchmark, but 113.4-122.2 ms full-model; numerical changes altered object lifecycle |
| Remove unused detector `fpn_3` output | No repeatable backbone gain; recompiled output also introduced avoidable numerical drift |
| Standalone tracker-neck MIGraphX graph | 3.13 to 2.72 ms microbenchmark; ~0.4 ms does not justify another artifact/runtime boundary |
| `torch.compile` tracker neck | 3.13 to 2.85 ms best case; lower gain than standalone MIGraphX |

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
2. Fixed B=2/B=4 backbone micro-batching. Cross-frame lookahead is now
   implemented for preloaded video, but batch-level weight reuse remains
   unexplored.

The current ROCm 7.14 Docker configuration is the best validated single-frame
configuration from this evaluation.
