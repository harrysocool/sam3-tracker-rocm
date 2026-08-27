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
  backbone_b2_rejected.json

/home/amd/project/sam3-artifacts/gpu/experiments/backbone-kernel-profile/
  backbone_kernel_trace.csv
  backbone_kernel_stats.csv
  mlp_counter_summary.json
  qkv_probe_result.json
  {OccupancyPercent,L2CacheHit,MemUnitBusy,VALUInsts,FETCH_SIZE,WRITE_SIZE}/

/home/amd/project/sam3-artifacts/gpu/experiments/mlp-fc1-triton/
  real_finalists.json

/home/amd/project/sam3-artifacts/gpu/experiments/hipblaslt-fc1/
  result.json

/home/amd/project/sam3-artifacts/gpu/experiments/rocwmma-fc1/
  real_layers_exact_final.json
  rocwmma_fc1_exact_summary.json

/home/amd/project/sam3-artifacts/gpu/experiments/memory-attention-grouped/
  b1_vs_b2_s7.json
  b1_vs_b3_s7.json
  s7_b1_concurrency_comparative.json

/home/amd/project/sam3-artifacts/gpu/experiments/low-precision-probes/
  sam3_int8_fc1_final.json
  sam3_int8_fc1_tune.json
  sam3_quant_fc1_results.json

/home/amd/project/sam3-artifacts/gpu/experiments/
  deep_optimization_rejections_20260826.json
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
| Fixed B=2 backbone | 135.43 ms/batch vs 136.10 ms for two B1 calls (~0.5% gain); missed 120 ms gate and changed B1-vs-B2 numerics |
| Triton first MLP + exact GELU | 0.502 ms median vs 0.523 ms MLIR (~4.0%); missed the 0.42 ms gate |
| hipBLASLt first MLP | Fused tanh-GELU 0.394-0.407 ms; exact two-kernel path 0.439-0.454 ms and missed the gate |
| Hand-written rocWMMA first MLP | Exact path 0.476-0.497 ms on real layers; 138 VGPR, 12 KiB LDS, no scratch |
| Memory-attention B=2 / B=3 | 18.67 / 29.87 ms vs repeated-B1 15.75 / 23.82 ms; both slower |
| Concurrent B1 memory attention | Same-session execution silently corrupted output; two sessions saved only ~0.99 ms per pair |
| Batched multi-object mask decoder | 2.7% without backbone prefetch, but -0.7% with it; 50-frame minimum mask IoU 0.9368 |
| Prefetch next-frame detector tail | Bit-exact, but 110.12 to 124.88 ms (+13.4% latency) |
| Collapse association GPU-to-CPU synchronizations | Bit-exact, but 137.58 to 138.16 ms on the optimized three-object pipeline |
| Triton dynamic W8A8 first MLP | 0.463 ms vs 0.502 ms FP16 (~8.4% throughput); missed 0.42 ms gate and layer relative-L2 was 1.8-2.5% |
| Triton FP8 first MLP | ~4.9 ms; gfx1151 lowered it to FP16 WMMA with 130 spills |
| Whole-backbone MIGraphX INT8 PTQ | ~47 GiB RSS; OOM during serialization and multi-minute warmup behavior |
| Custom QKV projection | Window: Triton 0.527 ms / Torch 0.427 ms vs MLIR 0.430 ms; global paths also only matched MLIR |

Rejected model artifacts were deleted; only compact result summaries remain.

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

ROCm 7.14 hardware counters make the first MLP projection the clearest custom
kernel target. Across 15 backbone runs, its fused
`mlir_dot_add_mul_erf_add_mul` dispatch averaged 522.85 us per layer and
accounted for 25.81% of all backbone kernel time. For the
`1296x1024 @ 1024x4736` GEMM this is about 24.0 FP16 TFLOP/s. rocprof measured
roughly 78.8 MiB fetched and 48.3 MiB written per dispatch, or about 255 GB/s
effective traffic, close to the APU memory-bandwidth ceiling. The kernel also
reports 248 VGPRs, 20 KiB LDS and 772 bytes of scratch. A useful replacement
must therefore reduce spill/re-read traffic and register pressure; merely
batching more frames cannot fix this kernel shape.

The second MLP projection averaged 395.49 us, about 31.8 FP16 TFLOP/s and
102 GB/s measured traffic. It is a secondary compute/utilization target after
the first projection. Counter CSVs are stored under
`gpu/experiments/backbone-kernel-profile/`.

Forcing the fused MLP kernels through hipBLASLt roughly doubled backbone time.
Further meaningful single-frame gains require a gfx1151-specific fused MLP
kernel or compiler work, not additional environment-variable tuning.

### Custom first-MLP prototype

A standalone Triton 3.6 kernel was tested on the real layer-0 input and
weights (`1296x1024 @ 1024x4736`). It used an offline-packed contiguous
weight transpose, FP32 accumulation, FP32 bias/GELU arithmetic and the exact
erf GELU formula. A 400-configuration synthetic sweep was followed by 25
real-input finalists. The best configuration was `64x128x32`, four waves per
workgroup, `matrix_instr_nonkdim=16`, `kpack=2` and default
`waves_per_eu=0`.

Its median/p95 latency was 0.502/0.512 ms, only about 4% faster than the
current 0.523 ms MLIR kernel and well above the 0.42 ms integration gate.
Numerics were sound versus an FP32 reference (maximum absolute error 0.00195,
mean absolute error 2.30e-5), and the generated kernel eliminated scratch,
but still used 228 VGPRs and 12 KiB LDS. A persistent-kernel variant was slower
at about 0.67 ms. Because integrating a Triton HSACO into the monolithic graph
would require at least a MIGraphX C++ custom-op plugin plus ONNX parser work,
the small standalone gain does not justify integration.

The lower-level follow-up used hipBLASLt and a hand-written rocWMMA 2.2
kernel. hipBLASLt algorithm 2032 (`128x96x64`, zero workspace) reached
0.394-0.407 ms with its fused `GELU_BIAS` epilogue, but the epilogue implements
the tanh approximation rather than exact-erf GELU. Keeping exact semantics by
running a separate HIP GELU kernel took 0.439-0.454 ms.

The custom rocWMMA implementation used native gfx1151
`v_wmma_f32_16x16x16_f16`, cooperative double-buffered LDS loads, a `64x32`
wave tile, FP32 accumulation, register-mapped bias/exact-GELU, and a separate
16-row tail. The main kernel used 138 VGPRs, 12 KiB dynamic LDS, and no scratch.
Despite a misleading 0.419 ms result on saturated synthetic values, real
layers 0, 15 and 31 measured 0.497, 0.489 and 0.476 ms median respectively.
The exact-erf cost is data-dependent and leaves the complete implementation
above the 0.42 ms gate. This route was therefore stopped before MIGraphX
custom-op integration.

### Multi-object batching and scheduling probes

The three-object `person` + `dog` workload invokes memory attention exactly
three times per propagation frame. All three objects have the same spatial
slot count on every tested frame, so it is an ideal batching case. Nevertheless,
static S7/P64 graphs scaled negatively:

| Memory-attention schedule | Pair/triple latency | Repeated B1 | Relative throughput |
|---|---:|---:|---:|
| B=2 graph | 18.67 ms | 15.75 ms | 0.843x |
| B=3 graph | 29.87 ms | 23.82 ms | 0.797x |

Rows were batch-independent, but the compiled batch graphs had small numerical
drift (maximum absolute difference 0.00586) and no speed advantage. Concurrent
B1 calls on one ORT session appeared faster but silently corrupted outputs
(maximum absolute difference 9.50). Two independent sessions were exact but
saved only 0.99 ms per pair before pipeline contention while doubling session
resources. Both alternatives were rejected.

The existing experimental batched mask decoder was also repaired locally to
extract the valid diagonal from Transformers 5.8.1's erroneous `[N,N,...]`
advanced-index result. It reduced a non-prefetched three-object run by 2.7%,
but regressed the active backbone-prefetch pipeline by 0.7%. More importantly,
recurrent feedback amplified batch-GEMM drift to a minimum mask IoU of 0.9368
over 50 frames. The patch was reverted.

Finally, prefetching frame N+1's detector after its backbone was bit-exact but
increased the steady output interval from 110.12 to 124.88 ms. The current
pipeline already overlaps the current detector tail with the next backbone;
serializing them in one lookahead lane removes useful overlap and increases
GPU contention. The accepted scheduler therefore remains backbone-only.

The final exact host-side candidate collapsed repeated `.item()`/`.tolist()`
synchronizations in association into one small transfer and grouped prompt
indices directly in Python. Although bit-exact, it changed the optimized
three-object pipeline from 137.58 to 138.16 ms, so this sub-millisecond target
was also left unchanged.

### Low-precision probes

gfx1151 does execute INT8 dot products with native
`v_wmma_i32_16x16x16_iu8`. A Triton W8A8 first-MLP kernel with static
per-output-channel weights and dynamic per-row activation scaling reached
0.463 ms including the 0.033 ms activation quantizer. This was only an 8.4%
throughput improvement over the best FP16 Triton kernel and missed the 0.42 ms
gate. On real layers 0, 15 and 31, relative-L2 error was 1.8-2.5% before any
32-layer accumulation. FP8 is not natively supported by the gfx1151 backend;
Triton lowered it through FP16 WMMA, used 256 VGPRs plus 130 spills, and took
about 4.9 ms.

A direct MIGraphX whole-backbone INT8 PTQ experiment was also attempted with
real-frame calibration, followed by FP16 conversion of the remaining graph.
The public API quantizes both dot and convolution operators and cannot target
only the MLPs. The quantized graph reached roughly 47 GiB process RSS and was
killed by the system while serializing the compiled program. A separate
performance-only build using default scales compiled, but did not finish five
warmup executions within several minutes. No deployable MXR was produced, and
the original artifacts were not modified.

The QKV projection was checked independently as well. MIGraphX already merges
the three 1024-wide Q/K/V projections into one 3072-wide dispatch. For the 28
window-attention layers, the current fused reshape/transpose/dot/bias kernel
takes 0.430 ms; a single Torch `addmm` only matched it at 0.427 ms, while the
best Triton candidate took 0.527 ms. The four global-attention projections
showed the same result (0.313 ms MLIR, 0.311 ms Torch, 0.324 ms Triton). There
is therefore no remaining straightforward QKV fusion opportunity.

## Remaining directions

1. Further exact MLP work would require modifying the hipBLASLt generator or
   rocMLIR lowering itself; standalone Triton, hipBLASLt plus exact epilogue,
   and hand-written rocWMMA all missed the acceptance gate.
2. Memory-history reduction can lower multi-object cost, but it changes model
   behavior and is therefore an explicit accuracy/performance mode rather than
   a default optimization.
3. Otherwise, further material gains require model-level changes such as
   structured sparsity, distillation, or token/layer pruning with a new
   accuracy budget; the remaining exact runtime-only candidates have not paid
   for their complexity.

The current ROCm 7.14 Docker configuration is the best validated single-frame
configuration from this evaluation.
