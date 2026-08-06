# SAM3 ViT XDNA2 NPU benchmark

This benchmark closes the project around one reproducible objective: compile
and run the complete SAM3 ViT detector backbone through two independent XDNA2
NPU routes and compare both with the same PyTorch FP16 reference.

## Canonical workload

```text
Input:        [1, 3, 504, 504] float image tensor
ViT tokens:   1296
Hidden size:  1024
Layers:       32
Outputs:      FPN p2-p5 plus last_hidden_state
Reference:    PyTorch FP16
Image:        assets/truck.jpg
```

## Routes

### flexml / ONNX Runtime VitisAI EP

The canonical ONNX contains the complete detector vision encoder, including
patch embedding, 32 ViT blocks, and the FPN neck.  Constructing a VitisAI EP
session with a new cache key is the compile step.  Reusing the resulting `.rai`
is the normal benchmark path.

The complete backbone is submitted as a long NPU job. Its validated latency is
about 2.5 seconds, so the runtime profile must use `tdr_timeout_ms >= 5000`.
The IRON recovery profile currently installed on the development machine uses
a 2000 ms scheduler timeout and will terminate this flexml job before it
completes. `run_benchmark.sh` detects that profile and refuses to run flexml
instead of generating a misleading TDR failure. Changing the module profile
requires the privileged, attended driver-reload procedure documented in the
`amdxdna-recovery` runbook; the normal runner does not change it.

When an operator is physically present and can enter sudo credentials, use the
guarded helper below. It verifies the validated module SHA, switches to a 10
second timeout, runs one warm-up plus three measurements, checks for scheduler
timeouts, and restores the normal 2 second profile through an EXIT/INT/TERM
trap:

```bash
bash eval/benchmarks/npu_vit/run_flexml_attended_tdr.sh
```

Do not launch this helper unattended. If a D-state task appears, it refuses to
unload the module and preserves the machine for attended recovery.

Validated artifacts:

```text
onnx_files_504/backbone_detector/single_simplified.onnx
npu_artifacts/voe_cache_504/backbone_detector_504_v1/backbone_detector_504_v1.rai
```

To compile into a new, non-overwriting cache key:

```bash
bash eval/benchmarks/npu_vit/compile_flexml_cache.sh \
  --onnx onnx_files_504/backbone_detector/single_simplified.onnx \
  --cache-dir npu_artifacts/voe_cache_504 \
  --cache-key backbone_detector_504_benchmark_v2
```

### MLIR-AIE / IRON

The canonical reproducible runtime is the frozen P14 M1536 release.  It runs
the 32 ViT blocks on the NPU, with patch embedding and the FPN neck on the GPU.
It contains a controlled BFP/affine approximation in 14 non-global layers; the
benchmark records its measured cosine values rather than describing it as an
exact-model result.

```text
/home/amd/project/npu_iron/releases/sam3-vit-p14-m1536-power-v1
```

Rebuild the C++ host without overwriting the frozen binary:

```bash
bash eval/benchmarks/npu_vit/build_iron_host.sh /tmp/sam3-vit-p14-rebuilt
```

The NPU artifact compilation is intentionally stage-based because it uses two
compiler environments.  The frozen release was assembled from:

| Artifact family | Compiler/build entry |
|---|---|
| shared BF16 GEMMs | `eval/benchmarks/npu_iron/gen_shared_gemm_candidate.sh` |
| ATB M1536 QKV/O/FFN1 | `atb_eval_20260727/build_atb_token_m1536_qkv_20260730.sh` and `build_atb_token_m1536_family_20260730.sh` |
| grouped window Flash + K4864 overlay | `atb_eval_20260727/build_sam_gemm_overlay_multiseq_20260728.sh` |
| frozen package and host link | `atb_eval_20260727/package_p14_release_20260804.sh` |

The release `MANIFEST.sha256` is the binding record between those stage outputs
and the canonical benchmark binary.

## Verify and run

Run from the tracker repository:

```bash
bash eval/benchmarks/npu_vit/verify_benchmark_artifacts.sh

bash eval/benchmarks/npu_vit/run_benchmark.sh iron \
  --warmup 1 --runs 5 --omp-threads 8

# Attended only; prompts for sudo and restores the 2000 ms profile.
bash eval/benchmarks/npu_vit/run_flexml_attended_tdr.sh
```

Each run writes JSON under `results/npu_vit_benchmark/` with:

- artifact and input SHA256;
- warm-up and measured-run counts;
- min, mean, p50, p95, and max latency;
- last-hidden, per-token, and FPN cosine metrics;
- route metadata, kernel version, and tracker Git revision.

Canonical result records are checked in under `reference_results/`:

| Route | p50 / p95 | last-hidden cosine | minimum FPN cosine |
|---|---:|---:|---:|
| IRON P14 M1536 | 701.449 / 703.845 ms | 0.992889 | 0.996359 |
| flexml/VitisAI EP | 2489.509 / 2661.741 ms | 0.951607 | 0.962097 |

For stable IRON tail latency, run the command inside the installed scoped
performance-mode wrapper.  Automatic runtime-PM may add a periodic outlier;
the JSON preserves it rather than silently filtering it.

The two routes use different validated scheduler profiles:

| Route | Required amdxdna scheduler profile |
|---|---|
| IRON | the 2000 ms TDR recovery profile is supported |
| flexml/VitisAI EP | TDR timeout at least 5000 ms, or the original validated no-TDR stack |

## Scope boundary

This benchmark validates compilation, execution, correctness, and performance.
It does not claim real-time robotics suitability.  Asynchronous video keyframe
integration, a 500 ms target, and further kernel exploration are outside the
closeout acceptance criteria.
