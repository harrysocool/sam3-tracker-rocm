# IRON sub-1s Phase 0 baseline — 2026-07-23

## Scope

This report freezes the first Phase 0 measurements for the SAM3 IRON backbone
before host zero-copy or shared-GEMM changes. The target is single-frame
latency; no cross-frame pipelining or batch amortization is included.

## Rebuild

The current committed source was rebuilt without overwriting the production
binary.

| Item | SHA256 |
|---|---|
| `bh_flash_test.cpp` | `819576e09088672a9c92f662b8ae99b5c85bf03ae1601dc4f2c3b8fbdaa6ffe5` |
| `bh_npu_backbone_flash_rebuild_20260723` | `a2e260cc90a329430ab83601132a8820e833dfd8c56467b0bae8daedec2f52b7` |
| `block0_in.bin` | `3ad14a382fff40a761d55bc91af809122aaa8340fdf207fe76967f41f7853615` |
| `final_feat.bin` | `8bd4fd1648e8dc13721c47220fff5989f1e68785ab6f1ee65095c7f8c7ffb9cd` |

Build flags matched the existing BF16 host recipe:

```text
-O3 -march=native -mavx512f -mavx512bf16 -ffast-math
-funroll-loops -fopenmp -std=c++17
```

Runtime stack:

```text
kernel       6.14.0-1020-oem
XRT          2.21.75
firmware     1.1.2.65
OMP threads  8
```

## First 30-frame run

The NPU-using `sam3_node` had been stopped, but Gazebo/RViz and the ROS launch
were still present. The run completed and is useful for separating stable host
cost from NPU dispatch outliers, but it is not the final clean baseline.

Accuracy against the stored `final_feat.bin` reference:

```text
cos = 0.99196
```

Latency statistics:

| Metric | min | mean | p50 | p90 | p95 | max | stddev |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall ms | 1169 | 1238.8 | 1189 | 1407.5 | 1432.8 | 1445 | 99.1 |
| dispatch ms | 810 | 869.9 | 817 | 1041.6 | 1057.1 | 1071 | 97.9 |
| wall-dispatch ms | 356 | 368.9 | 370 | 377.1 | 378 | 379 | 6.0 |

Seven periodic dispatch spikes occurred at frames 2, 6, 10, 14, 19, 23, and
27. Stable frames were approximately 1.17–1.20 s wall / 810–822 ms dispatch;
spike frames were approximately 1.40–1.45 s wall / 1.03–1.07 s dispatch. The
host gap remained stable, so the variance is inside NPU dispatch/wait rather
than CPU transforms.

Raw log:

```text
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/
baseline_rebuild_20260723.log
```

## Driver hang on the exclusive rerun

After the complete ROS showcase exited, a second 30-frame run loaded xclbins,
allocated BOs, and made weights resident, but hung before the first timed frame
completed. The runner was interrupted after exceeding the normal duration by
several times.

Subsequent `xrt-smi examine -r aie-partitions` also hung in the kernel:

```text
state  D
wchan  amdxdna_drm_open
```

An `amdxdna_js` kworker was also in D-state. The user cannot write the PCI reset
or driver unbind sysfs nodes and non-interactive sudo is unavailable. SIGINT,
SIGTERM, and SIGKILL cannot remove a process blocked in uninterruptible kernel
sleep. The NPU therefore requires a machine reboot or a privileged driver/PCI
reset before further hardware measurements.

No production binary, xclbin, driver package, or firmware was modified.

## Profiling implementation

Local branch:

```text
perf/iron-sub1s-backbone
```

Commit:

```text
db1f0bb perf(iron): add per-stage backbone profiler
```

The profiler compiles successfully and separates:

- QKV/O/FFN1/FFN2/flash dispatch;
- input packing and BO synchronization;
- QKV split and head layout;
- RoPE/flash packing;
- LayerNorm, GELU, partition/unpartition, and residual work.

It has not been executed because the device entered the driver hang described
above.

## Resume checklist after reboot

1. Confirm `xrt-smi examine -r aie-partitions` completes and reports no context.
2. Confirm the ROS showcase is not running.
3. Run one frame with the rebuilt uninstrumented binary.
4. Run five frames, then 30 frames; stop immediately on a repeated driver hang.
5. Run the profiler for five frames before a 30-frame profiling run.
6. Compare the rebuilt binary with the old production binary under identical
   conditions.
7. Investigate periodic dispatch spikes before using p95 as a release claim.
