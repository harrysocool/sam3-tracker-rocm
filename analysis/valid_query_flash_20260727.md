# SAM3 grouped valid-query FlashAttention

**Date:** 2026-07-27

**Status:** correctness, accuracy, 1/5/30-frame validation passed

## Motivation

The 36x36 token grid is padded to four 24x24 windows. Every window keeps 576
K/V positions because zero padded inputs become non-zero after QKV bias and
participate in the current model semantics. Only query/output rows can be
removed exactly.

Valid query counts are 576, 288, 288, and 144. The validated q64 microkernel
uses LQP=192, so the execution counts are padded to 576/384/384/192:

```text
full schedule:    4*576 = 2304 query rows
compact schedule: 576+384+384+192 = 1536 query rows
reduction:        33.3%
```

## Explored approaches

### High-level piecewise launch mapping

A 64-launch piecewise `(launch_id -> q_iter, head_group)` mapping was generated
with `divui/remui/select`. AIR locality verification rejected it because the
raw launch IV no longer appeared directly in every operand offset. Even
`strict=false` treated the missing disjoint access proof as a hard error. The
source snapshot was quarantined and not committed.

### q32 exact-ish tile

LQP=96/tile_q=32 would have executed 576/288/288/192 rows, a 41.7% reduction.
The design compiled statically but timed out on hardware at LQ=576. No D-state
remained. The q32 path was abandoned.

### q64 placed-runtime pruning

The known-good LQP=192/tile_q=64 uniform 64-head design lowers to 96 independent
runtime blocks. Each block contains 24 DMA tasks and corresponds to:

```text
block = q_iter*32 + two-head group
```

The placed-MLIR patcher removes:

- q_iter=1 for head groups 24-31;
- q_iter=2 for head groups 8-31.

Static operation counts changed exactly as expected:

```text
launch blocks       96 -> 64
DMA configure/start 2304 -> 1536
await               576 -> 384
free                1728 -> 1152
```

The core/init/enable CDOs are byte-identical to the uniform q576 base. Only the
runtime instruction stream differs.

## Correctness

Uniform q384 and q192 kernels passed the existing FlashAttention reference at
mean relative L1 4.54% and 4.58%. Both instruction streams also executed using
the q576 xclbin.

The final 64-head grouped runtime was checked against an independent NumPy
reference:

```text
elapsed_ms=4.612
mean_rel_L1=0.045234
max_abs=0.051896
unused_max=0.000000
VALID_QUERY_GROUPED=PASS
```

A separate CPU proof compacted valid spatial query rows and scattered outputs
back to the 36x36 grid while retaining full K/V padding semantics:

```text
valid-query window semantics PASS max_abs=1.6391277e-07
```

## Artifacts

```text
grouped xclbin:
  /home/amd/project/npu_iron/sam3_attn/valid_query_flash_grouped_wip_20260727/final.xclbin
  SHA256 9c77487eeaa99aa9e413feac7999307c593dd040e4288cf6a41120dcfb565f1c
grouped insts:
  SHA256 0e8c4e2dcf6fc7ec1aeb3d424822f7f41610f243392f9b40f7c90fa0019fa1d3
backbone:
  /home/amd/project/npu_iron/bh_validq_hostopt_20260727
  SHA256 af71f158080742d60da5aaf4412f6fb7a1d2aea0fb8ebdf8e9bcf06e1b80657d
```

Relevant commits:

```text
7cb38f6 feat(iron): generate valid-query flash shapes
2dd6d01 fix(iron): use q32 valid-query flash tiles
1c4c479 feat(iron): add q64 valid-query flash fallback
f5fe3b7 feat(iron): prune invalid window-query launches
365b38a perf(iron): compact valid window queries
8c07698 perf(iron): promote grouped valid-query backbone
```

## Backbone performance

One-frame gate:

```text
wall=785 ms
dispatch=658 ms
flash=151 ms
frame_completed=1
```

The promoted binary subsequently completed one frame at 780 ms.

Accuracy is unchanged:

```text
last_hidden_state  0.993185
per-token mean/min 0.993426 / -0.379358
FPN p2             0.999479
FPN p3             0.998525
FPN p4             0.997761
FPN p5             0.997274
```

Thirty-frame results:

| Metric | n | min | mean | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++ backbone wall ms | 30 | 774 | 826.1 | **784.5** | 1019.4 | 1032.0 | 1044 |
| dispatch ms | 30 | 651 | 696.3 | **654.0** | 894.7 | 903.8 | 907 |
| Python-visible NPU ms | 30 | 780.1 | 832.6 | **791.1** | 1026.0 | 1039.0 | 1052.0 |
| full image-to-FPN ms | 30 | 788.9 | 841.2 | **799.8** | 1033.3 | 1045.9 | 1060.9 |

All frames completed on the runtime-PM-fix driver. Post-run use count was zero
and no D-state task was present. Periodic arbitrary-dispatch stalls remain and
continue to dominate p95.

## Decision

- Grouped valid-query becomes the new validated performance base.
- Normal C++ backbone p50 improved from 980 ms baseline to 784.5 ms.
- Dynamic-K remains disabled; this optimization is independent of it.
- The next 500 ms lever must reduce projection/FFN dispatch or remove remaining
  host/device boundaries. Periodic stall elimination remains mandatory for
  p95 below one second.
