# SAM3 IRON projection compaction

**Date:** 2026-07-26

**Branch:** `perf/iron-sub1s-backbone`

**Status:** performance and accuracy passed; 30-frame stability run wedged after
25 completed frames

## Change

The 28 window-attention blocks previously partitioned the 36x36 grid into a
zero-padded 48x48 grid before QKV projection and ran both QKV and O projection
with M=2304. Projection is token-wise, so the candidate:

1. runs QKV on the 1296 valid rows using the existing M=1536 instruction
   stream;
2. maps QKV rows into the four 24x24 windows;
3. materializes padded rows as zero pre-bias projection output, then applies
   the original bias and position-specific RoPE;
4. keeps the existing padded K/V attention semantics;
5. gathers valid attention outputs before O projection;
6. runs O projection with M=1536 for every layer.

Commits:

```text
1e5ec3a perf(iron): compact window projection rows
fd59856 test(iron): gate compact projection candidate
9cb1295 test(iron): allow benchmark binary override
```

Artifacts:

```text
source: /home/amd/project/sam3-tracker-rocm/eval/benchmarks/npu_iron/bh_projcompact_20260726.cpp
binary: /home/amd/project/npu_iron/bh_projcompact_20260726
source SHA256: 3b055ebffc9966ed1c4abb817d8342c69fa6545ccf94ed8403c6f55ab6e0bdae
binary SHA256: 4d57c89b57faddd62c57a0467d5ac6b0a06b93989f105ed305e1eb839198d1f1
```

A reduced CPU semantic proof compares the original partition->QKV->attention
->O->unpartition order against the compact QKV->partition->attention
->unpartition->O order, including padded bias/RoPE/K/V behavior:

```text
projection-compaction semantics PASS max_abs=0
```

## Single-frame performance

The zero-input one-frame health gate completed normally:

```text
wall=930 ms
dispatch=728 ms
QKV=147.1 ms
O=98.9 ms
frame_completed=1
D-state tasks: none
```

The PyTorch accuracy probe produced two image-backed NPU frames at 890 and
887 ms. The timed Python-visible NPU call was 902 ms.

Compared with the exact production-candidate 980 ms profile:

```text
QKV dispatch   183.5 -> ~141-147 ms
O dispatch     107.7 -> ~94-99 ms
QKV packing     26.9 -> ~16-18 ms
partition        9.4 -> 0 ms
normal wall   970-1015 -> 877-918 ms
```

## Accuracy

The metrics are identical to the final sub-one-second baseline:

```text
last_hidden_state  0.993185
per-token mean/min 0.993426 / -0.379358
FPN p2             0.999479
FPN p3             0.998525
FPN p4             0.997761
FPN p5             0.997274
```

This confirms the projection reorder preserved the current padded-window model
semantics and BFP16 numerical path.

## Five-frame gate

The five timed full image->embed->NPU->FPN frames were:

```text
893.8, 1126.3(stall), 894.9, 899.5, 901.4 ms
```

The stall landed on FFN1 and completed normally. No D-state was present after
the gate.

## Incomplete 30-frame run and real wedge

The next run completed 25 frames, then blocked permanently on frame 26. The
completed C++ backbone wall times were:

```text
918, 883, 1084, 883, 908, 897, 918, 917, 1090, 881,
887, 882, 880, 1109, 896, 899, 881, 881, 882, 1181,
883, 883, 882, 877, 891 ms
```

Partial-run statistics, not release statistics:

| Metric | n | min | mean | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall ms | 25 | 877 | 926.9 | **887** | 1087.6 | 1105.2 | 1181 |
| dispatch ms | 25 | 698 | 740.8 | **703** | 903.0 | 924.6 | 983 |

Four completed periodic stalls landed on different stages/layers. On the next
frame the userspace binary remained in `drm_syncobj_array_wait` and an
`amdxdna_js` worker entered D-state. The production scheduler-TDR module did
not recover this real wedge. Userspace was terminated; no XRT diagnostic,
reset, unbind, module unload, or remote reboot was attempted.

This does not implicate projection compaction as a numerical or kernel-shape
failure: accuracy was unchanged, 25 frames completed, and the same periodic
arbitrary-dispatch stall existed before this candidate. It does prove that the
controlled host-response-delay TDR validation does not cover the real
driver/firmware wedge mode.

## Decision

- Projection compaction is retained as the new performance candidate.
- The achieved normal-frame range is about 880-900 ms, meeting the Phase 1
  target.
- The incomplete run cannot be published as a 30-frame release result.
- No further NPU work is allowed on this boot.
- Continue subsequent optimization offline until an on-site physical reboot.
- Before another long run, use a matched newer driver/firmware stack or extend
  recovery below the scheduler job layer; the current scheduler callback is
  insufficient for this real wedge.

Raw logs:

```text
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_1f_20260726.log
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_iron_vs_pt_20260726.log
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_5f_20260726.log
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_30f_20260726.log
```
