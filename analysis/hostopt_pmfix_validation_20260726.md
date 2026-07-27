# SAM3 IRON hostopt and runtime-PM fix validation

**Date:** 2026-07-26

**Driver:** temporary TDR + runtime-PM/mailbox-fix candidate

## Driver candidate

```text
workspace: /home/amd/project/amdxdna_tdr_pmfix_20260726
branch: fix/tdr-pm-deadlock
module SHA256: a302f61ed45f3bf6a053c967055d5b9d591d0ec96273caf208137a15d8a61124
tdr_timeout_ms=2000
tdr_dump_only=N
```

The candidate includes upstream runtime-PM deadlock fix `2be0d73` and mailbox
teardown guard `0220d14` on top of the scheduler-TDR backport.

## Dynamic-K v5 result

The v5 static artifact check passed, but the bounded common-xclbin ABI sequence
failed at the first K=4 -> K=20 transition:

```text
shape=ffn2
expected=5120
mean=4949.333333
bad=65536/1572864
first_value=1024
```

The end-of-output barrier therefore does not provide a reliable runtime-
sequence boundary. The process exited normally and no D-state appeared.
Dynamic-K v5 and the guarded Phase 2 backbone remain disabled.

## Hostopt single frame

The independent hostopt candidate retains the validated K=1024 shared overlay
plus separate FFN2 design. Its single-frame gate completed:

```text
wall=849 ms
dispatch=703 ms
host gap=146 ms
QKV pack=5.6 ms
FFN1 pack=5.6 ms
GELU pack=29.6 ms
D-state tasks: none
```

Compared with projection compaction alone, mapped input conversion and valid-
region GELU reduced normal host work by roughly 38-52 ms.

## Accuracy

The timed image-backed NPU call was 857 ms. Accuracy is unchanged:

```text
last_hidden_state  0.993185
per-token mean/min 0.993426 / -0.379358
FPN p2             0.999479
FPN p3             0.998525
FPN p4             0.997761
FPN p5             0.997274
```

## Five-frame gate

The full image->embed->NPU->FPN timed frames were:

```text
838.4, 1109.1(stall), 883.5, 867.3, 840.9 ms
```

The stall completed normally and no D-state appeared.

## Thirty-frame gate

The run completed all 30 timed frames, crossing the previous frame-26 wedge
point. Six periodic arbitrary-dispatch stalls remained, but none became a
permanent wait.

| Metric | n | min | mean | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++ backbone wall ms | 30 | 827 | 876.0 | **835.0** | 1064.7 | 1071.5 | 1103 |
| dispatch ms | 30 | 697 | 740.1 | **699.5** | 930.6 | 936.5 | 960 |
| Python-visible NPU ms | 30 | 832.4 | 882.5 | **841.5** | 1071.7 | 1077.5 | 1109.5 |
| full image-to-FPN ms | 30 | 840.8 | 890.8 | **850.6** | 1080.1 | 1085.6 | 1118.9 |

Post-run state:

```text
amdxdna use count: 0
D-state tasks: none
TDR-related kernel messages: none
```

This is strong first evidence that moving the runtime-PM get before scheduler
queueing prevents the real deadlock mode seen on the previous boot. It is not
proof of causality or production stability; repeated long runs and TDR log
inspection were still required. The privileged post-run check subsequently
confirmed `tdr_timeout_ms=2000`, recovery mode enabled, no TDR messages, and no
D-state. The periodic ~220-270 ms stall is unchanged and continues to keep p95
above one second.

## Decision

- Hostopt becomes the new validated performance candidate.
- PM-fix driver candidate passes load, one frame, accuracy, five frames, and
  one 30-frame run.
- Dynamic-K v5 remains WIP and must not be used by the backbone.
- Do not immediately repeat long stress; continue with offline attention/shape
  work and a later independent stability run.

Raw logs:

```text
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/dynamic_k_v5_gate_20260726.log
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_hostopt_1f_20260726.log
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_hostopt_accuracy_20260726.log
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_hostopt_pmfix_5f_20260726.log
/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_hostopt_pmfix_30f_20260726.log
```
