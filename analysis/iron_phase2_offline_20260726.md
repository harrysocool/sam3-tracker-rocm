# SAM3 IRON Phase 2 offline work

**Date:** 2026-07-26

**Status:** compiled and statically validated; hardware execution prohibited on
the current D-state boot

## Context

Projection compaction reduced normal backbone latency to roughly 880-900 ms
with unchanged accuracy. Its attempted 30-frame gate wedged after 25 completed
frames and left an `amdxdna_js` worker in D-state. All work in this report was
therefore performed without opening the NPU.

## Mapped host-input candidate

The projection-compaction source was extended with three host-side changes:

1. QKV BF16 input is converted directly into the mapped M=1536 BO;
2. FFN1 BF16 input is converted directly into its mapped BO;
3. CPU bias+GELU writes directly into the FFN2 input BO and evaluates only the
   valid 1296x4736 region.

The padded M rows and hidden columns are initialized to zero once and remain
zero. All 32 exported W1, b1, and W2 tensors were checked; every value in the
4736:5120 padded region is exactly zero:

```text
host-compaction padded weights PASS
W1_tail=0, b1_tail=0, W2_tail=0
```

Artifacts:

```text
commit: 68e4539 perf(iron): compact mapped host inputs
source: bh_projcompact_hostopt_20260726.cpp
source SHA256: b261c23bdc68bc5e4fae0a50f12cefd1aa6a5ef3e40be9ab06802a9389b22e41
binary: /home/amd/project/npu_iron/bh_projcompact_hostopt_20260726
binary SHA256: 36de8cc5e49b34707de0fb27148cb13380c28cbba6eb8c452a80269bf4d910de
```

Expected gain is concentrated in the prior QKV pack (~16 ms), FFN1 pack
(~16 ms), and GELU pack (~56 ms). The exact hardware gain is unmeasured.

## Dynamic-K v5 boundary barrier

Versions v3/v4 executed all K=4 shapes correctly but one 64-row portion of
FFN2 retained K=4 instead of switching to K=20. In v4 the core acquired the RTP
barrier, read the value, and immediately released the barrier before doing the
GEMM. That did not tie the lock lifecycle to completion of the previous
dispatch.

v5 holds the per-core RTP barrier through the entire core iteration. Each core
now performs:

```text
input acquire
RTP barrier acquire
RTP load
GEMM K loop
RTP barrier release to zero
final output release
```

Placing RTP release immediately before final output release means successful
output-DMA completion also proves that all cores returned their RTP barriers to
zero. The following runtime sequence writes the next K value and sets each
barrier to one before starting data DMA.

Commits:

```text
dfe1ea3 fix(iron): hold dynamic-K barrier through output
2e0902d chore(iron): isolate dynamic-K build logs
5f4bb9d test(iron): verify dynamic-K barrier artifacts
19023ae test(iron): stress dynamic-K switch boundaries
44e2698 test(iron): gate dynamic-K v5 after reboot
```

All six shapes compiled successfully under:

```text
/home/amd/project/npu_iron/sam3_attn/shared_gemm_dynamic_rtp_v5
```

Static validation passed for every shape:

- 32 RTP writes;
- 32 core compares using the RTP value;
- 32 barrier acquires;
- 32 barrier releases immediately before final output release;
- non-empty xclbin and instruction stream;
- identical packed core, init, and enable CDOs across all shapes.

Shared hashes:

```text
packed core CDO  2f2866a44bb8efe2015871a65a904a88c044b8784fe97cf9a3a0c87ba533ca74
init CDO         5d728038c7d5c304ade15dad3287a3819be9f9e8610af98b9176798c45eb650d
enable CDO       e3ec26491823b64d42214af6696d1528d8897cf492bea31268c95e172fa6daf0
```

The rebuilt ABI probe uses one common xclbin and runs the original five K=4
shapes, FFN2 K=20, then K=4 and K=20 again. This explicitly exercises both
switch directions and repeats K=20:

```text
probe binary: /home/amd/project/npu_iron/shared_gemm_abi_test_v5
SHA256: ce271237cfb577fc33e1ae8f80f6281b34390e1c50b244e76eeabbae247e1b3b
```

## Prepared gates

The following scripts are prepared but must not run until an on-site physical
reboot and temporary TDR-module reload:

```text
eval/benchmarks/npu_iron/run_dynamic_k_v5_gate.sh
eval/benchmarks/npu_iron/run_projcompact_hostopt_gate.sh
```

The dynamic-K gate runs only eight calls against one xclbin; it is not the
prohibited tight multi-design transition stress. The hostopt gate runs one
backbone frame. Neither gate has been executed.

## Current decision

- Keep projection compaction as the validated performance base.
- Keep hostopt and dynamic-K v5 as offline candidates.
- Do not integrate v5 into the backbone until the ABI boundary sequence passes
  exactly.
- Do not run any NPU command, XRT diagnostic, reset, unload, or remote reboot on
  the current boot.

## Compact FFN shape study

Six static FFN candidates were compiled for three configurations:

| Configuration | Arithmetic vs baseline | FFN1 launches | FFN2 launches | Cores |
|---|---:|---:|---:|---:|
| M1536/H5120 baseline | 100% | 120 | 24 | 32 |
| M1536/H4864 | 95.0% | 114 | 24 | 32 |
| M1408/H4864, tile_m16 | 87.1% | 209 | 44 | 32 |
| M1344/H4864, herd6x4 | 83.1% | 133 | 28 | 24 |

M1408 nearly doubles launch count, while M1344 gives up eight cores. The
M1536/H4864 candidate is therefore the only low-risk option: it preserves the
full 8x4 array and reduces FFN1 launch count plus hidden arithmetic by 5%.

Dynamic-K v5 variants were built for compact FFN1 K=4 and FFN2 K=19. Compact
FFN1 has packed core/init/enable CDOs identical to the common overlay. Compact
FFN2 has identical packed core and enable CDOs but a different init CDO because
K=4864 changes the static DMA/BD initialization. Using compact FFN2 would break
the one-xclbin overlay for a small compute saving, so it was rejected.

The guarded Phase 2 backbone uses:

```text
QKV/O             dynamic-K v5 common overlay
FFN1 M1536/H4864  compact instruction stream on the common overlay
FFN2 M1536/H5120  dynamic-K v5 common overlay
Flash window/global remain separate
```

It refuses to run unless `ALLOW_DYNAMIC_K_V5_WIP=1` is explicitly set, and the
build script has the same guard.

```text
source commit: c2abc62
binary: /home/amd/project/npu_iron/bh_phase2_dynamic_v5_wip
SHA256: 924c782d56ab52353496849afbc9bb70f33e023f6e7962640e45cd14c92a713b
```

The v5 ABI gate now includes compact FFN1 through the common qkv xclbin. Probe
SHA256 is `296210a2226ad1378a4e75cf8b977a42e254ab6db72886be03622703e5268d2c`.

## Driver real-wedge candidate

Upstream history identified two fixes absent from the February driver and the
minimal scheduler-TDR backport:

- `2be0d73`: move `pm_runtime_resume_and_get()` out of scheduler `run_job()`
  and into command submission, releasing the PM reference only during final
  job cleanup. Upstream explicitly describes a deadlock when runtime suspend
  drains the job workqueue while a running job tries to resume the device.
- `0220d14`: guard a NULL mailbox callback handle while flushing a timed-out or
  firmware-wedged channel during teardown.

Both were backported on top of the production TDR source without changing the
UAPI:

```text
workspace: /home/amd/project/amdxdna_tdr_pmfix_20260726
branch: fix/tdr-pm-deadlock
41da564 fix(driver): avoid runtime-PM scheduler deadlock
8308f1d fix(driver): guard mailbox teardown callback
7815ff9 test(driver): stage PM-fix candidate loader
module SHA256: a302f61ed45f3bf6a053c967055d5b9d591d0ec96273caf208137a15d8a61124
checkpatch: 0 errors, 0 warnings
```

This candidate is more relevant to the observed real wedge than another
scheduler timeout tweak, but it is still unvalidated and was not loaded.

## Recovery bundles

Verified bundles were created under the existing quarantine manifest tree:

```text
tracker_phase2_20260726.bundle
  SHA256 32191e364adbf8d7ab3cee1e56b6a08863d80675fbf24cac9891912e17a3164b
amdxdna_tdr_pmfix_20260726.bundle
  SHA256 1d33a74d5460bec16a6f36cf9ab7e99a92197c999ee867a97a73f7320399476f
```
