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
