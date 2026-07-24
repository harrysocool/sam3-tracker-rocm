# SAM3 shared GEMM overlay — offline design proof

**Date:** 2026-07-24
**Status:** offline build and static-binary comparison complete; hardware test pending a controlled power-cycle

## Goal

Reduce full-array xclbin reconfiguration by using one static 8×4 GEMM design
with multiple runtime instruction streams.

## Common tile

All six production GEMM shapes were rebuilt with:

```text
tile_m=32
tile_n=64
tile_k_l1=64
tile_k_l2=256
herd_m=8
herd_n=4
output=f32
```

Shapes:

```text
qkv_w  2304×1024×3072
qkv_g  1536×1024×3072
o_w    2304×1024×1024
o_g    1536×1024×1024
ffn1   1536×1024×5120
ffn2   1536×5120×1024
```

Artifacts are under:

```text
/home/amd/project/npu_iron/sam3_attn/shared_gemm_candidate_m32n64/
```

## K=1024 static-design identity

The build projects for `qkv_w`, `qkv_g`, `o_w`, `o_g`, and `ffn1` were
preserved and compared byte-for-byte. All five have identical:

- 32 core ELF set;
- core LLVM IR set;
- `matmul_seg_aie_cdo_elfs.bin`;
- `matmul_seg_aie_cdo_init.bin`;
- `matmul_seg_aie_cdo_enable.bin`;
- tile placement and routing.

Only the external-buffer shapes, launch/runtime MLIR, and instruction streams
differ. This is strong offline evidence that one common xclbin can execute all
five K=1024 instruction streams without reconfiguring the AIE array.

## FFN2 difference

FFN2 uses K=5120. Its placement, routing, and CDO init/enable are still the
same, but its packed core ELF differs. LLVM IR comparison shows only one
semantic change in each core:

```text
K=1024: outer K-tile loop limit = 4
K=5120: outer K-tile loop limit = 20
```

Therefore a fully shared six-shape design does not require a new dataflow or
microkernel. It requires making this one loop bound a runtime parameter. The
mlir-aie stack already supports per-core write-RTP buffers (`use_write_rtp` and
`NpuWriteRTPOp`), so the likely implementation is:

1. add a one-element i32 RTP buffer to each of the 32 compute tiles;
2. replace the static outer-loop upper bound with a load from that buffer;
3. write `4` for QKV/O/FFN1 instruction streams and `20` for FFN2;
4. retain the same tile placement, routing, and microkernel.

## Prepared hardware probes

Commit `e955c2f` adds `shared_gemm_abi_test.cpp`. It loads only the common
`qkv_w` xclbin, then executes the five K=1024 instruction streams once each.
Inputs A and B are all BF16 ones; every FP32 output element must equal 1024
within tolerance. No full-array design switching is performed.

Commit `74c780a` adds an end-to-end backbone variant:

- QKV window/global, O window/global, and FFN1 share one xclbin/context;
- FFN2 remains a separate xclbin until RTP work is complete;
- obsolete LayerNorm/QKT/softmax/PV/GELU contexts are not loaded;
- current flash window/global xclbins remain unchanged.

Both programs compile successfully. They have not been run because the current
boot is already in amdxdna D-state.

## Expected first-stage benefit

The K=1024-only shared variant removes the O-projection → FFN1 full-array
transition in every block: 32 transitions per frame. The same-xclbin
microbenchmark measured O projection at 1.188 ms p50, while the backbone's
per-call cost includes approximately 2 ms of design-switch overhead. Expected
gross savings are roughly 60–75 ms before accounting for common-tile throughput
changes.

The full six-shape RTP design would additionally remove FFN1 → FFN2 and
FFN2 → next-block QKV transitions, providing the larger latency and stability
benefit required by the sub-1-second target.

## Safe validation sequence

After a controlled on-site power-cycle:

1. one `xrt-smi` health check only;
2. run `shared_gemm_abi_test_20260724` once;
3. verify all five shapes and process exit;
4. verify NPU open/context cleanup once;
5. run one frame of `bh_shared_k1024_20260724`;
6. stop on any timeout; do not run transition stress;
7. proceed to 5/30 frames only after correctness and device health pass.

## Dynamic-K RTP implementation complete

The lowered placed MLIR was patched automatically rather than attempting to
make the AIR segment DMA loop dynamic. For each of the 32 compute tiles, the
patcher:

1. adds a one-element i32 RTP buffer;
2. loads and index-casts the RTP inside the core;
3. replaces only the outer K-tile loop comparison;
4. writes the desired value at the start of the shape-specific runtime sequence.

Values are:

```text
qkv_w/qkv_g/o_w/o_g/ffn1 → 4
ffn2                      → 20
```

All six patched MLIR files pass `aie-opt` verification and compile through
aiecc to xclbin plus instruction stream. Their generated device artifacts are
byte-identical:

```text
packed ELF CDO  6d61c6e614f794a4da6d1073ac39cf140380ce7d4a4bb9b8a2286f5cad2b69ac
CDO init        ce869aca0b42af248cf3790ca415647a61acc1a52419c4cc1432d1adc44f2246
CDO enable      e3ec26491823b64d42214af6696d1528d8897cf492bea31268c95e172fa6daf0
core ELF set    fc2f7bc91089bff1e68b8a3274cfc9308706d04871bdc86df47bf128747909c7
```

This completes the offline proof that one xclbin can serve all six production
GEMM shapes, including FFN2 K=5120.

Artifacts:

```text
/home/amd/project/npu_iron/sam3_attn/shared_gemm_dynamic_rtp_complete/
```

Commits:

```text
711893b feat(iron): build one dynamic-K GEMM xclbin
e1ab769 perf(iron): share one xclbin across all GEMMs
```

The updated ABI probe executes all six instruction streams against the qkv_w
xclbin. The end-to-end `bh_shared_dynamic_k` host loads only three active AIE
designs per frame: shared GEMM, window flash, and global flash.

## Updated safe validation sequence

After a controlled on-site power-cycle:

1. one `xrt-smi` health check;
2. run `shared_gemm_dynamic_abi_test_20260724` once (six calls total);
3. do not rerun if any call times out or mismatches;
4. confirm context cleanup once;
5. run one frame of `bh_shared_dynamic_k_20260724`;
6. compare cosine and per-stage times;
7. only then run 5 frames; defer 30 frames until device health is reconfirmed;
8. never run the previous tight five-design cycle on XRT 2.21.75.

## Hardware validation results

### Dynamic-K RTP synchronization

The first RTP design loaded the parameter in the core entry block. It produced
all-zero output because cores read the initial value before the runtime sequence
wrote the RTP.

Moving the load after the existing output-buffer acquire fixed a dedicated
single-shape run: all 1,572,864 `o_g` outputs were exactly 1024. A `qkv_w`
xclbin executing the `o_g` instruction stream also passed, proving cross-shape
xclbin/insts interchange.

However, when five K=1024 calls were followed by FFN2 K=5120 in the same
context, one 64-row portion retained the previous K=4 value: those outputs were
1024 instead of 5120. Adding a binary lock and explicitly rearming it did not
remove this final race. The core iterates multiple launch tiles inside one host
dispatch, while the host writes RTP once per dispatch; a correct full dynamic
design needs a dispatch-boundary/launch-count barrier, not a barrier on every
core tile iteration.

The full dynamic-K backbone is therefore still WIP. Its binary was moved to the
quarantine directory and its build script now requires an explicit
`ALLOW_DYNAMIC_K_WIP=1` opt-in.

### Validated K=1024 shared backbone

The safer design shares QKV/O/FFN1 and keeps FFN2 on its existing xclbin. It
passed accuracy and a 30-frame run without driver D-state.

| Metric | Existing baseline | K=1024 shared | Change |
|---|---:|---:|---:|
| wall p50 | 1182.0 ms | 1127.5 ms | -54.5 ms |
| wall p95 | 1422.2 ms | 1368.6 ms | -53.6 ms |
| dispatch p50 | 816.5 ms | 763.0 ms | -53.5 ms |
| dispatch p95 | 1056.6 ms | 1003.3 ms | -53.3 ms |
| host-gap p50 | 363.0 ms | 361.0 ms | -2.0 ms |
| stored-reference cosine | 0.99196 | 0.99196 | unchanged |

Stable stage medians changed approximately as follows:

```text
QKV dispatch   162 → 183 ms  (common n64 tile regression)
FFN1 dispatch  184 → 120 ms  (large improvement)
O dispatch     109 → 108 ms
```

The net gain is real despite the QKV regression. Periodic 200+ ms dispatch
stalls still occur because the frame still switches among shared GEMM, window
flash, global flash, and FFN2 designs. Removing one transition per block lowers
the stable latency but does not eliminate the driver/firmware stall mechanism.

The validated K=1024 shared version is the new optimization base for host
zero-copy and FFN device-chain work.

## Host zero-copy results

Three independent, accuracy-neutral changes were applied on top of the
validated K=1024 shared version:

1. map QKV projection output instead of copying it through `bo.read`;
2. map FlashAttention BF16 output and write the token-major O-projection BF16
   input directly, removing BF16→FP32→BF16 conversion;
3. split QKV, add bias, apply RoPE, and write mapped flash Q/K/V BOs in one
   pass; map O-projection FP32 output directly.

The stored-reference cosine remained `0.99196` throughout.

Representative stable progression:

| Version | Stable wall | Relevant host-stage change |
|---|---:|---|
| K=1024 shared | ~1127 ms | baseline for zero-copy |
| + mapped QKV output | ~1080 ms | `qkv_read` ~43→8–9 ms |
| + direct Flash→O input | ~1043 ms | unpack/layout/repack ~56→10 ms |
| + direct QKV→Flash + mapped O output | 970–997 ms | split removed; H2D ~24→6 ms; O read ~18→2.6 ms |

The final variant completed a 5-frame gate with normal frames:

```text
977, 972, 978, 1190(stall), 970 ms
```

Thus the normal single-frame path is demonstrably below one second, with
unchanged stored-reference cosine.

## 30-frame stability blocker

A subsequent 30-frame run produced correct 970/997/976 ms frames, then blocked
inside NPU wait before frame 3 completed. The safety timeout terminated the
userspace runner; an `amdxdna_js` worker was in D-state. No additional XRT
diagnostic process was started.

This means:

- sub-1-second normal-frame latency is achieved;
- a 30-frame p50/p95 release result is not yet available for the final version;
- current February 2026 amdxdna without merged TDR remains the release blocker;
- further long-run validation should wait for the TDR backport or a matched
  newer driver/plugin/firmware stack.

Do not attribute the D-state specifically to the latest zero-copy change: the
same driver was already shown to wedge under full-array switching stress, and
an `amdxdna_js` D-state worker had accumulated during this boot. The functional
outputs before the hang were correct.
