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
