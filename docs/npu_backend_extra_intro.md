## Part 1 — Why This Approach

### 1.1 Problem
- SAM3 text-prompt pipeline bottleneck: ViT-L backbone, 32 transformer blocks, 1296 tokens @ 504px (14*14 pixels/token)
- MIGraphX GPU path: 70 ms/frame, 91 W — already at the optimization ceiling
- Strix Halo APU has a 50 TOPS XDNA2 NPU completely unused

### 1.2 Why MLIR-AIE IRON, not VitisAI / flexml
- flexml: black-box compiler; hit a segfault on partition1 (191 KB graph), no debug path
- VitisAI EP: deployment-only, must go through flexml — same dead end
- MLIR-AIE IRON: white-box; write kernels directly; has a reference MHA implementation in the mlir-aie repo
- Trade-off: hand-written kernels, complex scheduling — but full control over every dispatch

### 1.3 BF16 vs INT8
- INT8: cos=0.932, needs per-token scale, visible accuracy loss
- BF16: cos=0.989, no quantization, FP32↔BF16 conversion done on CPU with AVX-512 `_mm512_cvtneps_pbh`
- Decision: BF16 is the primary path; INT8 binary exists but is not recommended

---

## Part 2 — Architecture

### 2.1 NPU / GPU / CPU Split Per ViT Block
(walk through the table in overview doc Section 1)

- **NPU xclbin**: matrix ops — QKV proj, O-proj, FFN1/2, QKᵀ, Softmax, LayerNorm
- **CPU / OpenMP + AVX-512**: RoPE, GELU, residual adds, window partition/unpartition, FP32↔BF16 conversion
- **PyTorch GPU**: patch embedding + tiled position encoding (`backbone.embeddings`), FPN neck

Key number: **288 dispatches/frame** (9 kernels × 32 blocks), ~3.4 ms XRT overhead each.
Dispatch overhead is the bottleneck — not compute throughput.

The 3.4 ms overhead is **pure software cost** — AIE tile reconfiguration (new DMA buffer descriptors, new kernel arguments) + completion sync signal. It is independent of how much computation the kernel does; a LayerNorm and a large matmul cost the same to dispatch.

**Why does each CPU operation (RoPE, GELU) force a new dispatch?**
Data never moves — it stays in shared DRAM (XRT buffer objects, UMA). But after RoPE writes its result back to the BO, the *next* NPU kernel must be triggered with a fresh `xrt::run()`, paying the 3.4 ms again. If RoPE had an NPU kernel, QKV proj → RoPE → QKᵀ could be fused into one xclbin and dispatched once. **Kernel fusion is therefore the only lever for reducing this overhead**.

### 2.2 Xclbin Layout
```
/home/amd/project/npu_iron/sam3_attn/   ← separate dir, NOT inside sam3-tracker-rocm repo
  layernorm/S1296/final.xclbin
  proj_mc/{qkvproj_w,qkvproj_g,oproj_w,oproj_g}/final.xclbin
  ffn_mc/{ffn1_half,ffn2}/final.xclbin
  qkt_{S576,S1296}/final.xclbin
  sm_{S576,S1296}/final.xclbin
  pv_{S576,S1296}/final.xclbin
```
Window attention (blocks 0–23) vs global attention (blocks 24–31) use separate xclbins
matched to their sequence lengths (576 and 1296 tokens respectively).

### 2.3 Why Flash Attention Does Not Work
> Flash Attention trades more small matmuls for one large DRAM write — a good deal on GPU
> (SRAM is fast), a bad deal on NPU (each small matmul = one dispatch).

- NPU bottleneck is DMA event count, not bandwidth
- Flash Attention requires more DMA events per head (tiled Q×K/V), not fewer
- Single-core Flash: 350 ms/head vs 2.7 ms with the 3-dispatch path — 130× slower
- **Do not retry this direction**

### 2.4 GPU+NPU Same-Process Conflict and the Subprocess Fix
- HIP static constructors reconfigure the IOMMU → XRT DMA mappings corrupted → NPU BO reads return garbage (~5.6×10¹⁰)
- Solution: run the NPU binary as a child subprocess with no CUDA context; communicate via binary pipe (stdin/stdout)

```
Python main (CUDA/PyTorch)
    │ backbone.embeddings → tokens
    │ ──stdin──► subprocess bh_npu_backbone_bf16  (no CUDA, XRT only)
    │ ◄─stdout── features (CPU f32)
    │ neck(features) → FPN outputs
```

Persistent server mode (process stays alive across frames): keyframe latency 8 s → 2.7 s.

---

## Part 3 — Code Walkthrough

Three files, all committed on `feat/npu-vit-backbone-bf16`:

| File | Role |
|---|---|
| `tracker/npu_backbone_service.py` | Python shim — spawns the C++ server, sends tokens via stdin, reads features from stdout, plugs into `Sam3VideoModel` as a drop-in vision encoder |
| `eval/benchmarks/npu_iron/backbone_host_bf16_20260617.cpp` | C++ server — loads xclbins once, runs the 32-block ViT loop, persistent stdin/stdout binary protocol |
| `demo_npu_parallel.py` | Demo — GPU tracker runs continuously; NPU re-detection runs in background thread every ~3.5 s |

---

## Part 4 — Reproduce

See `docs/npu_backbone_overview.md` Section 8 for full steps.

---

### Where to go next
- Dispatch overhead (288 × 3.4 ms) is the ceiling; kernel fusion is the only lever — blocked by the S×S DRAM roundtrip between QKᵀ and Softmax
- RoPE fusion into attention: attempted, blocked by the public `aiecc` allocator — adding a RoPE tile triggers a global buffer address reassignment that breaks already-validated QKᵀ kernels. AMD's internal Zen-Attention sidesteps this because (a) it targets LLM decode where Q length = 1 (QKᵀ output fits in registers, no DRAM roundtrip), and (b) the internal toolchain has a custom allocator without this constraint. A fix would require either an internal build or a contribution to the aiecc allocator upstream.
- INT8 accuracy (cos 0.932 → 0.97+) needs mixed-precision or calibrated per-tensor scale

---

*File: docs/npu_teaching_agenda.md*
