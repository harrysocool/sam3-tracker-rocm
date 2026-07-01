# SAM3 ViT Backbone NPU Offload — Technical Overview

**Platform**: AMD Strix Halo (Ryzen AI Max 395) — XDNA2 NPU (8×4 = 32 AIE2p tiles, 50 TOPS)  
**Target audience**: AIE colleagues who want to understand what we did and how  
**Date**: 2026-06-25

---

## Overview

We offload the SAM3 ViT-L backbone (32 transformer blocks, 1296 tokens @ 504px) to
the XDNA2 NPU using custom MLIR-AIE IRON kernels in **BF16 precision**. Inside the
NPU subprocess the split is **NPU + CPU only** (no GPU context): BF16 matrix operations
(QKV, O-proj, FFN, QKᵀ, Softmax, PV, LayerNorm) run on NPU xclbins, while RoPE, GELU,
residuals, and window partitioning run on CPU (OpenMP + AVX-512). Patch embedding and
the FPN neck run on the GPU (PyTorch) in the main process. No INT8 quantization —
weights and activations stay in BF16 throughout.

> Note: An INT8 variant exists (`bh_npu_backbone`, cos=0.932) but is not the
> primary pipeline. BF16 gives cos=0.989 at similar latency and is the recommended path.

**Bottom line**:
- **Accuracy**: cos = 0.989 vs PyTorch float32 (BF16 rounding only, no quantization error)
- **Power**: ~36.5 W for the NPU backbone subprocess alone (standalone); 39–81 W for the
  full streaming demo (NPU + concurrent GPU tracking), vs 91 W on MIGraphX GPU
- **Role**: background async re-detector in a streaming pipeline where GPU tracking runs
  at 8–10 FPS continuously

---

## 1. NPU / CPU / GPU Split Per ViT Block

Each of the 32 ViT blocks is split as follows.

### On NPU (MLIR-AIE IRON xclbins)

| Operation | Precision | Notes |
|---|---|---|
| LayerNorm 1 & 2 | BF16 | Single xclbin kernel, 64 dispatches/frame |
| QKV linear projection | BF16×BF16→BF16 | Full precision, no quantization |
| QKᵀ (attention score) | BF16 | Window: 64 heads×576 tokens; Global: 16 heads×1344 |
| Softmax | BF16 | Separate xclbin per sequence length |
| PV (attention weighted sum) | BF16 | Output: float32 |
| Output projection (O-proj) | BF16×BF16→BF16 | — |
| FFN1 (two halves in parallel) | BF16×BF16→BF16 | Split across cores for parallelism |
| FFN2 | BF16×BF16→BF16 | — |

**Total NPU dispatches**: 288 per frame (9 kernels × 32 blocks).  
NPU wall time: ~990 ms (dispatch overhead dominates — ~3.4 ms per dispatch on XRT).

### On CPU (host, OpenMP + AVX-512)

| Operation | Precision | Notes |
|---|---|---|
| RoPE (rotary position embedding) | FP32 | Applied to Q and K before NPU attention |
| GELU + bias_add | FP32 | In-place, after FFN1 |
| Residual additions | FP32 | 2 per block |
| Window partition / unpartition | FP32 | Layout rearrangement for windowed attention |
| Head unshuffle | FP32 | Reorder from [G, Sp, d] → [S, C] after attention |
| FP32↔BF16 conversion | — | AVX-512 `_mm512_cvtneps_pbh` at NPU boundary |

### Python GPU side (PyTorch)

- `backbone.embeddings(pixel_values)` — patch embedding + tiled position encoding
- `backbone.layer_norm(tokens_spatial)` — pre-norm before ViT layers (applied before passing to the binary)
- `neck(features)` — FPN decoder after all 32 blocks

---

## 2. Kernel Architecture (MLIR-AIE IRON)

### Xclbin organization

```
/home/amd/project/npu_iron/sam3_attn/
  layernorm/S1296/final.xclbin    # LayerNorm over full 1296-token sequence (padded to 1344)
  proj_mc/qkvproj_w/final.xclbin # QKV projection, window heads
  proj_mc/qkvproj_g/final.xclbin # QKV projection, global heads
  proj_mc/oproj_w/final.xclbin   # O-projection, window
  proj_mc/oproj_g/final.xclbin   # O-projection, global
  ffn_mc/ffn1_half/final.xclbin  # FFN linear 1 (split into 2 halves, multi-core)
  ffn_mc/ffn2/final.xclbin       # FFN linear 2 (multi-core)
  qkt_S576/final.xclbin          # QKᵀ, window attention (S=576)
  sm_S576/final.xclbin           # Softmax, window attention
  pv_S576/final.xclbin           # PV weighted sum, window attention
  qkt_S1296/final.xclbin         # QKᵀ, global attention (S=1296, padded to 1344)
  sm_S1296/final.xclbin          # Softmax, global attention
  pv_S1296/final.xclbin          # PV weighted sum, global attention
```

### Attention split (window vs global)

SAM3 ViT alternates between windowed attention (blocks 0–23, window size 24 → 64 windows of
576 tokens (24×24), 64 heads each) and global attention (blocks 24–31, full 1296-token sequence,
16 heads). We use separate xclbins for each to match the expected sequence lengths.

### Multi-core FFN (FFN speedup 30×)

`ffn_mc/` uses all 8 columns of the 8×4 AIE array with `whole_array` allocation.
FFN1 is split into two halves processed in parallel across columns, giving ~30× speedup
over single-core. FFN2 is parallelized across the array the same way.

### Weight format

Weights are stored as FP32 raw binary in `/home/amd/project/npu_iron/weights/cbb/` (exported by `export_weights_bf16.py`
from the SAM3 model checkpoint) and converted to BF16 at load time via AVX-512
`_mm512_cvtneps_pbh`. No per-tensor or per-token quantization.

---

## 3. GPU+NPU Coexistence on Strix Halo UMA

### The same-process conflict

HIP (via ROCm) and XRT (via XDNA driver) cannot both initialize in the same Python process:
HIP's static constructors reconfigure the IOMMU/MMU, which corrupts XRT's DMA mappings
and causes NPU BO reads to return garbage (constant ~5.6×10¹⁰).

### Solution: subprocess IPC

The NPU binary (`bh_npu_backbone_bf16`) runs as a **persistent child subprocess** with
no CUDA context. The Python main process communicates via **stdin/stdout binary pipe**:

```
Python (CUDA/PyTorch)                    subprocess bh_npu_backbone_bf16 (XRT only)
─────────────────────                    ─────────────────────────────────────────
backbone.embeddings → tokens [1,1296,1024]
stdin ◄── MAGIC(0x0000BF16) + tokens_f32 ─────────────────────────────────────────►
                                                      32× ViT blocks (NPU + CPU)
stdout ◄──────────────────────── MAGIC + features_f32 [1,1296,1024] ◄─────────────
neck(features) → FPN outputs
```

The subprocess stays alive across frames (persistent server mode). Weights (~760 MB BF16)
are loaded once at startup. Each keyframe pays only the binary pipe transfer (~13 MB)
and the 32-block NPU execution (~2.35 s).

### Data transfer (UMA)

Data stays in shared DRAM throughout — no physical copy. The binary pipe is a CPU
memcpy from Python numpy buffer to the subprocess's stdin buffer (both in DRAM).
The subprocess writes XRT BOs from its CPU-side read buffer, then `bo.sync(TO_DEVICE)`
flushes to the NPU's DMA-visible region.

---

## 4. Performance

### Timing breakdown (per frame, ~2290 ms total)

| Stage | Time | % |
|---|---|---|
| NPU dispatch (288 total) | ~990 ms | 43% |
| CPU host — RoPE, GELU, FP32↔BF16 conversion | ~200 ms | 9% |
| XRT BO write + sync | ~200 ms | 9% |
| CPU host — window partition/unpartition, residuals | ~160 ms | 7% |
| Python subprocess + pipe overhead | ~740 ms | 32% |

**Dispatch overhead dominates**: 288 × 3.4 ms/dispatch ≈ 979 ms, leaving only ~1 ms
actual compute time per dispatch for many kernels. This is the fundamental bottleneck on
current XRT — not compute throughput.

### Power

| Implementation | Latency | Avg Power | Peak Power | Energy/frame |
|---|---|---|---|---|
| MIGraphX FP16 (GPU only) | 70 ms | 91 W | 118 W | 6.4 J |
| **BF16 GPU+NPU (this work)** | **2290 ms** | **36.5 W**¹ | **38 W** | **83 J** |

¹ Measured on NPU backbone subprocess standalone. Full system (NPU + concurrent GPU tracking) is 39–81 W.

Power sensor: `hwmon5/power1_input` (GFX die, covers both GPU and NPU).

The 60% power reduction comes from shifting matrix operations from GPU SIMD (high
clock frequency, high power) to NPU systolic array (lower frequency, highly parallel).

### Accuracy

Cosine similarity vs PyTorch float32 reference: **cos = 0.989**

BF16 accumulated rounding error across 32 blocks is small (~0.5% loss). No
quantization artifacts. Remaining gap from 1.0 is inherent BF16 precision (7 mantissa
bits vs 23 for float32).

> INT8 variant (future improvement): cos=0.932, 2.35s — available as
> `bh_npu_backbone` binary but not the primary pipeline.

---

## 5. Integration: Async NPU Streaming Pipeline

The backbone runs too slowly (2.35 s) for real-time use. We integrate it as an
**asynchronous background re-detector** in `demo_npu_parallel.py`:

```
Main thread (GPU):                      NPU thread (subprocess):
─────────────────                       ──────────────────────────
Frame 0: SAM3 detect (MIG, ~2.5s) ──► init box trackers
Frame 1: MIGraphX backbone + ORT       │  NPU backbone (2.35s)
         mem_attn → propagate           │  + MIG DETR (~200ms)
Frame 2: propagate  (~120ms/frame) ←──┘ put result queue
Frame 3: propagate                      starts next frame...
...
Frame N: check result queue ──────────► reinit trackers from NPU detections
         propagate (uninterrupted)
```

**GPU tracker throughput**: 8–10 FPS (thing-class), 4–7 FPS (multi-prompt stuff-class)  
**NPU re-detection interval**: ~3.5 s (NPU backbone + MIG DETR)  
**NPU power during re-detection**: ~36.5 W for backbone subprocess alone (BF16);
  full system (NPU + concurrent GPU tracking) measured at 39–81 W depending on prompts

---

## 6. Known Limitations

### Dispatch overhead
288 dispatches × 3.4 ms = 979 ms overhead, regardless of compute. Reducing dispatch
count via kernel fusion would be the single highest-impact optimization. The S×S
attention score matrix passing through DRAM between QKᵀ and Softmax dispatches is
the key fusion challenge — cores cannot stream S×S back to L2 without a DRAM roundtrip
at current grid sizes, causing the Phase1→Phase2 deadlock we investigated.

### Flash Attention dead end
We attempted Flash Attention (online softmax, tile Q×K/V without materializing S×S).
Single-core: 350 ms/head vs 2.7 ms with 3-dispatch. Root cause: NPU bottleneck is
DMA event count (not bandwidth). Flash requires more DMA events per head, not fewer.
**Do not retry this direction.**

### Stuff-class tracking gaps
NPU re-detection at 3.5 s intervals is too infrequent for fast robot motion with
stuff-class prompts (floor, wall). The SAM2-style box tracker loses amorphous regions
within 5–10 frames. The Hybrid MIG pipeline (1 s keyframe interval, GPU DETR) is
better suited for stuff-class.

---


## 7. Reproducing

Full chain from source to running demo. Each stage depends on the previous.

```
Stage A: Build xclbins   (mlir-aie repo, separate env, hours)   ← skip if pre-built
Stage B: Dump weights    (SAM3 checkpoint → /home/amd/project/npu_iron/weights/vit_full/)
Stage C: Pack weights    (/home/amd/project/npu_iron/weights/vit_full/ → /home/amd/project/npu_iron/weights/cbb/)
Stage D: Compile binary  (C++ → /home/amd/project/npu_iron/bh_npu_backbone_bf16)
Stage E: Run demo
```

On the dev machine, Stages A–D are already done.
Start from Stage E to run the demo, or Stage B if weights need to be regenerated.

---

### Stage A — Build NPU xclbins (one-time, hours)

**Repo**: `github.com/harrysocool/mlir-aie`  **Branch**: `sam3-rope-attention`

```bash
git clone https://github.com/harrysocool/mlir-aie
cd mlir-aie
git checkout sam3-rope-attention

# Set up mlir-aie toolchain in a separate conda env (NOT rocm7p13-sam3)
# Follow mlir-aie README: requires LLVM/MLIR build + peano clang for aie2p target
# First-time toolchain build: 2-4 hours

# Build all SAM3 xclbins from mlir-aie/sam3_attn/:
cd sam3_attn
python layernorm_S1296.py       --target hw   # → layernorm/S1296/final.xclbin
python proj_mc/qkvproj_w.py    --target hw   # → proj_mc/qkvproj_w/final.xclbin
python proj_mc/qkvproj_g.py    --target hw   # → proj_mc/qkvproj_g/final.xclbin
python proj_mc/oproj_w.py      --target hw   # → proj_mc/oproj_w/final.xclbin
python proj_mc/oproj_g.py      --target hw   # → proj_mc/oproj_g/final.xclbin
python ffn_mc/ffn1_half.py     --target hw   # → ffn_mc/ffn1_half/final.xclbin
python ffn_mc/ffn2.py          --target hw   # → ffn_mc/ffn2/final.xclbin
python qkt_S576.py             --target hw   # → qkt_S576/final.xclbin
python qkt_S1296.py            --target hw   # → qkt_S1296/final.xclbin
python sm_S576.py              --target hw   # → sm_S576/final.xclbin
python sm_S1296.py             --target hw   # → sm_S1296/final.xclbin
python pv_S576.py              --target hw   # → pv_S576/final.xclbin
python pv_S1296.py             --target hw   # → pv_S1296/final.xclbin
# Each script takes 10-30 min
```

Copy to deployment machine:
```bash
rsync -av --include='*/' --include='*/final.xclbin' --exclude='*' \
    sam3_attn/ <dev-machine>:/home/amd/project/npu_iron/sam3_attn/
```

---

### Stage B — Dump model weights (one-time per machine)

```bash
cd /home/amd/project/sam3-tracker-rocm
conda activate rocm7p13-sam3
python tools/dump_vit_weights.py --checkpoint model/sam3 --imgsz 504 --out /home/amd/project/npu_iron/weights/vit_full
# ~2 min; writes 518 .npy files (weights for all 32 blocks + RoPE embeddings)
ls /home/amd/project/npu_iron/weights/vit_full/ | wc -l   # expect 518
```

---

### Stage C — Pack weights into binary format

```bash
conda activate rocm7p13-sam3
python tools/export_weights_bf16.py
# Reads /home/amd/project/npu_iron/weights/vit_full/*.npy → writes /home/amd/project/npu_iron/weights/cbb/*.bin (~390 files, BF16 raw binary)
ls /home/amd/project/npu_iron/weights/cbb/ | wc -l   # expect ~390
```

---

### Stage D — Compile C++ backbone binary

Requires: `/home/amd/project/npu_iron/weights/cbb/` (Stage C) and `/home/amd/project/npu_iron/sam3_attn/` (Stage A).
Both paths are hardcoded in the source.

```bash
source /opt/xilinx/xrt/setup.sh
cd /home/amd/project/sam3-tracker-rocm/eval/benchmarks/npu_iron

g++ -O3 -march=native -mavx512f -mavx512bf16 -ffast-math -funroll-loops \
    -fopenmp -std=c++17 \
    backbone_host_bf16_20260617.cpp -o /home/amd/project/npu_iron/bh_npu_backbone_bf16 \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil

# Smoke test — runs all 32 blocks on a reference frame:
OMP_NUM_THREADS=16 /home/amd/project/npu_iron/bh_npu_backbone_bf16
# Expected: "xclbins loaded" → per-block timing → "cos = 0.989"
```

---

### Stage E — Run streaming demo

```bash
cd /home/amd/project/sam3-tracker-rocm
source /opt/xilinx/xrt/setup.sh
conda activate rocm7p13-sam3
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTHONPATH=/opt/rocm-7.2.0/lib:/home/amd/project/sam3/repo/DART/.local_deps:$PYTHONPATH
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0

python demo_npu_parallel.py \
    --checkpoint model/sam3 \
    --onnx-dir onnx_files_504 \
    --video assets/blackswan.mp4 --text swan \
    --output results/demo_npu_$(date +%Y%m%d_%H%M%S).mp4
```

`tracker/npu_backbone_service.py` launches `/home/amd/project/npu_iron/bh_npu_backbone_bf16` automatically
as a persistent subprocess. Watch stderr for `"NPU subprocess ready"` before the
first keyframe fires (~3.5 s on first frame).

