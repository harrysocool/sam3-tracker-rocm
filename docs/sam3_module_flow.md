# SAM3 Module Flow

```mermaid
flowchart TD
    classDef gpu  fill:#1a73e8,stroke:#0d47a1,color:#fff
    classDef npu  fill:#e65100,stroke:#bf360c,color:#fff
    classDef cpu  fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef ort  fill:#6a1b9a,stroke:#4a148c,color:#fff
    classDef io   fill:#37474f,stroke:#263238,color:#fff
    classDef mem  fill:#f57f17,stroke:#e65100,color:#fff

    VIDEO["📹 Video Frame\nH×W×3 BGR"]:::io
    TEXTPROMPT["💬 Text Prompt\ne.g. 'swan'"]:::io
    BOXPROMPT["⬜ Box Prompt\n[x1,y1,x2,y2]"]:::io

    %% ── Backbone ──────────────────────────────────────────
    subgraph BB["① Backbone  —  per frame"]
        EMBED["Patch Embed + Positional Enc\n14×14 patches → 36×36\ntokens: 1×1296×1024\n━━━━━━━━━━━━━━━\nGPU · PyTorch"]:::gpu

        subgraph NPUSUB["NPU subprocess  (bh_npu_backbone_bf16)"]
            LN["LayerNorm\nNPU xclbin"]:::npu
            PROJ["QKV Projection\nNPU proj_mc xclbin"]:::npu
            ROPE["RoPE Encoding\nCPU · AVX-512"]:::cpu
            ATTN["Attention  QKᵀ → Softmax → PV\n3× NPU xclbin\nblocks 0–23: S=576  blocks 24–31: S=1296"]:::npu
            OPROJ["O-Projection + FFN (GELU)\nNPU proj_mc / ffn_mc xclbin"]:::npu
        end

        NECK["FPN Neck\n4-level pyramid\n[256×36², 256×18², 256×9², 256×5²]\n━━━━━━━━━━━━━━━\nGPU · PyTorch"]:::gpu
    end

    %% ── Text Detector ──────────────────────────────────────
    subgraph DET["② Text Detector  —  keyframe only  (text-prompt pipeline)"]
        CLIP["CLIP Text Encoder\nGPU · PyTorch"]:::gpu
        DENC["DETR Encoder\nORT MIG EP"]:::ort
        DDEC["DETR Decoder\nGPU · PyTorch"]:::gpu
        DBOX["Detection Boxes + Scores"]:::io
    end

    %% ── Tracker Init ───────────────────────────────────────
    subgraph TINIT["③ Tracker Init  —  frame 0 / keyframe"]
        MDINIT["mask_decoder_init\nORT CPU"]:::ort
        MENC0["memory_encoder\nORT CPU"]:::ort
    end

    MEMBANK[("MemoryBank\nFIFO · max 7 frames\nCPU tensor")]:::mem

    %% ── Tracker Propagate ──────────────────────────────────
    subgraph TPROP["④ Tracker Propagate  —  frames 1+  (loop per object)"]
        MEMATT["memory_attention\nORT MIG EP\n~17 ms / object"]:::ort
        MDPROP["mask_decoder_propagate\nORT CPU"]:::ort
        MENCN["memory_encoder\nORT CPU"]:::ort
    end

    OUT["🎭 Masks + Bounding Boxes\nper object · per frame"]:::io

    %% ── Edges ──────────────────────────────────────────────
    VIDEO      --> EMBED
    EMBED      --> LN --> PROJ --> ROPE --> ATTN --> OPROJ --> LN
    OPROJ      --> NECK

    %% text path
    TEXTPROMPT --> CLIP
    NECK       --> DENC
    DENC       --> DDEC
    CLIP       --> DDEC
    DDEC       --> DBOX
    DBOX       --> MDINIT

    %% box path
    BOXPROMPT  --> MDINIT

    %% init
    NECK       --> MDINIT
    MDINIT     --> MENC0
    MENC0      --> MEMBANK

    %% propagate loop
    MEMBANK    --> MEMATT
    NECK       --> MEMATT
    MEMATT     --> MDPROP
    NECK       --> MDPROP
    MDPROP     --> OUT
    MDPROP     --> MENCN
    MENCN      --> MEMBANK
```

## Hardware Legend

| Color | Hardware | Modules |
|---|---|---|
| 🔵 Blue | **GPU · PyTorch** | Patch embed, FPN neck, CLIP encoder, DETR decoder |
| 🟠 Orange | **NPU BF16** | LayerNorm, QKV proj, Attention, FFN inside each ViT block |
| 🟢 Green | **CPU · AVX-512** | RoPE, GELU, residual add, FP32↔BF16 conversion |
| 🟣 Purple | **ORT MIG EP / CPU** | DETR encoder, memory_attention, mask_decoder_* |
| 🟡 Yellow | **CPU memory** | MemoryBank FIFO (up to 7 frames) |

## Pipeline Comparison

| | Box-prompt | Text-prompt |
|---|---|---|
| Frame 0 input | Manual box `[x1,y1,x2,y2]` | CLIP + DETR auto-detection |
| Backbone | MIGraphX GPU `.mxr` | NPU BF16 subprocess |
| Keyframe | None (frame 0 only) | Async NPU re-detection every ~3.5 s |
| Propagation | Identical ↑ | Identical ↑ |
