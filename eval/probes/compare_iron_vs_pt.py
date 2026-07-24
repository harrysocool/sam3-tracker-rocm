#!/usr/bin/env python3
"""Compare IRON NPU backbone vs PyTorch baseline — cosine similarity.

Usage:
    python eval/probes/compare_iron_vs_pt.py --image assets/truck.jpg
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import cv2
from tracker import rocm_patches  # noqa: applies ROCm patches

def cos_stat(a: torch.Tensor, b: torch.Tensor, label: str):
    a = a.float().flatten()
    b = b.float().flatten()
    c = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    print(f"  {label:<42s}  cos = {c:.6f}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='model/sam3')
    p.add_argument('--image',      default='assets/truck.jpg')
    p.add_argument('--imgsz',      type=int, default=504)
    p.add_argument('--npu-bin', default=None,
                   help='optional IRON backbone binary override')
    return p.parse_args()

def main():
    args = parse_args()
    from tracker.live_inference import SAM3Live
    from tracker.npu_backbone_service import patch_sam3_with_npu_backbone

    # ── Load model via SAM3Live (handles all config/resolution correctly) ────
    print(f"Loading model {args.checkpoint} @ {args.imgsz}px ...")
    live = SAM3Live(checkpoint=args.checkpoint, prompts=['x'],
                    imgsz=args.imgsz, mig=False)
    model = live.model
    processor = live.processor

    # ── Preprocess ───────────────────────────────────────────────────────────
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    pv = processor(images=[[img]], return_tensors='pt')['pixel_values']
    pv = pv.to('cuda', model.dtype if hasattr(model, 'dtype') else torch.float16)
    print(f"pixel_values: {tuple(pv.shape)} {pv.dtype}")

    # ── PyTorch baseline ─────────────────────────────────────────────────────
    print("\n[PT] Running PyTorch backbone ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_out = model.detector_model.vision_encoder(pv)
    print(f"     {(time.perf_counter()-t0)*1000:.0f}ms")

    # ── Patch in IRON NPU backbone ───────────────────────────────────────────
    print("\n[NPU] Patching model with IRON NPU backbone ...")
    patch_kwargs = {'npu_bin': args.npu_bin} if args.npu_bin else {}
    npu_enc = patch_sam3_with_npu_backbone(model, **patch_kwargs)

    print("[NPU] Warmup run ...")
    with torch.no_grad():
        _ = model.detector_model.vision_encoder(pv)

    print("[NPU] Timed run ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        npu_out = model.detector_model.vision_encoder(pv)
    print(f"     {(time.perf_counter()-t0)*1000:.0f}ms")

    # ── Cosine comparison ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Cosine similarity: IRON NPU vs PyTorch baseline")
    print("="*60)

    cos_stat(npu_out.last_hidden_state, pt_out.last_hidden_state,
             "last_hidden_state [1,1296,1024]")

    # per-token cos
    npu_t = npu_out.last_hidden_state.squeeze(0).float()
    pt_t  = pt_out.last_hidden_state.squeeze(0).float()
    per_tok = F.cosine_similarity(npu_t, pt_t, dim=-1)
    print(f"  {'per-token cos  mean / min':<42s}  "
          f"{per_tok.mean():.6f} / {per_tok.min():.6f}")

    print()
    if npu_out.fpn_hidden_states and pt_out.fpn_hidden_states:
        for i, (n, p) in enumerate(zip(npu_out.fpn_hidden_states,
                                       pt_out.fpn_hidden_states)):
            cos_stat(n, p, f"FPN p{i+2} {tuple(n.shape)}")
    else:
        print("  (FPN outputs not available — comparing last_hidden_state only)")

    npu_enc.shutdown()
    print("\nDone.")

if __name__ == '__main__':
    main()
