#!/usr/bin/env python3
"""Run the NPU BF16 backbone path on a single image and save a mask overlay
to visually verify correctness.

Usage:
    python eval/probes/npu_single_image_visual.py \
        --image assets/truck.jpg --text truck --imgsz 504
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, '.')

import cv2
import numpy as np

_COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]


def overlay(img_bgr, mask, color, alpha=0.5):
    out = img_bgr.copy()
    if mask.any():
        layer = np.zeros_like(out)
        layer[mask] = color
        out = cv2.addWeighted(out, 1 - alpha, layer, alpha, 0)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='model/sam3')
    p.add_argument('--image', default='assets/truck.jpg')
    p.add_argument('--text', default='truck')
    p.add_argument('--imgsz', type=int, default=504)
    p.add_argument('--out-dir', default='results/npu_iron/visual')
    return p.parse_args()


def main():
    args = parse_args()
    from tracker import rocm_patches  # noqa: F401
    import torch
    from tracker.live_inference import SAM3Live
    from tracker.npu_backbone_service import patch_sam3_with_npu_backbone

    prompts = [p.strip() for p in args.text.split(',')]
    print(f"Loading model @ {args.imgsz}px, prompts={prompts} ...")
    live = SAM3Live(checkpoint=args.checkpoint, prompts=prompts,
                    imgsz=args.imgsz, mig=False, redetect_every=1)
    npu_enc = patch_sam3_with_npu_backbone(live.model, npu_bin=__import__("os").environ.get("NPU_BIN","/home/amd/project/npu_iron/bh_npu_backbone_bf16"))

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    print(f"Image: {img.shape}")

    print("Running NPU detection...")
    t0 = time.perf_counter()
    result = live.infer(img, full_detection=True)
    dt = (time.perf_counter() - t0) * 1000
    tm = npu_enc.timing

    ids = result['object_ids']
    print(f"\nDetected {len(ids)} object(s) in {dt:.0f}ms "
          f"(npu backbone {tm['npu_ms']:.0f}ms)")
    for oid in ids:
        x1,y1,x2,y2 = result['boxes'][oid]
        print(f"  obj {oid}: score={result['scores'][oid]:.3f} "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    # Draw overlays
    vis = img.copy()
    for i, oid in enumerate(ids):
        color = _COLORS[i % len(_COLORS)]
        vis = overlay(vis, result['masks'][oid], color)
        x1,y1,x2,y2 = [int(v) for v in result['boxes'][oid]]
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, f"{result['scores'][oid]:.2f}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    stem = Path(args.image).stem
    out_path = f"{args.out_dir}/{stem}_npu_bf16_{ts}.png"
    cv2.imwrite(out_path, vis)
    print(f"\nSaved: {out_path}")

    npu_enc.shutdown()


if __name__ == '__main__':
    main()
