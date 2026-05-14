#!/usr/bin/env python3
"""
Stage 1 probe: SAM3 text → mask on a single image, using the official
Sam3VideoModel pipeline (detector + tracker stitched as Meta intends).

Goal: verify that text-prompt segmentation works end-to-end on this stack
before deciding whether to wire it into the optimized tracker.

Verifies:
1. model.safetensors loads correctly via Sam3VideoModel
2. text → detection → mask path works end-to-end
3. Per-stage timing for an init frame

Requires:
    PYTHONPATH=/home/amd/project/sam3/repo/DART/.local_deps  (for Sam3VideoModel)
    HSA_OVERRIDE_GFX_VERSION=11.5.1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import Sam3VideoModel, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/home/amd/project/sam3/model/sam3"))
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True,
                    help='Text prompt, e.g. "swan", "a person on a bicycle"')
    ap.add_argument("--output", type=Path, default=Path("text_probe_out.jpg"))
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    print(f"\n[1/4] Loading Sam3VideoModel + AutoProcessor from {args.checkpoint} ...")
    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = Sam3VideoModel.from_pretrained(str(args.checkpoint))
    model = model.to(device).to(dtype).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"      load took {time.perf_counter() - t:.1f}s")

    img_pil = Image.open(args.image).convert("RGB")
    img_bgr = cv2.imread(str(args.image))
    H, W = img_pil.height, img_pil.width
    print(f"\n[2/4] Image: {W}x{H}, text='{args.text}'")

    # Init session — pass image as a 1-frame "video"
    t = time.perf_counter()
    session = processor.init_video_session(
        video=[img_pil],
        inference_device=device,
        dtype=dtype,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"      init_video_session: {(time.perf_counter() - t) * 1000:.1f}ms")

    t = time.perf_counter()
    processor.add_text_prompt(session, args.text)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"      add_text_prompt:    {(time.perf_counter() - t) * 1000:.1f}ms")

    print(f"\n[3/4] Running detector + tracker on frame 0 ...")
    # Warm-up not needed for probe — we want the cold-start number too
    t = time.perf_counter()
    with torch.inference_mode():
        out = model(inference_session=session, frame_idx=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first = time.perf_counter() - t
    print(f"      first call (cold):  {t_first * 1000:.1f}ms")

    # Second call (warm) — same frame, should be much faster (cache hit on vision feats)
    t = time.perf_counter()
    with torch.inference_mode():
        out2 = model(inference_session=session, frame_idx=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"      second call (warm): {(time.perf_counter() - t) * 1000:.1f}ms")

    print(f"\n[4/4] Result:")
    print(f"      object_ids: {out.object_ids}")
    print(f"      scores:     {out.obj_id_to_score}")

    if not out.object_ids:
        print("\n      No detections. Try a different prompt or lower confidence threshold.")
        return 1

    # Visualize
    vis = img_bgr.copy()
    palette = [(0, 200, 80), (200, 80, 0), (80, 0, 200),
               (200, 200, 0), (0, 200, 200), (200, 0, 200)]
    for i, obj_id in enumerate(out.object_ids):
        mask = out.obj_id_to_mask[obj_id]
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().float().cpu().numpy()
        mask = mask.squeeze()
        # Sam3VideoModel returns logits (e.g. shape 288x288, range ~[-16, +6]).
        # Threshold at 0 first, THEN resize to original res — resizing logits
        # before binarising would smear values across foreground/background.
        mask = (mask > 0)
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        score = float(out.obj_id_to_score.get(obj_id, 0.0))
        color = palette[i % len(palette)]

        layer = np.zeros_like(vis); layer[mask] = color
        vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        if contours:
            x, y, _w, _h = cv2.boundingRect(contours[0])
            cv2.putText(vis, f"#{obj_id} {score:.2f}", (x, max(y - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(vis, f"text='{args.text}'  N={len(out.object_ids)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.imwrite(str(args.output), vis)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
