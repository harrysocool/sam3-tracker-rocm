#!/usr/bin/env python3
"""Text-prompt demo for SAM3 — single image or any mp4 video.

Uses Sam3VideoModel (PyTorch + CLIP) end-to-end. Slower than the box-prompt
demo.py path (which is MIGraphX-accelerated) but supports open-vocabulary
text prompts like "swan", "person on a bike", etc.

Pipeline:
  Frame 0:   text → CLIP detection → mask (highest-scoring object kept)
  Frames 1+: SAM3 video tracker propagates that mask through subsequent frames

Usage:
  # Single image
  python demo_text.py \\
      --checkpoint model/sam3 \\
      --image assets/demo.jpg \\
      --text "truck"

  # Video (any mp4)
  python demo_text.py \\
      --checkpoint model/sam3 \\
      --video docs/images/demo_blackswan_mxr_504.mp4 \\
      --text "swan" \\
      --max-frames 60

Output goes to outputs/text/<input-stem>_text.{jpg,mp4} unless --output is given.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import Sam3VideoModel, AutoProcessor


# ─── Args ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to model/sam3 (containing model.safetensors)")
    p.add_argument("--text", type=str, required=True,
                   help='Text prompt, e.g. "swan", "a person on a bicycle"')
    p.add_argument("--image", type=Path, default=None,
                   help="Single image input (mutually exclusive with --video)")
    p.add_argument("--video", type=Path, default=None,
                   help="Video input — any mp4 readable by OpenCV")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Default: outputs/text/<input-stem>_text.{jpg,mp4}")
    p.add_argument("--max-frames", type=int, default=120,
                   help="Cap video frames loaded into the session (default 120)")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--mig", action="store_true",
                   help="Use MIGraphX backbone for vision_encoder (≈2× faster init "
                        "and per-frame backbone). Reads <onnx-dir>/backbone_detector/tuned.mxr "
                        "(build with: python export/backbone/export_backbone_single.py "
                        "--backbone-source detector + simplify + compile).")
    p.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_1008"),
                   help="Resolution root (e.g. onnx_files_1008). The MIG path reads "
                        "<onnx-dir>/backbone_detector/{single_simplified.onnx,tuned.mxr}.")
    args = p.parse_args()
    if (args.image is None) == (args.video is None):
        sys.exit("Pass exactly one of --image or --video")
    return args


# ─── Helpers ───────────────────────────────────────────────────────────────
def load_video_frames(path: Path, max_n: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames_pil, frames_bgr = [], []
    while len(frames_pil) < max_n:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
        frames_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames_pil, frames_bgr, fps


def overlay(bgr: np.ndarray, mask, score: float, text: str,
            frame_idx: int | None = None,
            color=(0, 200, 80)) -> np.ndarray:
    H, W = bgr.shape[:2]
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask).squeeze()
    # Sam3VideoModel returns logits — threshold at 0 first, THEN resize to
    # original res (resizing logits would smear FG/BG values).
    mask = mask > 0
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    vis = bgr.copy()
    if mask.any():
        layer = np.zeros_like(bgr)
        layer[mask] = color
        vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
    label = f"text='{text}'  score={score:.2f}"
    if frame_idx is not None:
        label += f"  f={frame_idx}"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    return vis


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    print(f"Loading Sam3VideoModel from {args.checkpoint} ...")
    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = (Sam3VideoModel.from_pretrained(str(args.checkpoint))
             .to(device).to(dtype).eval())
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  loaded in {time.perf_counter() - t:.1f}s")

    if args.mig:
        print(f"Patching detector_model.vision_encoder with MIGraphX backbone ...")
        t = time.perf_counter()
        from tracker.tracker import MIGraphXBackbone
        from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
        det_dir = args.onnx_dir / "backbone_detector"
        mxr = MIGraphXBackbone(
            onnx_path=det_dir / "single_simplified.onnx",
            cache_path=det_dir / "tuned.mxr",
        )
        mxr.warmup(n=2)
        patch_sam3_video_model_with_mig(model, mxr)
        print(f"  MIG backbone ready in {time.perf_counter() - t:.1f}s")

    # Collect frames
    if args.image is not None:
        bgr0 = cv2.imread(str(args.image))
        if bgr0 is None:
            sys.exit(f"Cannot read image: {args.image}")
        frames_bgr = [bgr0]
        frames_pil = [Image.fromarray(cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB))]
        fps = 1.0
    else:
        print(f"Loading video frames (max {args.max_frames}) ...")
        frames_pil, frames_bgr, fps = load_video_frames(args.video, args.max_frames)
        if not frames_pil:
            sys.exit(f"No frames decoded from {args.video}")
        print(f"  loaded {len(frames_pil)} frames @ ~{fps:.1f} fps")

    # Init session + prompt
    print("Initialising session ...")
    t = time.perf_counter()
    session = processor.init_video_session(
        video=frames_pil, inference_device=device, dtype=dtype,
    )
    processor.add_text_prompt(session, args.text)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  ready in {(time.perf_counter() - t) * 1000:.0f} ms")

    # Frame 0 — detection + tracker init
    print(f"Detecting '{args.text}' on frame 0 ...")
    t = time.perf_counter()
    with torch.inference_mode():
        out0 = model(inference_session=session, frame_idx=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t) * 1000
    n_obj = len(out0.object_ids)
    print(f"  init: {init_ms:.0f} ms  →  {n_obj} object(s) detected")
    if n_obj == 0:
        print("No detections — try a different prompt.")
        return 1
    primary = max(out0.object_ids, key=lambda i: out0.obj_id_to_score.get(i, 0))
    score0 = float(out0.obj_id_to_score[primary])
    mask0 = out0.obj_id_to_mask[primary]
    print(f"  primary: object #{primary}  score={score0:.2f}")

    # Output path
    if args.output is not None:
        out_path = args.output
    elif args.image is not None:
        out_path = Path("outputs/text") / f"{args.image.stem}_text.jpg"
    else:
        out_path = Path("outputs/text") / f"{args.video.stem}_text.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Single image — done
    if args.image is not None:
        vis = overlay(frames_bgr[0], mask0, score0, args.text)
        cv2.imwrite(str(out_path), vis)
        print(f"Saved: {out_path}")
        return 0

    # Video — propagate frame 1..N-1
    H, W = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    writer.write(overlay(frames_bgr[0], mask0, score0, args.text, frame_idx=0))

    n_total = len(frames_pil)
    print(f"Propagating through frames 1..{n_total - 1} ...")
    t_prop = time.perf_counter()
    for i in range(1, n_total):
        with torch.inference_mode():
            out = model(inference_session=session, frame_idx=i)
        if primary in (out.obj_id_to_mask or {}):
            mask = out.obj_id_to_mask[primary]
            score = float(out.obj_id_to_score.get(primary, 0.0))
        else:
            mask = np.zeros((H, W), dtype=bool)
            score = 0.0
        writer.write(overlay(frames_bgr[i], mask, score, args.text, frame_idx=i))
        if i % 20 == 0:
            elapsed = time.perf_counter() - t_prop
            print(f"  frame {i}/{n_total - 1}  ({i / elapsed:.1f} prop FPS)")
    if device.type == "cuda":
        torch.cuda.synchronize()
    writer.release()

    elapsed = time.perf_counter() - t_prop
    n_prop = n_total - 1
    print(f"\nPropagation: {n_prop} frames in {elapsed:.1f}s = {n_prop / elapsed:.2f} FPS")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
