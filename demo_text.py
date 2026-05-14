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
      --image assets/truck.jpg \\
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
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")

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
    p.add_argument("--min-score", type=float, default=0.5,
                   help="Minimum detection score to track (default 0.5). "
                        "Lower to track weaker detections, raise to filter false positives.")
    p.add_argument("--max-objects", type=int, default=0,
                   help="Max objects to track (default 0 = all above --min-score). "
                        "Objects are ranked by frame-0 detection score.")
    p.add_argument("--imgsz", type=int, default=1008,
                   help="Input resolution (504 or 1008). Must match the .mxr/.onnx "
                        "artefacts under --onnx-dir. Sam3VideoModel defaults to 1008.")
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


# Fixed palette for up to 8 objects (BGR).
_OBJ_COLORS = [
    (0, 200, 80),    # green
    (255, 80, 0),    # blue
    (0, 80, 255),    # red-orange
    (0, 220, 220),   # yellow
    (200, 0, 200),   # magenta
    (0, 200, 255),   # gold
    (180, 255, 100), # lime
    (255, 100, 180), # pink
]


def _to_mask(raw, H: int, W: int) -> np.ndarray:
    """Convert raw tensor/array logits to a boolean HxW mask at original resolution."""
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().float().cpu().numpy()
    m = np.asarray(raw).squeeze() > 0
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def overlay(bgr: np.ndarray,
            objects: list[tuple],
            text: str,
            frame_idx: int | None = None) -> np.ndarray:
    """Render all tracked objects onto one frame.

    objects: list of (mask, score, obj_id) — one entry per tracked object.
             mask can be a torch.Tensor (logits) or np.ndarray (bool/float).
    """
    H, W = bgr.shape[:2]
    vis = bgr.copy()

    for mask_raw, score, obj_id in objects:
        color = _OBJ_COLORS[obj_id % len(_OBJ_COLORS)]
        m = _to_mask(mask_raw, H, W)
        if not m.any():
            continue
        layer = np.zeros_like(bgr)
        layer[m] = color
        vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
        contours, _ = cv2.findContours(m.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        # Label near top of the largest contour bounding box
        if contours:
            x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label_pt = (max(x, 4), max(y - 6, 16))
        else:
            label_pt = (10, 30 + obj_id * 24)
        cv2.putText(vis, f"#{obj_id} {score:.2f}", label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Header: prompt + frame index
    header = f"'{text}'"
    if frame_idx is not None:
        header += f"  f={frame_idx}"
    cv2.putText(vis, header, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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

    # Build config first so we can rewrite image_size BEFORE module __init__
    # bakes derived sizes (backbone_feature_sizes, RoPE, low_res_mask_size).
    # The config's image_size setter cascades to detector + tracker sub-configs;
    # low_res_mask_size has no setter so we patch it manually.
    from transformers import Sam3VideoConfig
    config = Sam3VideoConfig.from_pretrained(str(args.checkpoint))
    if args.imgsz != 1008:
        config.image_size = args.imgsz
        config.low_res_mask_size = 4 * args.imgsz // 14
        # Processor side: image/video processors carry their own size + mask_size
        # which drive pixel_values shape and output mask shape respectively.
        new_size = {"height": args.imgsz, "width": args.imgsz}
        new_mask = {"height": 4 * args.imgsz // 14, "width": 4 * args.imgsz // 14}
        for sub in (getattr(processor, "image_processor", None),
                    getattr(processor, "video_processor", None)):
            if sub is not None:
                if hasattr(sub, "size"):
                    sub.size = new_size
                if hasattr(sub, "mask_size"):
                    sub.mask_size = new_mask
        if hasattr(processor, "target_size"):
            processor.target_size = args.imgsz
        print(f"  config rewritten: image_size={args.imgsz}, "
              f"low_res_mask_size={config.low_res_mask_size}")

    model = (Sam3VideoModel.from_pretrained(str(args.checkpoint), config=config)
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

        # Optional: also MIG-ize the DETR encoder (~245 ms PT → ~80 ms MIG per frame).
        detr_onnx = args.onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx"
        if detr_onnx.exists():
            print(f"Patching detector_model.detr_encoder with MIGraphX shim ...")
            from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder
            patch_sam3_video_model_detr_encoder(model, detr_onnx)
            print(f"  MIG detr_encoder ready")
        # Optional: also MIG-ize memory_attention (steady-state padding)
        mem_attn_onnx = args.onnx_dir / "tracker_modules" / "memory_attention_fixed_S7_P32.onnx"
        if mem_attn_onnx.exists():
            print(f"Patching tracker_model.memory_attention with MIGraphX shim ...")
            from tracker.mig_memory_attention import patch_sam3_video_model_memory_attention
            patch_sam3_video_model_memory_attention(model, mem_attn_onnx)
            print(f"  MIG memory_attention ready (PT fallback for non-steady-state shapes)")
        else:
            print(f"  (skipping detr_encoder MIG: build with "
                  f"export/detector/export_detr_encoder.py to enable)")

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

    # Filter by min-score and sort descending.
    tracked = sorted(
        [i for i in out0.object_ids
         if float(out0.obj_id_to_score.get(i, 0)) >= args.min_score],
        key=lambda i: out0.obj_id_to_score.get(i, 0),
        reverse=True,
    )
    if args.max_objects > 0:
        tracked = tracked[:args.max_objects]
    if not tracked:
        print(f"No detections above --min-score {args.min_score} — try lowering it.")
        return 1
    for obj_id in tracked:
        score = float(out0.obj_id_to_score[obj_id])
        print(f"  object #{obj_id}  score={score:.2f}")
    print(f"  tracking {len(tracked)} object(s)")

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
        objs0 = [(out0.obj_id_to_mask[i], float(out0.obj_id_to_score.get(i, 0)), i)
                 for i in tracked]
        vis = overlay(frames_bgr[0], objs0, args.text)
        cv2.imwrite(str(out_path), vis)
        print(f"Saved: {out_path}")
        return 0

    # Video — propagate frame 1..N-1
    H, W = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    objs0 = [(out0.obj_id_to_mask[i], float(out0.obj_id_to_score.get(i, 0)), i)
             for i in tracked]
    writer.write(overlay(frames_bgr[0], objs0, args.text, frame_idx=0))

    n_total = len(frames_pil)
    print(f"Propagating through frames 1..{n_total - 1} ...")
    t_prop = time.perf_counter()
    for i in range(1, n_total):
        with torch.inference_mode():
            out = model(inference_session=session, frame_idx=i)
        obj_map = out.obj_id_to_mask or {}
        score_map = out.obj_id_to_score or {}
        # Include all objects the model is currently tracking, not just the
        # original frame-0 set. Sam3VideoModel runs detection every frame and
        # adds newly confirmed objects (after hotstart_delay≈15 frames) to
        # out.object_ids automatically — this is re-detection for free.
        all_ids = set(obj_map.keys()) | set(out.object_ids or [])
        objs = [
            (obj_map[j], float(score_map.get(j, 0.0)), j)
            for j in all_ids
            if j in obj_map and (
                j in tracked or float(score_map.get(j, 0.0)) >= args.min_score
            )
        ]
        writer.write(overlay(frames_bgr[i], objs, args.text, frame_idx=i))
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
