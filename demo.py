#!/usr/bin/env python3
"""
SAM3 Video Tracker Demo

Tracks an object across frames (or a single image) using a box prompt on
the first frame, then purely memory-based propagation for all subsequent frames.

Usage:
    # Single image (init frame only):
    python demo.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
                   --image assets/truck.jpg --box 100,200,500,600

    # Video file:
    python demo.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
                   --video my_video.mp4 --box 100,200,500,600 \
                   --output tracked.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from tracker import SAM3OnnxTracker, preprocess_image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to downloaded facebook/sam3 model directory")
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Resolution root, e.g. onnx_files_504 or onnx_files_1008. "
                        "Reads <onnx-dir>/backbone_tracker/* and <onnx-dir>/tracker_modules/*.")
    p.add_argument("--imgsz", type=int, default=504,
                   help="Input resolution (504 for speed, 1008 for quality)")
    p.add_argument("--box", type=str, required=True,
                   help="Box prompt in original image coords: x1,y1,x2,y2")
    p.add_argument("--image", type=Path, default=None,
                   help="Single image input")
    p.add_argument("--video", type=Path, default=None,
                   help="Video file input")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Image mode default: outputs/box/<image-stem>_tracked.jpg. "
                        "Video mode default: outputs/box/tracked_output.mp4.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max frames to process from video (0 = all)")
    p.add_argument("--backbone", type=str, default="auto",
                   choices=["auto", "migraphx", "pytorch"],
                   help="Backbone: auto (default), migraphx, or pytorch")
    return p.parse_args()


def overlay_mask(img_bgr, mask, color=(0, 200, 80), alpha=0.45):
    out = img_bgr.copy()
    overlay = np.zeros_like(out)
    overlay[mask] = color
    out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
    return out


def main():
    args = parse_args()
    box = [float(x) for x in args.box.split(",")]

    tracker = SAM3OnnxTracker(
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        backbone=args.backbone,
    )

    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {args.image}")
        h, w = img.shape[:2]
        sx, sy = args.imgsz / w, args.imgsz / h
        box_s = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
        img_np = preprocess_image(img, args.imgsz)

        t0 = time.perf_counter()
        mask, score = tracker.init_frame(img_np, box_s)
        elapsed = (time.perf_counter() - t0) * 1000

        mask_orig = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        vis = overlay_mask(img, mask_orig)
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)

        # Default to outputs/ to avoid overwriting committed assets/docs/.
        # Honor --output if user passes one explicitly.
        if args.output is not None:
            out_path = args.output
        else:
            out_path = Path("outputs/box") / f"{args.image.stem}_tracked.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        print(f"Score: {score:.2f}  Mask: {mask.mean()*100:.1f}%  Latency: {elapsed:.0f}ms")
        print(f"Saved: {out_path}")

    elif args.video:
        cap = cv2.VideoCapture(str(args.video))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        sx, sy = args.imgsz / orig_w, args.imgsz / orig_h
        box_s = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]

        # Default to outputs/ when --output not given (mirrors image-mode behavior).
        out_path = args.output if args.output is not None else Path("outputs/box") / "tracked_output.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (orig_w, orig_h))
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret or (args.max_frames and fi >= args.max_frames):
                break
            img_np = preprocess_image(frame, args.imgsz)
            if fi == 0:
                mask, score = tracker.init_frame(img_np, box_s)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,255), 3)
            else:
                mask, score = tracker.propagate_frame(img_np)
            mask_orig = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            vis = overlay_mask(frame, mask_orig)
            cv2.putText(vis, f"score={score:.1f}  mask={mask.mean()*100:.1f}%",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            writer.write(vis)
            if fi % 20 == 0:
                print(f"  Frame {fi}: score={score:.2f}  mask={mask.mean()*100:.1f}%")
            fi += 1

        cap.release(); writer.release()
        print(f"\nSaved: {out_path}")
        tracker.print_timings()
    else:
        print("Provide --image or --video")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
