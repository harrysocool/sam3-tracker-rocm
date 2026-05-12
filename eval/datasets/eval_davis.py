#!/usr/bin/env python3
"""DAVIS 2017 semi-supervised evaluation for SAM3 ONNX tracker.

For each val sequence:
  - For each object (obj_id 1,2,...), derive box from frame-0 GT mask
  - Track with our ONNX pipeline (box prompt init, memory propagation)
  - Compute J (IoU) per frame against GT mask
Reports: per-sequence J, overall J mean, FPS

Usage:
    PYTHONPATH=repo/DART/.local_deps MIGRAPHX_SKIP_BENCHMARKING=1 \\
        python scripts/onnx/analysis/eval_davis.py \\
        [--imgsz 1008] [--davis dataset/DAVIS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tracker import SAM3OnnxTracker, preprocess_image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imgsz", type=int, default=1008)
    p.add_argument("--davis", type=Path,
                   default=WORKSPACE_ROOT / "dataset/DAVIS")
    p.add_argument("--split", default="val",
                   help="val or test-dev")
    p.add_argument("--onnx-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path,
                   default=WORKSPACE_ROOT / "model/sam3")
    p.add_argument("--num-maskmem", type=int, default=7)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def mask_to_box(mask: np.ndarray):
    """Convert binary mask to [x1,y1,x2,y2] bounding box. Returns None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0  # 1.0 if both empty


def main():
    args = parse_args()

    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE_ROOT / f"onnx_files_{args.imgsz}"
    if args.out is None:
        args.out = WORKSPACE_ROOT / f"results/tracker_demo/davis_{args.split}_{args.imgsz}px.json"

    img_dir  = args.davis / "JPEGImages" / "480p"
    ann_dir  = args.davis / "Annotations" / "480p"
    seq_list = (args.davis / "ImageSets" / "2017" / f"{args.split}.txt").read_text().strip().split()

    print(f"DAVIS 2017 {args.split}: {len(seq_list)} sequences at {args.imgsz}px")
    print(f"ONNX dir: {args.onnx_dir}")

    tracker = SAM3OnnxTracker(
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        num_maskmem=args.num_maskmem,
    )

    all_results = []
    t_global = time.perf_counter()

    for si, seq_name in enumerate(seq_list):
        frames = sorted((img_dir / seq_name).glob("*.jpg"))
        n_frames = len(frames)

        # Load frame-0 GT to find object IDs
        ann0 = np.array(Image.open(ann_dir / seq_name / "00000.png"))
        obj_ids = [oid for oid in np.unique(ann0) if oid > 0]
        img_h, img_w = ann0.shape

        sx = args.imgsz / img_w
        sy = args.imgsz / img_h

        seq_j_per_obj = []

        for obj_id in obj_ids:
            # Derive box from frame-0 GT mask
            gt_mask0 = (ann0 == obj_id)
            box_orig = mask_to_box(gt_mask0)
            if box_orig is None:
                continue
            box_scaled = [box_orig[0]*sx, box_orig[1]*sy,
                          box_orig[2]*sx, box_orig[3]*sy]

            # Reset tracker memory
            tracker.memory_bank._entries = deque(maxlen=args.num_maskmem)
            tracker._frame_idx = 0

            ious_obj = []
            t0 = time.perf_counter()

            for fi, fpath in enumerate(frames):
                img_bgr = cv2.imread(str(fpath))
                img_np  = preprocess_image(img_bgr, args.imgsz)

                if fi == 0:
                    mask_pred, score = tracker.init_frame(img_np, box_scaled)
                else:
                    mask_pred, score = tracker.propagate_frame(img_np)

                # GT mask for this frame & object
                ann_fi = np.array(Image.open(ann_dir / seq_name / fpath.with_suffix(".png").name))
                gt_fi = (ann_fi == obj_id)

                # Resize pred to original resolution
                pred_orig = cv2.resize(
                    mask_pred.astype(np.uint8), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST).astype(bool)

                ious_obj.append(mask_iou(pred_orig, gt_fi))

            obj_j = float(np.mean(ious_obj)) if ious_obj else 0.0
            seq_j_per_obj.append(obj_j)

        seq_j = float(np.mean(seq_j_per_obj)) if seq_j_per_obj else 0.0
        elapsed = time.perf_counter() - t0
        fps_seq = n_frames / elapsed

        all_results.append({
            "seq": seq_name, "n_objects": len(obj_ids),
            "frames": n_frames, "mean_j": seq_j, "fps": fps_seq,
        })

        elapsed_total = time.perf_counter() - t_global
        eta = elapsed_total / (si + 1) * (len(seq_list) - si - 1)
        print(f"[{si+1:2d}/{len(seq_list)}] {seq_name:25s} | "
              f"J={seq_j:.3f} | objs={len(obj_ids)} | "
              f"{fps_seq:.2f} FPS | ETA {eta/60:.0f}m")

    mean_j   = float(np.mean([r["mean_j"] for r in all_results]))
    mean_fps = float(np.mean([r["fps"] for r in all_results]))

    print(f"\n{'='*55}")
    print(f"DAVIS 2017 {args.split}  ({args.imgsz}px)")
    print(f"  Mean J (IoU):   {mean_j:.4f}  ({mean_j*100:.1f}%)")
    print(f"  Mean speed:     {mean_fps:.2f} FPS")
    print(f"{'='*55}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "dataset": "DAVIS2017-val",
                "imgsz": args.imgsz,
                "num_maskmem": args.num_maskmem,
                "onnx_dir": str(args.onnx_dir),
                "backbone": f"Sam3TrackerVideoModel.vision_encoder (PyTorch ROCm FP16, {args.imgsz}px)",
            },
            "mean_j": mean_j,
            "mean_fps": mean_fps,
            "sequences": all_results,
        }, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
