#!/usr/bin/env python3
"""SAM3 Tracker baseline eval on SG val subset (50 seqs, seed=42).

Usage:
    PYTHONPATH=repo/DART/.local_deps MIGRAPHX_SKIP_BENCHMARKING=1 \\
        python scripts/onnx/analysis/eval_tracker_baseline.py \\
        [--imgsz 504] [--n-seqs 50] [--seed 42]
        # onnx-dir and out auto-selected from imgsz
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pycocotools import mask as mask_utils
from tracker import SAM3OnnxTracker, preprocess_image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-seqs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imgsz", type=int, default=1008)
    p.add_argument("--gt-json", type=Path,
                   default=WORKSPACE_ROOT / "dataset/gt-annotations/saco_veval_smartglasses_val.json")
    p.add_argument("--img-root", type=Path,
                   default=WORKSPACE_ROOT / "dataset/saco_sg/JPEGImages_6fps")
    p.add_argument("--onnx-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path,
                   default=WORKSPACE_ROOT / "model/sam3")
    p.add_argument("--num-maskmem", type=int, default=7)
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON (default: auto from imgsz)")
    return p.parse_args()


def decode_rle(seg):
    if seg is None:
        return None
    rle = seg if isinstance(seg["counts"], str) else mask_utils.frPyObjects(
        seg, seg["size"][0], seg["size"][1])
    return mask_utils.decode(rle).astype(np.uint8)


def mask_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def main():
    args = parse_args()

    # Auto-select paths based on imgsz
    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE_ROOT / ("onnx_files_1008" if args.imgsz == 1008 else "onnx_files")
    if args.out is None:
        args.out = WORKSPACE_ROOT / f"results/tracker_demo/baseline_50seq_{args.imgsz}px.json"

    with open(args.gt_json) as f:
        gt = json.load(f)

    vid_map = {v["id"]: v for v in gt["videos"]}

    available = {v["video_name"] for v in gt["videos"]
                 if (args.img_root / v["video_name"]).exists()}
    seen, seq_ann_pairs = set(), []
    for ann in gt["annotations"]:
        name = vid_map[ann["video_id"]]["video_name"]
        if name not in available or name in seen or ann["bboxes"][0] is None:
            continue
        seen.add(name)
        seq_ann_pairs.append((name, ann))

    random.seed(args.seed)
    subset = random.sample(seq_ann_pairs, min(args.n_seqs, len(seq_ann_pairs)))
    print(f"Evaluating {len(subset)} sequences at {args.imgsz}px ...")
    print(f"ONNX dir: {args.onnx_dir}")

    tracker = SAM3OnnxTracker(
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        num_maskmem=args.num_maskmem,
    )

    results = []
    t_global = time.perf_counter()

    for si, (seq_name, ann) in enumerate(subset):
        vid = vid_map[ann["video_id"]]
        img_h, img_w = vid["height"], vid["width"]
        sx = args.imgsz / img_w
        sy = args.imgsz / img_h
        bx, by, bw, bh = ann["bboxes"][0]
        box_scaled = [bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy]

        # Reset memory bank between sequences
        tracker.memory_bank._entries = deque(maxlen=args.num_maskmem)
        tracker._frame_idx = 0

        ious_seq, scores_seq = [], []
        t0 = time.perf_counter()

        for fi, fname in enumerate(vid["file_names"]):
            img = cv2.imread(str(args.img_root / fname))
            img_np = preprocess_image(img, args.imgsz)

            if fi == 0:
                mask_pred, score = tracker.init_frame(img_np, box_scaled)
            else:
                mask_pred, score = tracker.propagate_frame(img_np)

            gt_mask = decode_rle(ann["segmentations"][fi])
            if gt_mask is not None and gt_mask.sum() > 0:
                pred_orig = cv2.resize(
                    mask_pred.astype(np.uint8), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST)
                ious_seq.append(mask_iou(pred_orig, gt_mask))
            scores_seq.append(score)

        elapsed = time.perf_counter() - t0
        mean_j = float(np.mean(ious_seq)) if ious_seq else 0.0
        fps_seq = len(vid["file_names"]) / elapsed
        results.append({
            "seq": seq_name, "noun": ann["noun_phrase"],
            "frames": len(vid["file_names"]), "visible_gt": len(ious_seq),
            "mean_iou": mean_j, "mean_score": float(np.mean(scores_seq)),
            "fps": fps_seq,
        })

        elapsed_total = time.perf_counter() - t_global
        eta = elapsed_total / (si + 1) * (len(subset) - si - 1)
        print(f"[{si+1:2d}/{len(subset)}] {seq_name} | {ann['noun_phrase']:28s} | "
              f"J={mean_j:.3f} | {fps_seq:.2f} FPS | ETA {eta/60:.0f}m")

    mean_j_all = float(np.mean([r["mean_iou"] for r in results]))
    mean_fps   = float(np.mean([r["fps"] for r in results]))

    print(f"\n{'='*55}")
    print(f"BASELINE  ({len(subset)} seqs, {args.imgsz}px, seed={args.seed})")
    print(f"  Mean J (IoU):  {mean_j_all:.4f}  ({mean_j_all*100:.1f}%)")
    print(f"  Mean speed:    {mean_fps:.2f} FPS")
    print(f"{'='*55}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "imgsz": args.imgsz,
                "num_maskmem": args.num_maskmem,
                "seed": args.seed,
                "n_seqs": len(subset),
                "backbone": f"Sam3TrackerVideoModel.vision_encoder (PyTorch ROCm FP16, {args.imgsz}px)",
                "memory_attention": f"memory_attention_fixed_N7.onnx (MIGraphX)",
                "onnx_dir": str(args.onnx_dir),
            },
            "mean_j": mean_j_all,
            "mean_fps": mean_fps,
            "sequences": results,
        }, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
