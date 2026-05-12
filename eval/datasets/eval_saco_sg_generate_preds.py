#!/usr/bin/env python3
"""
Generate predictions for SA-Co/VEval SmartGlasses val in the official
YT-VIS-like format required by saco_veval_eval.py (cgF1 / pHOTA metrics).

Usage:
    python eval/eval_saco_sg_generate_preds.py \
        --gt-json dataset/gt-annotations/saco_veval_smartglasses_val.json \
        --img-root dataset/saco_sg/JPEGImages_6fps \
        --onnx-dir onnx_files \
        --imgsz 504 \
        --out results/saco_sg_val_504px_preds.json

Then run official eval:
    python3 /home/amd/project/sam3-repo-sparse/sam3/eval/saco_veval_eval.py one \
        --gt_annot_file dataset/gt-annotations/saco_veval_smartglasses_val.json \
        --pred_file results/saco_sg_val_504px_preds.json \
        --eval_res_file results/saco_sg_val_504px_eval_res.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")

from tracker import SAM3OnnxTracker, preprocess_image


def encode_mask_rle(mask: np.ndarray, h: int, w: int) -> dict:
    """Encode bool mask (h,w) to RLE."""
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def zero_rle(h: int, w: int) -> dict:
    """Empty mask RLE (all zeros)."""
    zeros = np.zeros((h, w), dtype=np.uint8, order="F")
    rle = mask_utils.encode(zeros)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def bbox_from_rle(rle: dict | None) -> list[float]:
    """[x, y, w, h] from RLE, or [0,0,0,0] if None/empty."""
    if rle is None:
        return [0, 0, 0, 0]
    bb = mask_utils.toBbox(rle)
    return [float(x) for x in bb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-json", type=Path,
                    default=WORKSPACE / "dataset/gt-annotations/saco_veval_smartglasses_val.json")
    ap.add_argument("--img-root", type=Path,
                    default=WORKSPACE / "dataset/saco_sg/JPEGImages_6fps")
    ap.add_argument("--onnx-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=WORKSPACE / "model/sam3")
    ap.add_argument("--imgsz", type=int, default=504)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="Save partial results every N annotations")
    ap.add_argument("--start-from", type=int, default=0,
                    help="Resume from annotation index (skip already-done)")
    args = ap.parse_args()

    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE / (f"onnx_files_{args.imgsz}")
    if args.out is None:
        args.out = WORKSPACE / f"results/saco_sg_val_{args.imgsz}px_preds.json"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out.with_suffix(".partial.json")

    print(f"GT:        {args.gt_json}")
    print(f"Images:    {args.img_root}")
    print(f"ONNX dir:  {args.onnx_dir}")
    print(f"imgsz:     {args.imgsz}px")
    print(f"Output:    {args.out}")

    with open(args.gt_json) as f:
        gt = json.load(f)

    vid_map = {v["id"]: v for v in gt["videos"]}
    anns = gt["annotations"]
    print(f"\nTotal annotations: {len(anns)}")
    print(f"Total videos:      {len(gt['videos'])}")

    # Load partial results if resuming
    predictions = []
    if args.start_from > 0 and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            predictions = json.load(f)
        print(f"Resuming from annotation {args.start_from}, "
              f"loaded {len(predictions)} existing predictions")

    tracker = SAM3OnnxTracker(
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
    )

    t_global = time.perf_counter()
    n_total = len(anns)

    for si, ann in enumerate(anns):
        if si < args.start_from:
            continue

        vid = vid_map[ann["video_id"]]
        img_h, img_w = ann["height"], ann["width"]
        n_frames = len(vid["file_names"])
        bboxes_gt = ann["bboxes"]           # list of [x,y,w,h] or [0,0,0,0]
        segs_gt = ann["segmentations"]       # list of RLE or None

        # Find first frame with a valid GT bbox
        init_fi = None
        for fi, bb in enumerate(bboxes_gt):
            if bb is not None and len(bb) == 4 and bb[2] > 0 and bb[3] > 0:
                init_fi = fi
                break

        if init_fi is None:
            # No valid frame found — yield empty prediction
            pred_segs = [zero_rle(img_h, img_w)] * n_frames
            pred_bboxes = [[0, 0, 0, 0]] * n_frames
            pred_areas = [0] * n_frames
            pred_score = 0.0
        else:
            # Build per-frame prediction
            pred_segs = [None] * n_frames
            pred_bboxes = [[0, 0, 0, 0]] * n_frames
            pred_areas = [0] * n_frames
            scores = []

            # Frames before init: zero prediction
            for fi in range(init_fi):
                pred_segs[fi] = zero_rle(img_h, img_w)

            # Init frame
            bb_gt = bboxes_gt[init_fi]  # [x, y, w, h]
            x1_s = bb_gt[0] * args.imgsz / img_w
            y1_s = bb_gt[1] * args.imgsz / img_h
            x2_s = (bb_gt[0] + bb_gt[2]) * args.imgsz / img_w
            y2_s = (bb_gt[1] + bb_gt[3]) * args.imgsz / img_h
            box_scaled = [x1_s, y1_s, x2_s, y2_s]

            fnames = vid["file_names"]
            img_path = args.img_root / fnames[init_fi]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                # Image missing — zero prediction for remaining frames
                for fi in range(init_fi, n_frames):
                    pred_segs[fi] = zero_rle(img_h, img_w)
                pred_score = 0.0
            else:
                img_np = preprocess_image(img_bgr, args.imgsz)
                mask, score = tracker.init_frame(img_np, box_scaled)
                scores.append(score)
                mask_orig = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                pred_segs[init_fi] = encode_mask_rle(mask_orig, img_h, img_w)
                x, y, w, h = cv2.boundingRect(mask_orig.astype(np.uint8))
                pred_bboxes[init_fi] = [float(x), float(y), float(w), float(h)]
                pred_areas[init_fi] = int(mask_orig.sum())

                # Propagation frames
                for fi in range(init_fi + 1, n_frames):
                    img_path = args.img_root / fnames[fi]
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        pred_segs[fi] = zero_rle(img_h, img_w)
                        continue
                    img_np = preprocess_image(img_bgr, args.imgsz)
                    mask, score = tracker.propagate_frame(img_np)
                    scores.append(score)
                    mask_orig = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    pred_segs[fi] = encode_mask_rle(mask_orig, img_h, img_w)
                    if mask_orig.any():
                        x, y, w, h = cv2.boundingRect(mask_orig.astype(np.uint8))
                        pred_bboxes[fi] = [float(x), float(y), float(w), float(h)]
                    pred_areas[fi] = int(mask_orig.sum())

                pred_score = float(np.mean(scores)) if scores else 0.0

            # Fill any remaining Nones
            for fi in range(n_frames):
                if pred_segs[fi] is None:
                    pred_segs[fi] = zero_rle(img_h, img_w)

        predictions.append({
            "video_id": ann["video_id"],
            "category_id": ann["category_id"],
            "segmentations": pred_segs,
            "bboxes": pred_bboxes,
            "areas": pred_areas,
            "score": pred_score,
        })

        # Progress
        elapsed = time.perf_counter() - t_global
        eta = elapsed / (si - args.start_from + 1) * (n_total - si - 1)
        noun = ann["noun_phrase"][:24]
        print(f"[{si+1:4d}/{n_total}] vid={ann['video_id']:4d} "
              f"{noun:24s} | init_fi={init_fi} | score={pred_score:.2f} "
              f"| ETA {eta/3600:.1f}h")

        # Checkpoint
        if (si + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(predictions, f)
            print(f"  [checkpoint] {len(predictions)} predictions saved")

    with open(args.out, "w") as f:
        json.dump(predictions, f)
    print(f"\nSaved {len(predictions)} predictions → {args.out}")

    # Cleanup partial checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
