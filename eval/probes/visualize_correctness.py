#!/usr/bin/env python3
"""
Run the optimized SAM3 tracker on DAVIS sequences and emit a horizontal grid
showing original + GT mask + predicted mask at evenly-spaced frames, with
per-frame IoU. Lets us visually verify accuracy after the NHWC fix.

Usage:
    python eval/visualize_correctness.py \
        --davis-root /home/amd/project/sam3/dataset/DAVIS \
        --checkpoint /home/amd/project/sam3/model/sam3 \
        --onnx-dir onnx_files \
        --imgsz 504 \
        --sequence blackswan bmx-trees dog \
        --num-samples 6 \
        --output-dir results/correctness_504
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tracker import SAM3OnnxTracker, preprocess_image


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def overlay(img_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
            alpha: float = 0.5) -> np.ndarray:
    out = img_bgr.copy()
    if mask.any():
        layer = np.zeros_like(out)
        layer[mask] = color
        out = cv2.addWeighted(out, 1 - alpha, layer, alpha, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def run_sequence(tracker: SAM3OnnxTracker, davis_root: Path, seq: str,
                 imgsz: int, sample_indices: list[int]) -> tuple[list[np.ndarray], list[float]]:
    jpg_dir = davis_root / "JPEGImages" / "480p" / seq
    ann_dir = davis_root / "Annotations" / "480p" / seq

    jpgs = sorted(jpg_dir.glob("*.jpg"))
    n_frames = len(jpgs)

    # Read first GT annotation, pick object_id=1 (single-object DAVIS)
    ann0 = cv2.imread(str(ann_dir / "00000.png"), cv2.IMREAD_GRAYSCALE)
    if ann0 is None:
        raise FileNotFoundError(ann_dir / "00000.png")
    gt0 = (ann0 > 0)
    if not gt0.any():
        raise ValueError(f"{seq}: empty GT in frame 0")

    h_orig, w_orig = ann0.shape
    bx1, by1, bx2, by2 = bbox_from_mask(gt0)
    sx, sy = imgsz / w_orig, imgsz / h_orig
    box_s = [bx1 * sx, by1 * sy, bx2 * sx, by2 * sy]

    snapshots: dict[int, np.ndarray] = {}
    ious: dict[int, float] = {}

    for fi, jpg in enumerate(jpgs):
        frame = cv2.imread(str(jpg))
        img_np = preprocess_image(frame, imgsz)
        if fi == 0:
            mask, score = tracker.init_frame(img_np, box_s)
        else:
            mask, score = tracker.propagate_frame(img_np)

        if fi in sample_indices:
            ann_path = ann_dir / f"{fi:05d}.png"
            ann = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
            gt = (ann > 0) if ann is not None else np.zeros_like(mask, dtype=bool)

            # Resize pred to original frame size for visualization
            mask_orig = cv2.resize(mask.astype(np.uint8), (w_orig, h_orig),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

            # Compute IoU on ORIGINAL res (compare with GT)
            ious[fi] = iou(mask_orig, gt) if gt.any() else float("nan")

            vis = overlay(frame, gt, (0, 255, 0), alpha=0.25)            # GT green
            vis = overlay(vis, mask_orig, (255, 80, 0), alpha=0.50)      # Pred blue
            cv2.putText(vis, f"f={fi}  IoU={ious[fi]:.2f}  s={score:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            snapshots[fi] = vis

    ordered = [snapshots[i] for i in sample_indices if i in snapshots]
    iou_list = [ious[i] for i in sample_indices if i in ious]
    return ordered, iou_list


def make_grid(snapshots: list[np.ndarray], pad: int = 6) -> np.ndarray:
    if not snapshots:
        raise ValueError("no snapshots")
    h = max(s.shape[0] for s in snapshots)
    w = max(s.shape[1] for s in snapshots)
    pads = []
    for s in snapshots:
        if s.shape[0] != h or s.shape[1] != w:
            s = cv2.copyMakeBorder(s, 0, h - s.shape[0], 0, w - s.shape[1],
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pads.append(s)
    sep = np.zeros((h, pad, 3), dtype=np.uint8)
    row = pads[0]
    for s in pads[1:]:
        row = np.hstack([row, sep, s])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=504)
    ap.add_argument("--sequence", nargs="+", required=True)
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--backbone", default="auto", choices=["auto", "migraphx", "pytorch"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracker = SAM3OnnxTracker(
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        backbone=args.backbone,
    )

    summary = []
    for seq in args.sequence:
        n_frames = len(list((args.davis_root / "JPEGImages/480p" / seq).glob("*.jpg")))
        sample_indices = list(np.linspace(0, n_frames - 1, args.num_samples, dtype=int))
        print(f"\n=== {seq} ({n_frames} frames @ {args.imgsz}px), sampling {sample_indices} ===")

        snapshots, ious = run_sequence(tracker, args.davis_root, seq, args.imgsz, sample_indices)
        grid = make_grid(snapshots)

        out_path = args.output_dir / f"{seq}_{args.imgsz}px.jpg"
        cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mean_iou = float(np.nanmean(ious))
        summary.append((seq, args.imgsz, mean_iou, ious, out_path))
        print(f"  saved {out_path}  mean IoU={mean_iou:.3f}  per-frame={[f'{x:.2f}' for x in ious]}")

    print("\n=== Summary ===")
    for seq, sz, m, ios, p in summary:
        print(f"  {seq:14s} @ {sz}px  mean IoU={m:.3f}  -> {p}")


if __name__ == "__main__":
    raise SystemExit(main())
