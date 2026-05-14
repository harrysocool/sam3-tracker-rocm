#!/usr/bin/env python3
"""Benchmark single-frame vs propagation pipeline for SAM3 ONNX tracker.

Pipeline A (single-frame): backbone (PyTorch ROCm FP16) + mask_decoder_init (ONNX CPU)
Pipeline B (propagation) : backbone + memory_attention + mask_decoder_propagate + memory_encoder

Usage:
    python eval/bench_pipeline.py \\
        --checkpoint model/sam3 \\
        --onnx-dir onnx_files \\
        --imgsz 504
"""

from __future__ import annotations
import argparse, copy, os, statistics, sys
from pathlib import Path

import cv2, numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
from tracker import SAM3OnnxTracker, preprocess_image

BOX_ORIG = [85, 281, 1710, 850]    # truck box at original 1800×1200


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=WORKSPACE / "model" / "sam3")
    p.add_argument("--onnx-dir",   type=Path, default=WORKSPACE / "onnx_files_504")
    p.add_argument("--image",      type=Path, default=WORKSPACE / "assets" / "demo.jpg")
    p.add_argument("--imgsz",      type=int,  default=504)
    p.add_argument("--warmup",     type=int,  default=8)
    p.add_argument("--runs",       type=int,  default=30)
    return p.parse_args()


def print_row(label, vals, indent=2):
    m = statistics.mean(vals) * 1000
    s = statistics.stdev(vals) * 1000
    print(f"{'':>{indent}}{label:<36}  {m:>7.1f} ± {s:>4.1f} ms")


def clear(tracker):
    tracker._timings = {k: [] for k in tracker._timings}


def main():
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
    args = parse_args()

    img_bgr = cv2.imread(str(args.image))
    H, W    = img_bgr.shape[:2]
    img_np  = preprocess_image(img_bgr, args.imgsz)
    sx, sy  = args.imgsz / W, args.imgsz / H
    box_s   = [BOX_ORIG[0]*sx, BOX_ORIG[1]*sy, BOX_ORIG[2]*sx, BOX_ORIG[3]*sy]

    print(f"\nLoading SAM3OnnxTracker (imgsz={args.imgsz}) ...")
    tracker = SAM3OnnxTracker(
        checkpoint=str(args.checkpoint),
        onnx_dir=str(args.onnx_dir),
        imgsz=args.imgsz,
    )

    # ── Warmup ──────────────────────────────────────────────────────────
    print(f"Warmup ({args.warmup} init_frame passes) ...")
    for _ in range(args.warmup):
        tracker.reset()
        tracker.init_frame(img_np, box_s)
    clear(tracker)

    # ── Pipeline A: single-frame ─────────────────────────────────────────
    print(f"\nPipeline A — single-frame ({args.runs} runs) ...")
    for _ in range(args.runs):
        tracker.reset()
        tracker.init_frame(img_np, box_s)

    bb_a, di_a, tot_a = (tracker._timings[k].copy()
                          for k in ("backbone", "dec_init", "total"))
    clear(tracker)

    # ── Pipeline B: propagation ───────────────────────────────────────────
    # Seed 1 memory entry, then benchmark propagate at steady state (1 slot).
    print(f"Pipeline B — propagation ({args.runs} runs) ...")
    tracker.reset()
    tracker.init_frame(img_np, box_s)
    saved_entries = copy.copy(tracker.memory_bank._entries)
    clear(tracker)

    for _ in range(args.runs):
        tracker.memory_bank._entries = copy.copy(saved_entries)
        tracker._frame_idx = 1
        tracker.propagate_frame(img_np)

    bb_b  = tracker._timings["backbone"].copy()
    ma_b  = tracker._timings["mem_attn"].copy()
    dp_b  = tracker._timings["dec_prop"].copy()
    me_b  = tracker._timings["mem_enc"].copy()
    tot_b = tracker._timings["total"].copy()

    # ── Report ────────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  SAM3 ONNX Tracker — {args.imgsz}px  (gfx1151 / ROCm, n={args.runs})")
    print(sep)

    mean_a = statistics.mean(tot_a) * 1000
    print(f"\n  Pipeline A: Single-frame  (backbone + mask_decoder_init)")
    print_row("backbone  [PyTorch ROCm FP16]", bb_a)
    print_row("mask_decoder_init  [ONNX CPU]", di_a)
    print(f"  {'─'*56}")
    print(f"  {'Total':<38}  {mean_a:>7.1f} ms  →  {1000/mean_a:.2f} FPS")

    mean_b = statistics.mean(tot_b) * 1000
    print(f"\n  Pipeline B: Propagation  (backbone + tracking modules)")
    print_row("backbone  [PyTorch ROCm FP16]", bb_b)
    print_row("memory_attention  [ONNX *]", ma_b)
    print_row("mask_decoder_propagate  [ONNX CPU]", dp_b)
    print_row("memory_encoder  [ONNX CPU]", me_b)
    print(f"  {'─'*56}")
    print(f"  {'Total':<38}  {mean_b:>7.1f} ms  →  {1000/mean_b:.2f} FPS")

    print(f"\n  Propagation overhead vs single-frame:  +{mean_b - mean_a:.1f} ms")
    print(sep + "\n")


if __name__ == "__main__":
    main()
