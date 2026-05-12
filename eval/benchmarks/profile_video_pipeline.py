#!/usr/bin/env python3
"""Profile Sam3VideoModel._det_track_one_frame step-by-step.

Hooks the 5 internal methods (get_vision_features → run_detection →
run_tracker_propagation → run_tracker_update_planning_phase →
run_tracker_update_execution_phase) and reports mean latency per stage.

Optionally patches the detector vision_encoder with the MIGraphX backbone
shim (--mig) so we can compare PT-only vs MIG-backbone profiles.

Usage:
    python eval/benchmarks/profile_video_pipeline.py --image assets/demo.jpg --text truck --frames 10 [--mig]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("/home/amd/project/sam3/model/sam3"))
    ap.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_1008"))
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--frames", type=int, default=10, help="How many timed iterations")
    ap.add_argument("--video", type=Path, default=None,
                    help="Optional mp4 — profile propagation frames 1..N (frame 0 init separate). "
                         "Without this, profiles N independent init frames (re-init every iter).")
    ap.add_argument("--max-frames", type=int, default=11, help="Cap video frames")
    ap.add_argument("--mig", action="store_true", help="Use MIG backbone (matches demo_text.py --mig)")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    print(f"Loading Sam3VideoModel from {args.checkpoint} ...")
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = Sam3VideoModel.from_pretrained(str(args.checkpoint)).to(device).to(dtype).eval()

    if args.mig:
        from tracker.tracker import MIGraphXBackbone
        from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
        det_dir = args.onnx_dir / "backbone_detector"
        mxr = MIGraphXBackbone(
            onnx_path=det_dir / "single_simplified.onnx",
            cache_path=det_dir / "tuned.mxr",
        )
        mxr.warmup(n=2)
        patch_sam3_video_model_with_mig(model, mxr)
        from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder
        detr_onnx = args.onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx"
        if detr_onnx.exists():
            patch_sam3_video_model_detr_encoder(model, detr_onnx)
            print("  detr_encoder patched in")
        print("  MIG backbone patched in")

    if args.video is not None:
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        frames = []
        while len(frames) < args.max_frames:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        cap.release()
        if len(frames) < 2:
            sys.exit("Need at least 2 frames in video for prop profiling")
        print(f"Loaded {len(frames)} video frames")
    else:
        frames = [Image.open(args.image).convert("RGB")]

    # Wrap target methods on the model with timing hooks
    timings: dict[str, list[float]] = defaultdict(list)
    targets = [
        ("get_vision_features", "detector_model"),     # backbone
        ("run_detection", None),                        # DETR enc/dec + mask head
        ("get_vision_features_for_tracker", None),      # tracker_neck
        ("run_tracker_propagation", None),              # memory_attention + dec_propagate
        ("run_tracker_update_planning_phase", None),    # NMS, association
        ("run_tracker_update_execution_phase", None),   # memory_encoder
        ("build_outputs", None),                        # final mask resize
    ]

    def make_wrap(name, fn):
        def wrapped(*a, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - t)
            return out
        return wrapped

    for name, attr_holder in targets:
        host = getattr(model, attr_holder) if attr_holder else model
        if not hasattr(host, name):
            continue
        original = getattr(host, name)
        setattr(host, name, make_wrap(name, original))

    if args.video is not None:
        # Init session once, profile propagation frames 1..N
        print(f"\nProfiling propagation across {len(frames)-1} frames (init = frame 0, not counted) ...")
        session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
        processor.add_text_prompt(session, args.text)
        with torch.inference_mode():
            _ = model(inference_session=session, frame_idx=0)  # init, not counted
        # Reset timings (init frame measurements distort prop stats)
        timings.clear()
        totals = []
        for i in range(1, len(frames)):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(inference_session=session, frame_idx=i)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            totals.append(elapsed)
            print(f"  prop frame {i}: {elapsed*1000:.0f} ms")
    else:
        print(f"\nRunning {args.frames} init-frame iterations ...")
        totals = []
        img = frames[0]
        for i in range(args.frames + 2):  # 2 warmup
            session = processor.init_video_session(video=[img], inference_device=device, dtype=dtype)
            processor.add_text_prompt(session, args.text)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(inference_session=session, frame_idx=0)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if i >= 2:
                totals.append(elapsed)
                print(f"  iter {i-2}: {elapsed*1000:.0f} ms")

    total_ms = statistics.mean(totals) * 1000
    print(f"\n=== Mean total: {total_ms:.0f} ms (over {len(totals)} iters) ===\n")

    print(f"{'Stage':<40s} {'Calls/iter':>10s} {'Mean ms':>10s} {'% of total':>10s}")
    print("-" * 75)
    n_iter = len(totals) if args.video else args.frames + 2
    accounted = 0.0
    for name, _ in targets:
        ts = timings.get(name, [])
        if not ts:
            continue
        # Each iter may invoke method multiple times
        if len(ts) % n_iter == 0:
            calls = len(ts) // n_iter
            warmup_drop = 0 if args.video else 2
            per_iter = [sum(ts[i*calls:(i+1)*calls]) for i in range(n_iter)][warmup_drop:]
        else:
            calls = len(ts) / n_iter
            per_iter = ts[-len(totals):]
        mean = statistics.mean(per_iter) * 1000
        pct = 100 * mean / total_ms if total_ms else 0
        accounted += mean
        cstr = f"{calls:.1f}" if isinstance(calls, float) else str(calls)
        print(f"{name:<40s} {cstr:>10s} {mean:>10.1f} {pct:>9.1f}%")
    print("-" * 75)
    other = total_ms - accounted
    print(f"{'(unaccounted: tokenize, build, host overhead)':<40s} {'':>10s} {other:>10.1f} {100*other/total_ms:>9.1f}%")
    print(f"{'TOTAL':<40s} {'':>10s} {total_ms:>10.1f} {100.0:>9.1f}%")


if __name__ == "__main__":
    sys.exit(main() or 0)
