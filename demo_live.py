#!/usr/bin/env python3
"""Streaming SAM3 demo — frame-by-frame, multi-prompt.

Simulates a live sensor by reading a video file frame-by-frame with OpenCV and
feeding each frame into ``SAM3Live.infer()``. Unlike ``demo_text.py``, no video
is pre-loaded into the session; frames arrive one at a time, as they would
from a webcam, RTSP stream, or robot camera.

Examples
--------

Single prompt, MIG accelerated:
    python demo_live.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \\
        --video assets/blackswan.mp4 --text swan --imgsz 504 --mig

Multi-class (key new feature):
    python demo_live.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \\
        --video assets/parkour.mp4 --text person trees buildings \\
        --imgsz 504 --mig

Mid-stream prompt switch (every --switch-every frames, cycling through
--text-set arguments):
    python demo_live.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \\
        --video assets/parkour.mp4 --imgsz 504 --mig \\
        --text-set person --text-set "trees,buildings" --switch-every 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")

import cv2
import numpy as np

# Trigger tracker/__init__.py ROCm patches before any HF model import.
from tracker.live_inference import SAM3Live


# Fixed palette (BGR).
_OBJ_COLORS = [
    (0, 200, 80), (255, 80, 0), (0, 80, 255), (0, 220, 220),
    (200, 0, 200), (0, 200, 255), (180, 255, 100), (255, 100, 180),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True,
                   help="Input video. Will be read frame-by-frame to simulate live input.")
    p.add_argument("--text", type=str, nargs="+", default=None,
                   help="One or more text prompts (e.g. --text car sidewalk grass). "
                        "Mutually exclusive with --text-set.")
    p.add_argument("--text-set", type=str, action="append", default=None,
                   help="Repeatable: a prompt-list (comma-separated) to switch to "
                        "during the stream. Use with --switch-every. Example: "
                        "--text-set car --text-set 'pedestrian,bicycle'.")
    p.add_argument("--switch-every", type=int, default=0,
                   help="With --text-set: rotate to next prompt set every N frames. "
                        "0 disables switching.")
    p.add_argument("--redetect-every", type=int, default=1,
                   help="Full SAM3 every Nth frame; intermediate frames are tracker "
                        "propagation only (faster, no new objects discovered). "
                        "Default 1 = full detection every frame.")
    p.add_argument("--max-objects", type=int, default=-1,
                   help="Cap tracked objects per prompt. -1 = use SAM3Live default (5). "
                        "0 = explicitly unlimited (NOT recommended — session accumulates "
                        "ghost detections that bloat tracker propagation to seconds per "
                        "frame). Positive int = per-prompt cap.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output mp4. Default: results/<video-stem>_live_<ts>.mp4")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap frames processed (0 = entire video).")
    p.add_argument("--imgsz", type=int, default=504, choices=(504, 1008))
    p.add_argument("--mig", action="store_true")
    p.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_504"))
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="Filter detections below this score (default 0.5).")
    args = p.parse_args()
    if (args.text is None) == (args.text_set is None):
        sys.exit("Pass exactly one of --text or --text-set.")
    return args


def overlay(bgr: np.ndarray, result: dict, prompts: list[str],
            frame_idx: int, fps: float) -> np.ndarray:
    """Draw multi-class masks + per-prompt legend."""
    H, W = bgr.shape[:2]
    vis = bgr.copy()

    # Assign one color per prompt (consistent across frames).
    prompt_color = {p: _OBJ_COLORS[i % len(_OBJ_COLORS)] for i, p in enumerate(prompts)}

    # Per-object draw; color comes from the prompt that owns the obj_id.
    obj_to_prompt = {}
    for prompt, oids in result["prompt_to_obj_ids"].items():
        for oid in oids:
            obj_to_prompt[oid] = prompt

    for oid in result["object_ids"]:
        prompt = obj_to_prompt.get(oid, "?")
        color = prompt_color.get(prompt, (255, 255, 255))
        mask = result["masks"][oid]
        if not mask.any():
            continue
        layer = np.zeros_like(bgr)
        layer[mask] = color
        vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label = f"#{oid} {prompt} {result['scores'][oid]:.2f}"
            cv2.putText(vis, label, (max(x, 4), max(y - 6, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Header: frame + per-prompt object counts + live fps
    counts = ", ".join(f"{p}:{len(result['prompt_to_obj_ids'].get(p, []))}"
                       for p in prompts)
    cv2.putText(vis, f"f={frame_idx}  {counts}  {fps:.1f} FPS",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def filter_result(result: dict, min_score: float) -> dict:
    """Drop objects below threshold (in-place on a copy)."""
    keep = [oid for oid in result["object_ids"]
            if result["scores"].get(oid, 0.0) >= min_score]
    keep_set = set(keep)
    return {
        "object_ids": keep,
        "scores": {k: v for k, v in result["scores"].items() if k in keep_set},
        "masks": {k: v for k, v in result["masks"].items() if k in keep_set},
        "boxes": {k: v for k, v in result["boxes"].items() if k in keep_set},
        "prompt_to_obj_ids": {p: [o for o in oids if o in keep_set]
                              for p, oids in result["prompt_to_obj_ids"].items()},
        "frame_idx": result["frame_idx"],
    }


def main():
    args = parse_args()

    # Resolve prompt schedule.
    if args.text is not None:
        prompt_sets = [list(args.text)]
    else:
        prompt_sets = [[t.strip() for t in s.split(",") if t.strip()]
                       for s in args.text_set]
    current_set_idx = 0
    current_prompts = prompt_sets[0]
    print(f"[demo_live] prompt schedule: {prompt_sets}")
    if args.switch_every > 0 and len(prompt_sets) > 1:
        print(f"[demo_live] rotating every {args.switch_every} frames")

    import torch
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    live = SAM3Live(
        checkpoint=args.checkpoint,
        prompts=current_prompts,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        dtype=dtype,
        mig=args.mig,
        redetect_every=args.redetect_every,
        # -1 = let SAM3Live use its default; 0 = explicit None (unlimited); >0 = cap
        max_objects_per_prompt=(
            None if args.max_objects == 0
            else (args.max_objects if args.max_objects > 0 else 5)
        ),
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output path
    if args.output is not None:
        out_path = args.output
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"{args.video.stem}_live_{ts}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (W, H))

    print(f"[demo_live] streaming {args.video.name} → {out_path}")
    n = 0
    latencies = []  # per-frame infer() wall time, ms
    t_total = time.perf_counter()
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Mid-stream prompt switch
        if (args.switch_every > 0 and len(prompt_sets) > 1
                and n > 0 and n % args.switch_every == 0):
            next_idx = (current_set_idx + 1) % len(prompt_sets)
            current_set_idx = next_idx
            current_prompts = prompt_sets[next_idx]
            t_sw = time.perf_counter()
            live.reset_prompts(current_prompts)
            print(f"[demo_live] f={n}: switched prompts → {current_prompts} "
                  f"({(time.perf_counter()-t_sw)*1000:.0f} ms)")

        t_infer = time.perf_counter()
        result = live.infer(frame_bgr)
        latency_ms = (time.perf_counter() - t_infer) * 1000
        latencies.append(latency_ms)

        result = filter_result(result, args.min_score)

        # Rolling FPS over last 10 frames
        window = latencies[-10:]
        live_fps = 1000.0 / (sum(window) / len(window))
        vis = overlay(frame_bgr, result, current_prompts, n, live_fps)
        writer.write(vis)

        if n % 20 == 0:
            counts = {p: len(result["prompt_to_obj_ids"].get(p, []))
                      for p in current_prompts}
            print(f"  f={n}  latency={latency_ms:5.1f} ms  "
                  f"FPS={live_fps:5.2f}  objs={counts}")
        n += 1
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    writer.release()
    t_total = time.perf_counter() - t_total

    if latencies:
        lat = np.asarray(latencies)
        # Separate first-frame (cold) from steady state
        first = lat[0]
        steady = lat[1:] if len(lat) > 1 else lat
        print(f"\n[demo_live] {n} frames in {t_total:.1f}s — "
              f"end-to-end {n/t_total:.2f} FPS")
        print(f"  first-frame latency: {first:.1f} ms")
        print(f"  steady-state:        mean={steady.mean():.1f} ms  "
              f"p50={np.median(steady):.1f}  p95={np.percentile(steady,95):.1f}  "
              f"max={steady.max():.1f}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
