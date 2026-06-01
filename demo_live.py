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
    p.add_argument(
        "--bootstrap-frames", type=int, default=0,
        help="First N frames run in pure text mode to capture high-confidence "
             "exemplar boxes; subsequent frames inject them as box prompts. "
             "Default 0 = pure text-prompt (original behaviour). 5 is a good "
             "starting value when multi-prompt empty-mask is a problem.",
    )
    p.add_argument("--bootstrap-min-score", type=float, default=0.3,
                   help="Confidence floor for boxes captured during bootstrap.")
    p.add_argument("--hybrid", action="store_true",
                   help="Use SAM3HybridLive: SAM3 detect every --keyframe-every-ms, "
                        "lightweight SAM2-style tracker propagates between keyframes. "
                        "Significantly faster for multi-prompt at the cost of some "
                        "mask freshness on intermediate frames.")
    p.add_argument("--keyframe-every-ms", type=float, default=1000.0,
                   help="Hybrid: wall-clock interval between SAM3 keyframe detections.")
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
            frame_idx: int, fps: float, live) -> np.ndarray:
    """Draw multi-class masks + 4-line drift/bootstrap HUD.

    HUD layout (top-left):
      L1: frame + keyframe/propagation + mode (BOOT/REBOOT/EXEM) + kept count
      L2: per-prompt mask pixel counts
      L3: per-prompt current avg score / drift baseline (ratio %)
      L4: per-prompt drift rolling mean + critical-warning + trigger threshold
    """
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
        # Mask-only blend: addWeighted on the whole frame darkens pixels
        # outside the mask too, compounding per object.
        color_arr = np.array(color, dtype=np.float32)
        vis[mask] = (vis[mask] * 0.55 + color_arr * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label = f"#{oid} {prompt} {result['scores'][oid]:.2f}"
            cv2.putText(vis, label, (max(x, 4), max(y - 6, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── 4-line drift/bootstrap HUD (mirrors opennav poc_on_mp4 visualisation) ──
    # `live` is either SAM3Live (direct) or SAM3HybridLive (wraps SAM3Live as .live)
    inner = live.live if hasattr(live, "live") else live

    # L1: mode + keyframe marker + kept count
    keyframe = result.get("keyframe", True)
    mark = "[K]" if keyframe else "[P]"
    bootstrap_remaining = getattr(inner, "_bootstrap_remaining", {})
    booting = any(v > 0 for v in bootstrap_remaining.values()) if bootstrap_remaining else False
    rebooting = getattr(inner, "_drift_pending_rebootstrap", False)
    if rebooting:
        mode, mode_color = "REBOOT", (0, 0, 255)
    elif booting:
        mode, mode_color = "BOOT", (0, 165, 255)
    else:
        mode, mode_color = "EXEM", (0, 255, 0)
    kept = sum(len(result['prompt_to_obj_ids'].get(p, [])) for p in prompts)
    total_obj = len(result.get('object_ids', []))
    hud1 = f"f={frame_idx:4d} {mark} [{mode}] kept={kept}/{total_obj}  {fps:.1f} FPS"
    cv2.putText(vis, hud1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    # L2: per-prompt mask pixel count
    px_parts = []
    for p in prompts:
        px = 0
        for oid in result['prompt_to_obj_ids'].get(p, []):
            m = result['masks'].get(oid)
            if m is not None and m.any():
                px += int(m.sum())
        px_parts.append(f"{p}:{px}")
    cv2.putText(vis, "  ".join(px_parts), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # L3: per-prompt cur_avg / baseline / ratio (drift signal)
    text_to_id = {}
    try:
        text_to_id = {v: k for k, v in inner.session.prompts.items()}
    except Exception:
        pass
    baseline_map = getattr(inner, "_drift_baseline_score", {}) or {}
    score_parts = []
    for prompt in prompts:
        pid = text_to_id.get(prompt)
        oids = result['prompt_to_obj_ids'].get(prompt, [])
        cur_avg = (sum(float(result['scores'].get(o, 0)) for o in oids) / len(oids)) if oids else 0.0
        baseline = baseline_map.get(pid)
        if baseline is None:
            score_parts.append(f"{prompt}:{cur_avg:.2f}/-")
        else:
            ratio = cur_avg / max(baseline, 1e-6)
            score_parts.append(f"{prompt}:{cur_avg:.2f}/{baseline:.2f}({ratio*100:.0f}%)")
    hud3 = "  ".join(score_parts) + "  (now/baseline)"
    cv2.putText(vis, hud3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # L4: rolling mean + critical warning + trigger threshold
    recent_map = getattr(inner, "_drift_recent_scores", {}) or {}
    drop_thresh = getattr(inner, "_drift_drop_threshold", 0.4)
    roll_parts = []
    for prompt in prompts:
        pid = text_to_id.get(prompt)
        recent = recent_map.get(pid)
        if recent and len(recent) > 0:
            rmean = sum(recent) / len(recent)
            baseline = baseline_map.get(pid, 0)
            min_acceptable = baseline * (1.0 - drop_thresh) if baseline else 0
            crit = "!" if (baseline and rmean < min_acceptable * 1.05) else " "
            roll_parts.append(f"{prompt}:roll{rmean:.2f}{crit}")
        else:
            roll_parts.append(f"{prompt}:roll-")
    hud4 = "  ".join(roll_parts) + f"  trigger<baseline*{1 - drop_thresh:.2f}"
    cv2.putText(vis, hud4, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
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

    max_obj = (
        None if args.max_objects == 0
        else (args.max_objects if args.max_objects > 0 else 5)
    )
    if args.hybrid:
        from tracker.hybrid_inference import SAM3HybridLive
        live = SAM3HybridLive(
            checkpoint=args.checkpoint,
            prompts=current_prompts,
            onnx_dir=args.onnx_dir,
            imgsz=args.imgsz,
            dtype=dtype,
            mig=args.mig,
            keyframe_every_ms=args.keyframe_every_ms,
            max_objects_per_prompt=max_obj,
            bootstrap_frames=args.bootstrap_frames,
            bootstrap_min_score=args.bootstrap_min_score,
        )
    else:
        live = SAM3Live(
            checkpoint=args.checkpoint,
            prompts=current_prompts,
            onnx_dir=args.onnx_dir,
            imgsz=args.imgsz,
            dtype=dtype,
            mig=args.mig,
            redetect_every=args.redetect_every,
            max_objects_per_prompt=max_obj,
            bootstrap_frames=args.bootstrap_frames,
            bootstrap_min_score=args.bootstrap_min_score,
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
        vis = overlay(frame_bgr, result, current_prompts, n, live_fps, live)
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

    # Transcode to H.264 if ffmpeg is available — cv2 writes MPEG-4 Part 2
    # (mp4v) which most browsers and VS Code's built-in video player don't
    # render. H.264 (avc1) works everywhere. Falls back silently if ffmpeg
    # is missing or the transcode errors out.
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        tmp_h264 = out_path.with_suffix(".h264.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(out_path),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(tmp_h264)],
                check=True,
            )
            tmp_h264.replace(out_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            tmp_h264.unlink(missing_ok=True)
            print(f"  ffmpeg transcode failed ({e}); leaving cv2 mp4v output as-is "
                  f"(may not play in browsers/VS Code).")
    else:
        print(f"  ffmpeg not found on PATH; output is cv2 mp4v "
              f"(may not play in browsers/VS Code).")

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
