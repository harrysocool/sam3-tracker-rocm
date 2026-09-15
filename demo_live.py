#!/usr/bin/env python3
"""Streaming SAM3 demo — frame-by-frame, multi-prompt.

Simulates a live sensor by reading a video file at source cadence with OpenCV.
A bounded latest-frame scheduler continuously drains capture, drops stale
waiting frames, and starts the complete ``SAM3Live.infer()`` path only after
the preceding inference finishes. Unlike ``tools/text_baseline.py``, no video
is pre-loaded into the session and no N+1 GPU work is launched.

Examples
--------

Single prompt, MIG accelerated:
    python demo_live.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \\
        --video assets/blackswan.mp4 --text swan --imgsz 504 --mig

Multi-class (key new feature):
    python demo_live.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \\
        --video assets/parkour.mp4 --text person trees buildings \\
        --imgsz 504 --mig

Prompt changes require an explicit pipeline close, session reset, and a new
pipeline generation; this demo intentionally does not switch them mid-stream.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from tracker.rocm_env import apply as _apply_rocm_env; _apply_rocm_env()

import cv2
import numpy as np

# Trigger tracker/__init__.py ROCm patches before any HF model import.
from tracker.live_inference import SAM3Live
from tracker.output_processing import DEFAULT_MAX_OBJECTS_PER_PROMPT, filter_result


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
    p.add_argument("--redetect-interval-ms", type=float, default=0.0,
                   help="Wall-clock interval between SAM3 detections (ms). "
                        "0 = SAM3 on every consumed latest frame (default; "
                        "recommended for occupancy-grid freshness). "
                        ">0 = SAM3 on keyframes; the same model's native tracker "
                        "propagates between. "
                        "Hybrid propagation is opt-in.")
    p.add_argument(
        "--bootstrap-frames", type=int, default=0,
        help="First N frames run in pure text mode to capture high-confidence "
             "exemplar boxes; subsequent frames inject them as box prompts. "
             "Default 0 = pure text-prompt (original behaviour). 5 is a good "
             "starting value when multi-prompt empty-mask is a problem.",
    )
    p.add_argument("--bootstrap-min-score", type=float, default=0.3,
                   help="Confidence floor for boxes captured during bootstrap.")
    p.add_argument("--periodic-rebootstrap-seconds", type=float, default=180.0,
                   help="Force re-bootstrap every N seconds wall-clock. Catches scene "
                        "changes the score-only drift signal cannot detect. "
                        "Default 180s (3 min safety net); set to 0 to disable.")
    p.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS_PER_PROMPT,
                   help="Maximum tracked objects per prompt (default 5; 0 = unlimited). "
                        "The legacy -1 value also selects the default.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output mp4. Default: results/<video-stem>_live_<ts>.mp4")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap source frames read (0 = entire video). Output frames may "
                        "be fewer because the live path drops stale frames.")
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Explicit live benchmark warmup count. Full-detection mode detects "
             "on every warmup frame; hybrid mode detects on the first and uses "
             "tracker propagation thereafter. Synchronizes, resets tracking, "
             "then seeks to frame 0. Default 0 performs no preload/warmup.",
    )
    p.add_argument("--imgsz", type=int, default=504, choices=(504, 1008))
    mig_group = p.add_mutually_exclusive_group()
    mig_group.add_argument(
        "--mig",
        dest="mig",
        action="store_true",
        help="Enable the supported MIGraphX path (default).",
    )
    mig_group.add_argument(
        "--no-mig",
        dest="mig",
        action="store_false",
        help="Use pure PyTorch for diagnosis.",
    )
    p.set_defaults(mig=True)
    parallel = p.add_mutually_exclusive_group()
    parallel.add_argument(
        "--parallel-tail",
        dest="parallel_tail",
        action="store_true",
        help="Enable detector/tracker HIP-stream overlap (default when --mig is set).",
    )
    parallel.add_argument(
        "--no-parallel-tail",
        dest="parallel_tail",
        action="store_false",
        help="Disable detector/tracker overlap for diagnosis or compatibility.",
    )
    p.set_defaults(parallel_tail=None)
    fixed_decoder = p.add_mutually_exclusive_group()
    fixed_decoder.add_argument(
        "--fixed-detr-decoder",
        dest="fixed_detr_decoder",
        action="store_true",
        help="Require the fixed 504px direct-MXR DETR decoder artifact.",
    )
    fixed_decoder.add_argument(
        "--no-fixed-detr-decoder",
        dest="fixed_detr_decoder",
        action="store_false",
        help="Use the native PyTorch DETR decoder for diagnosis.",
    )
    p.set_defaults(fixed_detr_decoder=None)
    p.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path(
            os.environ.get("SAM3_DEFAULT_ONNX_DIR", "onnx_files_504_mgx217")
        ),
        help="MIG artifact root (defaults to the supported 2.17 runtime root).",
    )
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="Filter detections below this score (default 0.5).")
    args = p.parse_args()
    if (args.text is None) == (args.text_set is None):
        sys.exit("Pass exactly one of --text or --text-set.")
    if args.text is not None:
        args.text = list(dict.fromkeys(prompt.strip() for prompt in args.text))
        if not all(args.text):
            p.error("--text prompts must not be empty")
    if args.max_objects == -1:
        args.max_objects = DEFAULT_MAX_OBJECTS_PER_PROMPT
    if args.max_objects < 0:
        p.error("--max-objects must be non-negative (or -1 for the default)")
    if args.max_frames < 0:
        p.error("--max-frames must be non-negative; 0 reads to the end")
    if not 0.0 <= args.min_score <= 1.0:
        p.error("--min-score must be between 0 and 1")
    if args.parallel_tail is None:
        args.parallel_tail = args.mig
    if args.parallel_tail and not args.mig:
        sys.exit("--parallel-tail requires --mig")
    if args.fixed_detr_decoder is True and not args.mig:
        sys.exit("--fixed-detr-decoder requires --mig")
    if (
        args.bootstrap_frames > 0
        and args.mig
        and args.imgsz == 504
        and args.fixed_detr_decoder is not False
    ):
        sys.exit(
            "--bootstrap-frames is incompatible with the fixed 32-token DETR "
            "decoder; pass --no-fixed-detr-decoder for diagnosis"
        )
    if args.warmup_frames < 0:
        sys.exit("--warmup-frames must be >= 0")
    if args.text_set is not None and args.switch_every > 0:
        sys.exit(
            "the default latest-frame scheduler requires prompt changes to "
            "close/reset/restart the pipeline"
        )
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


def _warmup_uses_full_detection(index: int, *, hybrid: bool) -> bool:
    """Warm full mode every frame; warm hybrid detection then propagation."""
    return not hybrid or index == 0


def _capture_latest_video(cap, pipeline, source_fps: float, max_frames: int,
                          state: dict) -> None:
    """Pace a file like a live source and publish frames with latest semantics."""
    period = 1.0 / source_fps
    started_at = time.perf_counter()
    sequence = 0
    try:
        while max_frames <= 0 or sequence < max_frames:
            scheduled_at = started_at + sequence * period
            delay = scheduled_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            ok, frame_bgr = cap.read()
            if not ok:
                break
            captured_at = time.perf_counter()
            if not pipeline.submit(
                frame_bgr,
                sequence=sequence,
                captured_at=captured_at,
            ):
                break
            state["captured"] = sequence + 1
            sequence += 1
    except BaseException as exc:
        state["error"] = exc
        pipeline.abort()
    finally:
        state["finished_at"] = time.perf_counter()
        pipeline.finish_input()


def main():
    args = parse_args()

    # Resolve prompt schedule.
    if args.text is not None:
        prompt_sets = [list(args.text)]
    else:
        prompt_sets = [[t.strip() for t in s.split(",") if t.strip()]
                       for s in args.text_set]
    current_prompts = prompt_sets[0]
    print(f"[demo_live] prompt schedule: {prompt_sets}")
    if args.switch_every > 0 and len(prompt_sets) > 1:
        print(f"[demo_live] rotating every {args.switch_every} frames")

    import torch
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    max_obj = None if args.max_objects == 0 else args.max_objects
    hybrid_mode = args.redetect_interval_ms > 0.0
    if not hybrid_mode:
        live = SAM3Live(
            checkpoint=args.checkpoint,
            prompts=current_prompts,
            onnx_dir=args.onnx_dir,
            imgsz=args.imgsz,
            dtype=dtype,
            mig=args.mig,
            parallel_tail=args.parallel_tail,
            fixed_detr_decoder=args.fixed_detr_decoder,
            redetect_every=1,
            max_objects_per_prompt=max_obj,
            bootstrap_frames=args.bootstrap_frames,
            bootstrap_min_score=args.bootstrap_min_score,
            periodic_rebootstrap_seconds=args.periodic_rebootstrap_seconds,
        )
    else:
        from tracker.hybrid_inference import SAM3HybridLive
        live = SAM3HybridLive(
            checkpoint=args.checkpoint,
            prompts=current_prompts,
            onnx_dir=args.onnx_dir,
            imgsz=args.imgsz,
            dtype=dtype,
            mig=args.mig,
            parallel_tail=args.parallel_tail,
            fixed_detr_decoder=args.fixed_detr_decoder,
            redetect_interval_ms=args.redetect_interval_ms,
            max_objects_per_prompt=max_obj,
            bootstrap_frames=args.bootstrap_frames,
            bootstrap_min_score=args.bootstrap_min_score,
            periodic_rebootstrap_seconds=args.periodic_rebootstrap_seconds,
        )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.warmup_frames:
        warmup_description = (
            "full detection on frame 0, then detector-skip propagation"
            if hybrid_mode
            else "full detection on every frame"
        )
        print(
            f"[demo_live] explicit warmup: {args.warmup_frames} file frame(s), "
            f"{warmup_description}; these are not live arrivals"
        )
        for index in range(args.warmup_frames):
            ok, warm_frame = cap.read()
            if not ok:
                cap.release()
                live.close()
                sys.exit(f"Warmup frame {index} unavailable in {args.video}")
            live.infer(
                warm_frame,
                full_detection=_warmup_uses_full_detection(
                    index,
                    hybrid=hybrid_mode,
                ),
            )
        torch.cuda.synchronize(device=live.device)
        live.reset_tracking()
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            cap.release()
            live.close()
            sys.exit(f"Cannot seek {args.video} back to frame 0 after warmup")

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
    print(
        f"[demo_live] latest-frame source cadence={fps_in:.3f} FPS; "
        "no next-frame GPU lookahead; output video contains emitted frames "
        "only and is time-compressed when frames drop"
    )
    n = 0
    latencies = []  # per-frame infer() wall time, ms
    frame_ages = []  # source arrival -> completed output, ms
    completion_times = []
    source_sequences = []
    t_total = time.perf_counter()
    pipeline = None
    capture_thread = None
    capture_state = {"captured": 0, "error": None, "finished_at": None}

    def consume_result(frame_bgr, result, latency_ms: float, age_ms: float,
                       source_sequence: int, completed_at: float) -> None:
        nonlocal n
        latencies.append(latency_ms)
        frame_ages.append(age_ms)
        completion_times.append(completed_at)
        source_sequences.append(source_sequence)

        filtered = filter_result(result, args.min_score)
        recent = completion_times[-11:]
        live_fps = (
            (len(recent) - 1) / max(recent[-1] - recent[0], 1e-9)
            if len(recent) >= 2 else 0.0
        )
        vis = overlay(frame_bgr, filtered, current_prompts,
                      source_sequence, live_fps, live)
        writer.write(vis)

        if n % 20 == 0:
            counts = {
                p: len(filtered["prompt_to_obj_ids"].get(p, []))
                for p in current_prompts
            }
            print(
                f"  out={n} src={source_sequence} latency={latency_ms:5.1f} ms"
                f"  age={age_ms:5.1f} ms  FPS={live_fps:5.2f}  objs={counts}"
            )
        n += 1

    try:
        from tracker.latest_frame import LatestFramePipeline

        pipeline = LatestFramePipeline(live)
        pipeline.start()
        capture_thread = threading.Thread(
            target=_capture_latest_video,
            args=(cap, pipeline, fps_in, args.max_frames, capture_state),
            name="sam3-latest-frame-capture",
        )
        capture_thread.start()
        while True:
            item = pipeline.infer_next()
            if item is None:
                break
            consume_result(
                item.packet.frame_bgr,
                item.output,
                item.service_ms,
                item.age_ms,
                item.packet.sequence,
                item.completed_at,
            )
        capture_thread.join()
        if capture_state["error"] is not None:
            raise RuntimeError("latest-frame video capture failed") from capture_state["error"]
    finally:
        if pipeline is not None:
            pipeline.close()
        if capture_thread is not None and capture_thread.is_alive():
            capture_thread.join()
        cap.release()
        writer.release()
        if hasattr(live, "close"):
            live.close()
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
        age = np.asarray(frame_ages)
        cadence = np.diff(np.asarray(completion_times)) * 1000.0
        stats = pipeline.stats()
        output_hz = 1000.0 / cadence.mean() if len(cadence) else 0.0
        source_gaps = np.diff(np.asarray(source_sequences))
        print(
            f"  latest-frame output: Hz={output_hz:.2f}  "
            f"age_p50={np.median(age):.1f} ms  "
            f"age_p95={np.percentile(age,95):.1f} ms"
        )
        print(
            "  latest-frame drops:  "
            f"latest={stats['dropped_frames']}  "
            f"failed={stats['failed_frames']}  "
            f"abort={stats['aborted_frames']}  "
            f"drain_failed={stats['drain_failed']}  "
            f"max_source_gap={int(source_gaps.max()) if len(source_gaps) else 0}"
        )
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
