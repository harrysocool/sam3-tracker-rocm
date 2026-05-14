#!/usr/bin/env python3
"""
Stage 2: per-module profile of Sam3VideoModel init-frame forward, to decide
which submodules are worth replacing with the existing MIGraphX-optimized
backbone (and which still need work).

Hooks each top-level submodule of detector_model + tracker_model, runs
N warmup + N timed iterations on a single image, and prints a breakdown.

Requires:
    PYTHONPATH=/home/amd/project/sam3/repo/DART/.local_deps:<sam3-tracker-rocm>
    HSA_OVERRIDE_GFX_VERSION=11.5.1
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from transformers import Sam3VideoModel, AutoProcessor


def attach_timers(modules: dict[str, torch.nn.Module]) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = defaultdict(list)

    def make_hooks(name: str):
        def pre(m, inputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            m._t0 = time.perf_counter()

        def post(m, inputs, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - m._t0)
        return pre, post

    for name, module in modules.items():
        if module is None:
            continue
        pre, post = make_hooks(name)
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)
    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/home/amd/project/sam3/model/sam3"))
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    print(f"\nLoading {args.checkpoint} ...")
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = Sam3VideoModel.from_pretrained(str(args.checkpoint))
    model = model.to(device).to(dtype).eval()
    torch.cuda.synchronize()

    img = Image.open(args.image).convert("RGB")

    det = model.detector_model
    trk = model.tracker_model

    submodules = {
        "vision_encoder    (= our MIGraphX backbone)": getattr(det, "vision_encoder", None),
        "text_encoder     (CLIP, 300M)": getattr(det, "text_encoder", None),
        "text_projection  (linear 1024->256)": getattr(det, "text_projection", None),
        "geometry_encoder (box prompts; usually skipped for text)": getattr(det, "geometry_encoder", None),
        "detr_encoder     (DETR enc layers)": getattr(det, "detr_encoder", None),
        "detr_decoder     (DETR dec layers)": getattr(det, "detr_decoder", None),
        "mask_decoder     (det mask head)": getattr(det, "mask_decoder", None),
        "tracker_neck     (FPN-like)": getattr(model, "tracker_neck", None),
        "tracker_model    (full tracker step)": trk,
    }

    print(f"\nAttaching timers to {sum(1 for m in submodules.values() if m is not None)} submodules")
    timings = attach_timers(submodules)

    totals: list[float] = []

    print(f"\nRun: {args.warmup} warmup + {args.iters} timed iterations")
    for i in range(args.warmup + args.iters):
        # Fresh session each iter so init-frame path runs every time
        session = processor.init_video_session(
            video=[img], inference_device=device, dtype=dtype,
        )
        processor.add_text_prompt(session, args.text)

        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            _out = model(inference_session=session, frame_idx=0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t

        if i >= args.warmup:
            totals.append(elapsed)
            print(f"  iter {i - args.warmup}: total {elapsed * 1000:.1f}ms")

    # Trim per-module timings to last `iters` calls (each module may fire
    # multiple times per forward, so we report mean per-iteration sum)
    print(f"\n=== Per-submodule timings (mean ± stdev over {args.iters} iters) ===")
    print(f"{'Module':<48s} {'Calls/iter':>10s} {'Mean ms':>10s} {'± stdev':>10s} {'% of total':>10s}")
    print("-" * 95)

    total_mean_ms = statistics.mean(totals) * 1000
    rows = []
    for name in submodules:
        ts = timings.get(name, [])
        if not ts:
            continue
        # Each iteration may invoke a module N times; partition into per-iter sums
        n_iter = args.warmup + args.iters
        if len(ts) % n_iter == 0:
            calls_per_iter = len(ts) // n_iter
            per_iter_sum = [
                sum(ts[i * calls_per_iter:(i + 1) * calls_per_iter])
                for i in range(n_iter)
            ][args.warmup:]
        else:
            # Module fires variable number of times — fall back to flat mean × calls_per_iter heuristic
            calls_per_iter = len(ts) / n_iter
            per_iter_sum = ts[-args.iters:]
        mean = statistics.mean(per_iter_sum) * 1000
        stdev = statistics.stdev(per_iter_sum) * 1000 if len(per_iter_sum) > 1 else 0
        pct = 100 * mean / total_mean_ms if total_mean_ms > 0 else 0
        rows.append((name, calls_per_iter, mean, stdev, pct))

    for name, calls, mean, stdev, pct in rows:
        cstr = f"{calls:.1f}" if isinstance(calls, float) else str(calls)
        print(f"{name:<48s} {cstr:>10s} {mean:>10.1f} {stdev:>10.1f} {pct:>9.1f}%")

    print("-" * 95)
    accounted = sum(r[2] for r in rows if "tracker_model" not in r[0])  # avoid double-counting tracker
    other = total_mean_ms - sum(r[2] for r in rows)
    print(f"{'TOTAL':<48s} {'':>10s} {total_mean_ms:>10.1f} {'':>10s} {100.0:>9.1f}%")
    print(f"\nNote: 'tracker_model' contains tracker_neck + memory ops. "
          "If you sum naively you'll double-count.")


if __name__ == "__main__":
    raise SystemExit(main())
