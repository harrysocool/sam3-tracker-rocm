#!/usr/bin/env python3
"""Profile per-module timing in Sam3VideoModel propagation with FULL MIG path.

Patches all 3 MIG modules (backbone, detr_encoder, memory_attention) and
hooks the remaining PT modules to identify the next bottleneck for Phase 3
MIG-ization. Outputs both stdout table and JSON.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from transformers import Sam3VideoModel, AutoProcessor, Sam3VideoConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("model/sam3"))
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=504, choices=[504, 1008])
    p.add_argument("--max-frames", type=int, default=15,
                   help="frames after init to profile (default 15)")
    p.add_argument("--mig", action="store_true", default=True)
    p.add_argument("--no-mig", action="store_false", dest="mig")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    onnx_dir = Path(f"onnx_files_{args.imgsz}")
    device, dtype = torch.device("cuda"), torch.float16

    # Build model with config-based init for non-1008
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    config = Sam3VideoConfig.from_pretrained(str(args.checkpoint))
    if args.imgsz != 1008:
        config.image_size = args.imgsz
        config.low_res_mask_size = 4 * args.imgsz // 14
        new_size = {"height": args.imgsz, "width": args.imgsz}
        new_mask = {"height": 4 * args.imgsz // 14, "width": 4 * args.imgsz // 14}
        for sub in (getattr(processor, "image_processor", None),
                    getattr(processor, "video_processor", None)):
            if sub is not None:
                if hasattr(sub, "size"):
                    sub.size = new_size
                if hasattr(sub, "mask_size"):
                    sub.mask_size = new_mask
        if hasattr(processor, "target_size"):
            processor.target_size = args.imgsz
    model = (Sam3VideoModel.from_pretrained(str(args.checkpoint), config=config)
             .to(device).to(dtype).eval())

    if args.mig:
        from tracker.migraphx_runtime import MIGraphXBackbone
        from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
        from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder
        from tracker.mig_memory_attention import patch_sam3_video_model_memory_attention
        det_dir = onnx_dir / "backbone_detector"
        mxr = MIGraphXBackbone(det_dir / "single_simplified.onnx", det_dir / "tuned.mxr")
        patch_sam3_video_model_with_mig(model, mxr)
        detr_onnx = onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx"
        if detr_onnx.exists():
            patch_sam3_video_model_detr_encoder(model, detr_onnx)
        mem_onnx = onnx_dir / "tracker_modules" / "memory_attention_fixed_S7_P32.onnx"
        if mem_onnx.exists():
            patch_sam3_video_model_memory_attention(model, mem_onnx)

    trk = model.tracker_model
    timings = defaultdict(list)

    def make_wrap(name, fn):
        def w(*a, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - t)
            return out
        return w

    # Hook tracker submodules
    for mod_name in ["memory_attention", "mask_decoder", "memory_encoder",
                     "prompt_encoder", "obj_ptr_proj"]:
        if hasattr(trk, mod_name):
            m = getattr(trk, mod_name)
            if m is not None and hasattr(m, "forward"):
                m.forward = make_wrap(mod_name, m.forward)

    # Hook detector_model submodules
    det = getattr(model, "detector_model", None)
    if det is not None:
        for mod_name in ["vision_encoder", "detr_encoder", "tracker_neck",
                         "detr_decoder", "text_encoder"]:
            if hasattr(det, mod_name):
                m = getattr(det, mod_name)
                if m is not None and hasattr(m, "forward"):
                    m.forward = make_wrap(f"det.{mod_name}", m.forward)

    # Tracker neck — sometimes lives directly on model not under detector
    for parent_name in ["tracker_neck"]:
        if hasattr(model, parent_name):
            m = getattr(model, parent_name)
            if m is not None:
                m.forward = make_wrap(parent_name, m.forward)

    # Methods
    for meth_name in ["_prepare_memory_conditioned_features",
                      "_run_single_frame_inference", "_encode_new_memory"]:
        if hasattr(trk, meth_name):
            setattr(trk, meth_name, make_wrap(meth_name, getattr(trk, meth_name)))

    # Frames
    cap = cv2.VideoCapture(str(args.video))
    frames = []
    while len(frames) < args.max_frames + 1:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    print(f"Loaded {len(frames)} frames")

    sess = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
    processor.add_text_prompt(sess, args.text)
    with torch.inference_mode():
        _ = model(inference_session=sess, frame_idx=0)

    # Reset (drop init)
    timings.clear()

    # Profile prop
    prop_totals = []
    for i in range(1, len(frames)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(inference_session=sess, frame_idx=i)
        torch.cuda.synchronize()
        prop_totals.append(time.perf_counter() - t0)

    n = len(prop_totals)
    mean_total_ms = statistics.mean(prop_totals) * 1000
    print(f"\nMode: {'MIG' if args.mig else 'PT'} @ {args.imgsz}px")
    print(f"Mean prop frame: {mean_total_ms:.0f} ms (over {n} frames)")
    print(f"\n{'Module':<45s} {'Calls/iter':>12s} {'Mean ms':>10s} {'% total':>8s}")
    print("-" * 80)

    rows = []
    for name, samples in sorted(timings.items(), key=lambda kv: -sum(kv[1])):
        if not samples:
            continue
        total_s = sum(samples)
        per_iter_mean = total_s / n * 1000
        calls_per_iter = len(samples) / n
        pct = per_iter_mean / mean_total_ms * 100
        print(f"{name:<45s} {calls_per_iter:>12.1f} {per_iter_mean:>10.1f} {pct:>7.1f}%")
        rows.append({"module": name, "calls_per_iter": calls_per_iter,
                     "mean_ms": per_iter_mean, "pct_total": pct})

    summary = {
        "mode": "MIG" if args.mig else "PT",
        "imgsz": args.imgsz,
        "video": str(args.video),
        "text": args.text,
        "n_frames": n,
        "mean_total_ms": mean_total_ms,
        "modules": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
