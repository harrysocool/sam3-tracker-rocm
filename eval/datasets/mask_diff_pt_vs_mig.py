#!/usr/bin/env python3
"""Compare PT vs MIG mask outputs frame-by-frame for text-prompt path.

Run a short clip through both PT and MIG paths, compute per-frame
IoU between PT and MIG masks. Catches MIG numerical regressions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from transformers import Sam3VideoModel, AutoProcessor, Sam3VideoConfig
import tracker  # noqa: F401  -- applies ROCm patches (scipy fill_holes + PyTorch NMS)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("model/sam3"))
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=1008, choices=[504, 1008])
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="Default: onnx_files_<imgsz>")
    p.add_argument("--out", type=Path, required=True,
                   help="JSON output with per-frame IoU stats")
    return p.parse_args()


def load_video(path: Path, n: int):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def build_model(ckpt: Path, imgsz: int, device, dtype):
    processor = AutoProcessor.from_pretrained(str(ckpt))
    config = Sam3VideoConfig.from_pretrained(str(ckpt))
    if imgsz != 1008:
        config.image_size = imgsz
        config.low_res_mask_size = 4 * imgsz // 14
        new_size = {"height": imgsz, "width": imgsz}
        new_mask = {"height": 4 * imgsz // 14, "width": 4 * imgsz // 14}
        for sub in (getattr(processor, "image_processor", None),
                    getattr(processor, "video_processor", None)):
            if sub is not None:
                if hasattr(sub, "size"):
                    sub.size = new_size
                if hasattr(sub, "mask_size"):
                    sub.mask_size = new_mask
        if hasattr(processor, "target_size"):
            processor.target_size = imgsz
    model = (Sam3VideoModel.from_pretrained(str(ckpt), config=config)
             .to(device).to(dtype).eval())
    return processor, model


def patch_mig(model, onnx_dir: Path):
    from tracker.tracker import MIGraphXBackbone
    from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
    det_dir = onnx_dir / "backbone_detector"
    mxr = MIGraphXBackbone(
        onnx_path=det_dir / "single_simplified.onnx",
        cache_path=det_dir / "tuned.mxr",
    )
    patch_sam3_video_model_with_mig(model, mxr)
    detr_onnx = onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx"
    if detr_onnx.exists():
        from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder
        patch_sam3_video_model_detr_encoder(model, detr_onnx)
    mem_attn_onnx = onnx_dir / "tracker_modules" / "memory_attention_fixed_S7_P32.onnx"
    if mem_attn_onnx.exists():
        from tracker.mig_memory_attention import patch_sam3_video_model_memory_attention
        patch_sam3_video_model_memory_attention(model, mem_attn_onnx)


def run_path(processor, model, frames_pil, text, device, dtype, n: int):
    """Returns list of (mask_bool_np_array_HxW, score) for each of n frames."""
    session = processor.init_video_session(
        video=frames_pil, inference_device=device, dtype=dtype,
    )
    processor.add_text_prompt(session, text)
    masks_scores = []
    for i in range(n):
        with torch.inference_mode():
            out = model(inference_session=session, frame_idx=i)
        if i == 0:
            if not out.object_ids:
                return None
            primary = max(out.object_ids, key=lambda j: out.obj_id_to_score.get(j, 0))
        if primary in (out.obj_id_to_mask or {}):
            m = out.obj_id_to_mask[primary]
            s = float(out.obj_id_to_score.get(primary, 0.0))
            if isinstance(m, torch.Tensor):
                m = m.detach().float().cpu().numpy()
            m = np.asarray(m).squeeze() > 0
        else:
            m = np.zeros_like(masks_scores[0][0]) if masks_scores else np.zeros((1,1), dtype=bool)
            s = 0.0
        masks_scores.append((m, s))
    return masks_scores


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def main():
    args = parse_args()
    if args.onnx_dir is None:
        args.onnx_dir = Path(f"onnx_files_{args.imgsz}")
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"Loading {args.max_frames} frames ...")
    frames = load_video(args.video, args.max_frames)
    n = len(frames)
    print(f"  {n} frames")

    print(f"=== PT @{args.imgsz} ===")
    t = time.perf_counter()
    processor, model = build_model(args.checkpoint, args.imgsz, device, dtype)
    pt_results = run_path(processor, model, frames, args.text, device, dtype, n)
    pt_time = time.perf_counter() - t
    print(f"  done in {pt_time:.1f}s")

    if pt_results is None:
        print("PT detected no objects")
        return 1

    del model
    torch.cuda.empty_cache()

    print(f"\n=== MIG @{args.imgsz} ===")
    t = time.perf_counter()
    processor2, model2 = build_model(args.checkpoint, args.imgsz, device, dtype)
    patch_mig(model2, args.onnx_dir)
    mig_results = run_path(processor2, model2, frames, args.text, device, dtype, n)
    mig_time = time.perf_counter() - t
    print(f"  done in {mig_time:.1f}s")

    if mig_results is None:
        print("MIG detected no objects")
        return 1

    print(f"\n=== Per-frame IoU PT vs MIG @{args.imgsz} ===")
    per_frame = []
    for i, ((pm, ps), (mm, ms)) in enumerate(zip(pt_results, mig_results)):
        ij = iou(pm, mm)
        per_frame.append({"frame": i, "iou": ij, "pt_score": ps, "mig_score": ms,
                          "pt_pix": int(pm.sum()), "mig_pix": int(mm.sum())})
        print(f"  frame {i:3d}: IoU={ij:.3f}  PT score={ps:.2f} pix={pm.sum():>6}  "
              f"MIG score={ms:.2f} pix={mm.sum():>6}")

    ious = [pf["iou"] for pf in per_frame]
    summary = {
        "imgsz": args.imgsz,
        "video": str(args.video),
        "text": args.text,
        "n_frames": n,
        "pt_time_s": pt_time,
        "mig_time_s": mig_time,
        "iou_mean": float(np.mean(ious)),
        "iou_min":  float(np.min(ious)),
        "iou_p10":  float(np.percentile(ious, 10)),
        "first_drop_below_0.95_frame": next((pf["frame"] for pf in per_frame if pf["iou"] < 0.95), -1),
        "first_drop_below_0.80_frame": next((pf["frame"] for pf in per_frame if pf["iou"] < 0.80), -1),
        "per_frame": per_frame,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary @{args.imgsz}:")
    print(f"  mean IoU = {summary['iou_mean']:.3f}")
    print(f"  min  IoU = {summary['iou_min']:.3f}  (p10 = {summary['iou_p10']:.3f})")
    print(f"  first drop <0.95 at frame {summary['first_drop_below_0.95_frame']}")
    print(f"  first drop <0.80 at frame {summary['first_drop_below_0.80_frame']}")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
