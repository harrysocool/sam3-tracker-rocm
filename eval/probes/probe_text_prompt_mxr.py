#!/usr/bin/env python3
"""
Stage A integration probe: SAM3 text→mask using our MIGraphX backbone in
place of the PyTorch ViT backbone, with the official `Sam3VideoModel.run_detection`
postproc (presence-aware scores + mask-IoU NMS) ported to host code.

Bypasses `Sam3VideoModel` and calls `Sam3Model.forward` directly. That's the
official single-image text→mask entry point and is sufficient to validate the
backbone speedup *and* mask quality (vs. the PyTorch reference path).

Outputs:
  - <output-stem>_pt.jpg    PyTorch backbone + detector + new postproc
  - <output-stem>_mxr.jpg   MIGraphX backbone + detector + new postproc
  - Per-detection IoU between the two paths (numerical equivalence check)

Requires:
    PYTHONPATH=/home/amd/project/sam3/repo/DART/.local_deps:<sam3-tracker-rocm>
    HSA_OVERRIDE_GFX_VERSION=11.5.1
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from tracker.tracker import MIGraphXBackbone
from tracker.text_detector_postproc import (
    Detection,
    bbox_from_mask,
    mask_logits_to_image,
    postprocess_detector_outputs,
)


def build_mxr_vision_features(
    pixel_values: torch.Tensor,
    mxr_backbone: MIGraphXBackbone,
    pos_enc_module: torch.nn.Module,
) -> Sam3VisionEncoderOutput:
    """Run our MIGraphX backbone and wrap output in Sam3VisionEncoderOutput.

    pixel_values: (1, 3, H, W) torch fp16/fp32 on cuda
    Returns Sam3VisionEncoderOutput with all 4 FPN levels. If the backbone is
    an older 3-output build, fpn_3 falls back to zeros (detector path will
    misbehave — re-run export/export_backbone_single.py to fix).
    """
    device = pixel_values.device
    dtype = pixel_values.dtype

    np_in = pixel_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
    outs = mxr_backbone(np_in); fpn0, fpn1, fpn2, fpn3 = outs[:4]

    fpn = [
        torch.from_numpy(np.ascontiguousarray(t)).to(device=device, dtype=dtype)
        for t in (fpn0, fpn1, fpn2)
    ]

    if fpn3 is None:
        # Backward compat: older 3-output backbone. Detector will produce garbage.
        h4, w4 = fpn[2].shape[2] // 2, fpn[2].shape[3] // 2
        fpn.append(torch.zeros(1, 256, h4, w4, device=device, dtype=dtype))
    else:
        fpn.append(torch.from_numpy(np.ascontiguousarray(fpn3)).to(device=device, dtype=dtype))

    pe = [pos_enc_module(f.shape, f.device, f.dtype) for f in fpn]

    return Sam3VisionEncoderOutput(
        fpn_hidden_states=tuple(fpn),
        fpn_position_encoding=tuple(pe),
    )


def overlay_detections(
    img_bgr: np.ndarray,
    detections: list[Detection],
    text: str,
    label: str,
) -> np.ndarray:
    """Draw masks + scores + bboxes on a copy of img_bgr."""
    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()
    palette = [(0, 200, 80), (200, 80, 0), (80, 0, 200), (200, 200, 0), (0, 200, 200)]

    for i, det in enumerate(detections):
        mask = mask_logits_to_image(det.mask_logits, (H, W))
        bbox = bbox_from_mask(mask)
        color = palette[i % len(palette)]

        layer = np.zeros_like(vis)
        layer[mask] = color
        vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, f"#{i} {det.score:.2f}", (x1, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

    cv2.putText(
        vis, f"{label}  text='{text}'  N={len(detections)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2,
    )
    return vis


def mask_iou_pair(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/home/amd/project/sam3/model/sam3"))
    ap.add_argument("--onnx-dir", type=Path,
                    default=Path("onnx_files_1008"),
                    help="Resolution root (e.g. onnx_files_1008). Reads <onnx-dir>/backbone_detector/.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--output-stem", type=Path, default=Path("outputs/probes/text_probe"))
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.1,
                    help="Mask-IoU threshold for NMS (0 to disable)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.float16
    print(f"Device: {device}, dtype: {dtype}\n")

    # 1. Load SAM3 + processor
    print("[1/5] Loading Sam3VideoModel ...")
    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = Sam3VideoModel.from_pretrained(str(args.checkpoint))
    model = model.to(device).to(dtype).eval()
    detector = model.detector_model
    pos_enc_module = detector.vision_encoder.neck.position_encoding
    torch.cuda.synchronize()
    print(f"      took {time.perf_counter() - t:.1f}s\n")

    # 2. Load MIGraphX backbone
    print(f"[2/5] Loading MIGraphX backbone from {args.onnx_dir / "backbone_detector" / "tuned.mxr"} ...")
    mxr = MIGraphXBackbone(
        onnx_path=args.onnx_dir / "backbone_detector" / "single_simplified.onnx",
        cache_path=args.onnx_dir / "backbone_detector" / "tuned.mxr",
    )
    mxr.warmup(n=3)
    print()

    # 3. Preprocess
    img_pil = Image.open(args.image).convert("RGB")
    img_bgr = cv2.imread(str(args.image))
    H, W = img_pil.height, img_pil.width
    print(f"[3/5] Image: {W}x{H}, text='{args.text}'\n")

    session = processor.init_video_session(video=[img_pil], inference_device=device, dtype=dtype)
    pixel_values = session.get_frame(0).unsqueeze(0)  # (1, 3, 1008, 1008) fp16

    text_inputs = processor.tokenizer(
        args.text, return_tensors="pt", padding="max_length", max_length=32,
    ).to(device)

    # 4. Time both paths
    print(f"[4/5] Timing (warmup={args.warmup}, iters={args.iters}):")

    pt_times = []
    out_pt = None
    for i in range(args.warmup + args.iters):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            ve_pt = detector.vision_encoder(pixel_values)
            out_pt = detector(
                vision_embeds=ve_pt,
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
        torch.cuda.synchronize()
        pt_times.append(time.perf_counter() - t)
    pt_mean = statistics.mean(pt_times[args.warmup:]) * 1000
    print(f"      PyTorch backbone + detector: {pt_mean:.1f} ms (mean of {args.iters})")

    mxr_times = []
    out_mxr = None
    for i in range(args.warmup + args.iters):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            ve_mxr = build_mxr_vision_features(pixel_values, mxr, pos_enc_module)
            out_mxr = detector(
                vision_embeds=ve_mxr,
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
        torch.cuda.synchronize()
        mxr_times.append(time.perf_counter() - t)
    mxr_mean = statistics.mean(mxr_times[args.warmup:]) * 1000
    print(f"      MIGraphX backbone + detector: {mxr_mean:.1f} ms (mean of {args.iters})")
    speedup = pt_mean / mxr_mean if mxr_mean > 0 else float("nan")
    print(f"      Speedup: {speedup:.2f}x  ({pt_mean - mxr_mean:.0f} ms saved)\n")

    # 5. Postproc + viz + comparison
    print("[5/5] Postprocessing (presence-aware scores + mask-IoU NMS) and comparing ...")

    presence_pt = float(torch.sigmoid(out_pt.presence_logits[0, 0]))
    presence_mxr = float(torch.sigmoid(out_mxr.presence_logits[0, 0]))
    print(f"      presence:  PyTorch={presence_pt:.3f}  MIGraphX={presence_mxr:.3f}")

    det_pt = postprocess_detector_outputs(
        out_pt.pred_logits, out_pt.presence_logits,
        out_pt.pred_masks, out_pt.pred_boxes,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou,
    )
    det_mxr = postprocess_detector_outputs(
        out_mxr.pred_logits, out_mxr.presence_logits,
        out_mxr.pred_masks, out_mxr.pred_boxes,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou,
    )

    print(f"      Detections kept: PyTorch={len(det_pt)}  MIGraphX={len(det_mxr)}")
    for i, d in enumerate(det_pt[:5]):
        print(f"        PT  #{i}: score={d.score:.3f}")
    for i, d in enumerate(det_mxr[:5]):
        print(f"        MXR #{i}: score={d.score:.3f}")

    # IoU between top detections (paired by rank)
    pair_n = min(len(det_pt), len(det_mxr))
    if pair_n > 0:
        print(f"\n      Pairwise mask IoU (top {pair_n} by score, both paths use same postproc):")
        for i in range(pair_n):
            mask_pt = mask_logits_to_image(det_pt[i].mask_logits, (H, W))
            mask_mxr = mask_logits_to_image(det_mxr[i].mask_logits, (H, W))
            iou = mask_iou_pair(mask_pt, mask_mxr)
            score_diff = abs(det_pt[i].score - det_mxr[i].score)
            tag = "OK" if iou > 0.9 else ("FAIR" if iou > 0.7 else "FAIL")
            print(f"        rank {i}: IoU={iou:.4f}  score Δ={score_diff:.4f}  [{tag}]")

    # Visualize
    pt_path = args.output_stem.with_name(args.output_stem.name + "_pt.jpg")
    mxr_path = args.output_stem.with_name(args.output_stem.name + "_mxr.jpg")
    cv2.imwrite(str(pt_path), overlay_detections(img_bgr, det_pt, args.text, "PyTorch"))
    cv2.imwrite(str(mxr_path), overlay_detections(img_bgr, det_mxr, args.text, "MIGraphX"))
    print(f"\nSaved: {pt_path}\n       {mxr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
