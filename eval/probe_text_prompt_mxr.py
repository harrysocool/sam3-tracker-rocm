#!/usr/bin/env python3
"""
Stage 3 integration probe: SAM3 text→mask using our MIGraphX backbone in
place of the PyTorch ViT backbone.

Approach: bypass Sam3VideoModel (which needs `last_hidden_state` for its
tracker_neck — not exported by our current MIGraphX .mxr) and call
`Sam3Model.forward(vision_embeds=..., text_embeds=...)` directly. That's the
official single-image `text -> mask` entry point and is sufficient to
demonstrate the backbone speedup and mask quality.

Side-by-side timing vs. the pure-PyTorch path (probe_text_prompt.py).

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


def build_mxr_vision_features(
    pixel_values: torch.Tensor,
    mxr_backbone: MIGraphXBackbone,
    pos_enc_module: torch.nn.Module,
) -> Sam3VisionEncoderOutput:
    """Run our MIGraphX backbone and wrap output in Sam3VisionEncoderOutput.

    pixel_values: (1, 3, H, W) torch fp16/fp32 on cuda
    Returns Sam3VisionEncoderOutput with 4-level FPN (4th is dummy — Sam3Model
    discards it via `[:-1]`).
    """
    device = pixel_values.device
    dtype = pixel_values.dtype

    # MIGraphX wants float32 numpy on CPU
    np_in = pixel_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
    fpn0, fpn1, fpn2, _ = mxr_backbone(np_in)
    # back to torch on cuda matching the model dtype
    fpn = [
        torch.from_numpy(np.ascontiguousarray(t)).to(device=device, dtype=dtype)
        for t in (fpn0, fpn1, fpn2)
    ]

    pe = [pos_enc_module(f.shape, f.device, f.dtype) for f in fpn]

    # Dummy 4th level — Sam3Model.forward slices [:-1] so values don't matter,
    # but the tuple must have the expected length
    h4, w4 = fpn[2].shape[2] // 2, fpn[2].shape[3] // 2
    dummy = torch.zeros(1, 256, h4, w4, device=device, dtype=dtype)
    dummy_pe = pos_enc_module(dummy.shape, device, dtype)

    return Sam3VisionEncoderOutput(
        fpn_hidden_states=tuple(fpn) + (dummy,),
        fpn_position_encoding=tuple(pe) + (dummy_pe,),
    )


def detector_inference(
    detector_model,
    vision_embeds: Sam3VisionEncoderOutput,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    score_threshold: float = 0.5,
) -> tuple[list[np.ndarray], list[float]]:
    """Run text→mask via Sam3Model.forward, return masks above score threshold."""
    with torch.inference_mode():
        out = detector_model(
            vision_embeds=vision_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    # out has pred_masks, pred_boxes, pred_scores (or similar — print first time to confirm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/home/amd/project/sam3/model/sam3"))
    ap.add_argument("--onnx-dir", type=Path,
                    default=Path("onnx_files_1008"),
                    help="Dir with backbone_mxr_tuned.mxr at SAM3 default 1008px")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--output", type=Path, default=Path("text_probe_mxr_out.jpg"))
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    print(f"Device: {device}, dtype: {dtype}\n")

    # 1. Load SAM3 + processor (we'll use detector_model + tokenizer)
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
    print(f"[2/5] Loading MIGraphX backbone from {args.onnx_dir} ...")
    mxr = MIGraphXBackbone(
        onnx_path=args.onnx_dir / "backbone_single_simplified.onnx",
        cache_path=args.onnx_dir / "backbone_mxr_tuned.mxr",
    )
    mxr.warmup(n=3)
    print()

    # 3. Preprocess via SAM3 processor (gives us pixel_values at the right size + normalization)
    img_pil = Image.open(args.image).convert("RGB")
    img_bgr = cv2.imread(str(args.image))
    H, W = img_pil.height, img_pil.width
    print(f"[3/5] Image: {W}x{H}, text='{args.text}'\n")

    session = processor.init_video_session(video=[img_pil], inference_device=device, dtype=dtype)
    pixel_values = session.get_frame(0).unsqueeze(0)  # (1, 3, 1008, 1008) fp16

    # Tokenize text
    text_inputs = processor.tokenizer(
        args.text, return_tensors="pt", padding="max_length", max_length=32,
    ).to(device)

    # 4. Time both paths side-by-side
    print(f"[4/5] Timing (warmup={args.warmup}, iters={args.iters}):")

    # Path A: pure PyTorch (vision_encoder + detector head)
    pt_times = []
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

    # Path B: MIGraphX backbone + detector
    mxr_times = []
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

    # 5. Inspect / visualize MIGraphX-path output
    print("[5/5] Output structure:")
    pred_logits = out_mxr.pred_logits     # (1, 200) raw scores
    pred_masks = out_mxr.pred_masks       # (1, 200, 288, 288) logits
    presence = torch.sigmoid(out_mxr.presence_logits).item()
    print(f"      pred_logits: shape={tuple(pred_logits.shape)}, max={float(pred_logits.max()):.3f}")
    print(f"      pred_masks:  shape={tuple(pred_masks.shape)} dtype={pred_masks.dtype}")
    print(f"      presence:    {presence:.3f}")

    # Apply sigmoid → probabilities, threshold to keep candidates
    scores = torch.sigmoid(pred_logits).detach().float().cpu().numpy().squeeze()
    keep = np.where(scores >= args.score_threshold)[0]
    print(f"      Detections above {args.score_threshold}: {len(keep)}  (top-5 scores={sorted(scores, reverse=True)[:5]})")

    # Visualize
    vis = img_bgr.copy()
    palette = [(0, 200, 80), (200, 80, 0), (80, 0, 200), (200, 200, 0), (0, 200, 200)]
    for i, det_idx in enumerate(keep):
        mask = pred_masks[0, det_idx].detach().float().cpu().numpy()
        mask = (mask > 0).astype(bool)
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        score = float(scores[det_idx])
        color = palette[i % len(palette)]
        layer = np.zeros_like(vis); layer[mask] = color
        vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        if contours:
            x, y, _w, _h = cv2.boundingRect(contours[0])
            cv2.putText(vis, f"#{i} {score:.2f}", (x, max(y - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(vis, f"text='{args.text}'  N={len(keep)}  MIGraphX backbone",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.imwrite(str(args.output), vis)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
