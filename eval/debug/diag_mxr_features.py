#!/usr/bin/env python3
"""Diagnose why MIGraphX backbone output produces presence=0 in detector.

Compares PyTorch vs MIGraphX FPN features at three levels:
  1. raw FPN tensor stats (shape, dtype, range, mean, std)
  2. abs/relative diff between PT and MIG features
  3. detector outputs (pred_logits, presence_logits, pred_masks stats)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from tracker.tracker import MIGraphXBackbone


def stats(t: torch.Tensor, name: str):
    f = t.detach().float()
    print(f"  {name:<24s} shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={f.min().item():+.3f} max={f.max().item():+.3f} "
          f"mean={f.mean().item():+.3f} std={f.std().item():.3f} "
          f"|nan|={int(torch.isnan(f).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("/home/amd/project/sam3/model/sam3"))
    ap.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_1008"))
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--text", type=str, default="truck")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    print("Loading model + backbone ...")
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = Sam3VideoModel.from_pretrained(str(args.checkpoint))
    model = model.to(device).to(dtype).eval()
    detector = model.detector_model
    pos_enc_module = detector.vision_encoder.neck.position_encoding

    mxr = MIGraphXBackbone(
        onnx_path=args.onnx_dir / "backbone_detector" / "single_simplified.onnx",
        cache_path=args.onnx_dir / "backbone_detector" / "tuned.mxr",
    )
    mxr.warmup(n=2)

    img_pil = Image.open(args.image).convert("RGB")
    session = processor.init_video_session(video=[img_pil], inference_device=device, dtype=dtype)
    pixel_values = session.get_frame(0).unsqueeze(0)
    text_inputs = processor.tokenizer(args.text, return_tensors="pt", padding="max_length", max_length=32).to(device)

    print(f"\npixel_values: shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}")
    stats(pixel_values, "pixel_values")

    # PyTorch path
    print("\n=== PyTorch backbone ===")
    with torch.inference_mode():
        ve_pt = detector.vision_encoder(pixel_values)
    print(f"  has fpn_hidden_states ({len(ve_pt.fpn_hidden_states)} levels):")
    for i, t in enumerate(ve_pt.fpn_hidden_states):
        stats(t, f"fpn[{i}]")
    print(f"  has fpn_position_encoding ({len(ve_pt.fpn_position_encoding)} levels):")
    for i, t in enumerate(ve_pt.fpn_position_encoding):
        stats(t, f"pe[{i}]")

    # MIGraphX path
    print("\n=== MIGraphX backbone ===")
    np_in = pixel_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
    fpn_raw = mxr(np_in)
    print(f"  raw mxr output: {len(fpn_raw)} arrays, shapes={[t.shape for t in fpn_raw]}, dtypes={[t.dtype for t in fpn_raw]}")
    for i, t in enumerate(fpn_raw):
        f = t.astype(np.float32)
        print(f"  raw[{i}]: min={f.min():+.3f} max={f.max():+.3f} mean={f.mean():+.3f} std={f.std():.3f} |nan|={int(np.isnan(f).sum())}")

    fpn_mxr = [
        torch.from_numpy(np.ascontiguousarray(t)).to(device=device, dtype=dtype)
        for t in fpn_raw[:3]
    ]
    pe_mxr = [pos_enc_module(f.shape, f.device, f.dtype) for f in fpn_mxr]

    print("  on cuda+fp16:")
    for i, t in enumerate(fpn_mxr):
        stats(t, f"fpn[{i}]")
    for i, t in enumerate(pe_mxr):
        stats(t, f"pe[{i}]")

    # diff per level
    print("\n=== Diff (PyTorch vs MIGraphX), per FPN level ===")
    for i in range(3):
        a = ve_pt.fpn_hidden_states[i].detach().float()
        b = fpn_mxr[i].detach().float()
        if a.shape != b.shape:
            print(f"  fpn[{i}] SHAPE MISMATCH: pt={tuple(a.shape)}  mxr={tuple(b.shape)}")
            continue
        diff = (a - b).abs()
        print(f"  fpn[{i}] {tuple(a.shape)}  |diff| max={diff.max().item():.3f} "
              f"mean={diff.mean().item():.4f}  rel mean={diff.mean().item() / (a.abs().mean().item() + 1e-9):.3f}")

    for i in range(3):
        a = ve_pt.fpn_position_encoding[i].detach().float()
        b = pe_mxr[i].detach().float()
        if a.shape != b.shape:
            print(f"  pe[{i}] SHAPE MISMATCH: pt={tuple(a.shape)}  mxr={tuple(b.shape)}")
            continue
        diff = (a - b).abs()
        print(f"  pe[{i}]  {tuple(a.shape)}  |diff| max={diff.max().item():.3f} mean={diff.mean().item():.4f}")

    # Run detector on each
    print("\n=== Detector outputs ===")
    h4, w4 = fpn_mxr[2].shape[2] // 2, fpn_mxr[2].shape[3] // 2
    dummy = torch.zeros(1, 256, h4, w4, device=device, dtype=dtype)
    dummy_pe = pos_enc_module(dummy.shape, device, dtype)
    ve_mxr = Sam3VisionEncoderOutput(
        fpn_hidden_states=tuple(fpn_mxr) + (dummy,),
        fpn_position_encoding=tuple(pe_mxr) + (dummy_pe,),
    )

    with torch.inference_mode():
        out_pt = detector(vision_embeds=ve_pt, input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        out_mxr = detector(vision_embeds=ve_mxr, input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)

    print(f"  PT  presence_logits = {float(out_pt.presence_logits[0,0]):+.3f}  -> {torch.sigmoid(out_pt.presence_logits[0,0]).item():.4f}")
    print(f"  MXR presence_logits = {float(out_mxr.presence_logits[0,0]):+.3f}  -> {torch.sigmoid(out_mxr.presence_logits[0,0]).item():.4f}")
    print(f"  PT  pred_logits  max={out_pt.pred_logits.max().item():+.3f}")
    print(f"  MXR pred_logits  max={out_mxr.pred_logits.max().item():+.3f}")

    # Check if substituting only ONE FPN level breaks it
    print("\n=== Ablation: substitute one MXR FPN level at a time into PT features ===")
    for k in range(3):
        fpn_hybrid = list(ve_pt.fpn_hidden_states)
        fpn_hybrid[k] = fpn_mxr[k]
        pe_hybrid = list(ve_pt.fpn_position_encoding)
        ve_hybrid = Sam3VisionEncoderOutput(
            fpn_hidden_states=tuple(fpn_hybrid),
            fpn_position_encoding=tuple(pe_hybrid),
        )
        with torch.inference_mode():
            out_h = detector(vision_embeds=ve_hybrid, input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        print(f"  swap fpn[{k}] only: presence={torch.sigmoid(out_h.presence_logits[0,0]).item():.4f}  "
              f"pred_logits.max={out_h.pred_logits.max().item():+.3f}")


if __name__ == "__main__":
    sys.exit(main())
