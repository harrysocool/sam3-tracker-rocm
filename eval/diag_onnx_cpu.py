#!/usr/bin/env python3
"""Run the simplified backbone ONNX through ONNXRuntime CPU and compare to PT.

Isolates whether the fpn[2] divergence is introduced by:
  (a) torch.onnx.export — bug in PyTorch ONNX export
  (b) onnxsim simplification — semantic change during simplification
  (c) MIGraphX compile/runtime — bug in MIG (likely candidate per current findings)

If ORT CPU matches PyTorch → bug is in MIG.
If ORT CPU matches MIG → bug is in ONNX/simplification.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel


def stat(name, t):
    f = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else t.astype(np.float32)
    print(f"  {name:<28s} std={f.std():.4f} mean={f.mean():+.4f}  min={f.min():+.3f} max={f.max():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("/home/amd/project/sam3/model/sam3"))
    ap.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_1008"))
    ap.add_argument("--onnx-name", type=str, default="backbone_single_simplified.onnx")
    ap.add_argument("--image", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = Sam3VideoModel.from_pretrained(str(args.checkpoint))
    model_fp16 = model.to(device).to(dtype).eval()
    detector_fp16 = model_fp16.detector_model

    # also keep an fp32 cpu copy for reference
    print("Loading separate FP32 CPU copy for reference ...")
    model_fp32 = Sam3VideoModel.from_pretrained(str(args.checkpoint)).cpu().eval()
    detector_fp32 = model_fp32.detector_model

    img_pil = Image.open(args.image).convert("RGB")
    session = processor.init_video_session(video=[img_pil], inference_device=device, dtype=dtype)
    pixel_values_fp16 = session.get_frame(0).unsqueeze(0)
    pixel_values_fp32 = pixel_values_fp16.float().cpu()

    print("\n=== PyTorch FP16 (cuda) backbone ===")
    with torch.inference_mode():
        ve_pt16 = detector_fp16.vision_encoder(pixel_values_fp16)
    for i, t in enumerate(ve_pt16.fpn_hidden_states):
        stat(f"fpn[{i}] PT-fp16", t)

    print("\n=== PyTorch FP32 (cpu) backbone ===")
    with torch.inference_mode():
        ve_pt32 = detector_fp32.vision_encoder(pixel_values_fp32)
    for i, t in enumerate(ve_pt32.fpn_hidden_states):
        stat(f"fpn[{i}] PT-fp32", t)

    print(f"\n=== ONNX Runtime CPU on {args.onnx_name} ===")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.onnx_dir / args.onnx_name), providers=["CPUExecutionProvider"])
    np_in = pixel_values_fp32.numpy().astype(np.float32)
    onnx_outs = sess.run(None, {"pixel_values": np_in})
    print(f"  ONNX returned {len(onnx_outs)} outputs, shapes: {[o.shape for o in onnx_outs]}")
    for i, o in enumerate(onnx_outs):
        stat(f"fpn[{i}] ONNX-cpu-fp32", o)

    # Diff: ONNX vs PT-fp32
    print("\n=== Diff: ONNX-cpu-fp32 vs PT-fp32-cpu ===")
    for i, o in enumerate(onnx_outs):
        if i >= len(ve_pt32.fpn_hidden_states):
            print(f"  fpn[{i}] (no PT counterpart at this index)")
            continue
        a = o.astype(np.float32)
        b = ve_pt32.fpn_hidden_states[i].detach().float().cpu().numpy()
        if a.shape != b.shape:
            print(f"  fpn[{i}] SHAPE MISMATCH: ONNX={a.shape} PT={b.shape}")
            continue
        d = np.abs(a - b)
        rel = d.mean() / (np.abs(b).mean() + 1e-9)
        print(f"  fpn[{i}] {a.shape}  |diff| max={d.max():.4f} mean={d.mean():.6f}  rel mean={rel:.4f}")


if __name__ == "__main__":
    main()
