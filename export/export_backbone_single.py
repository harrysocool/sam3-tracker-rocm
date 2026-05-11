#!/usr/bin/env python3
"""Export Sam3TrackerVideoModel vision_encoder as a single-session ONNX backbone.

Produces one fp32 ONNX (`backbone_single_fp32.onnx` by default) that takes
`pixel_values` (1, 3, H, H) and returns the three FPN levels
(`fpn_0`, `fpn_1`, `fpn_2`) — identical contract to the 3-session pipeline,
but as one graph for better MIGraphX kernel fusion.

Pipeline:
    [export_backbone_single.py]  ->  backbone_single_fp32.onnx
    [simplify_backbone.py]       ->  backbone_single_simplified.onnx
    [compile_backbone_mxr.py]    ->  backbone_mxr_tuned.mxr  (the runtime cache)

Usage:
    python export/export_backbone_single.py --imgsz 504 --output-dir onnx_files
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, default=WORKSPACE / "model" / "sam3")
    p.add_argument("--output-dir", type=Path, default=WORKSPACE / "onnx_files")
    p.add_argument("--imgsz", type=int, default=504)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--output-name",
        type=str,
        default="backbone_single_fp32.onnx",
        help="Output filename inside --output-dir",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Run CPU ONNX vs PyTorch comparison after export",
    )
    return p.parse_args()


class SingleSessionBackbone(nn.Module):
    """Whole vision_encoder (ViT backbone + FPN neck) as one graph.

    Returns the three FPN levels in the order tracker.py expects:
        fpn_0: (1, 256, 4*P, 4*P)   high-res
        fpn_1: (1, 256, 2*P, 2*P)   mid-res
        fpn_2: (1, 256,   P,   P)   image embedding
    where P = imgsz // 14.
    """

    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values):
        out = self.vision_encoder(pixel_values, return_dict=True)
        f0, f1, f2 = out.fpn_hidden_states[:3]
        return f0, f1, f2


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name

    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoModel,
    )
    from tracker.tracker import retarget_resolution

    print(f"Loading Sam3TrackerVideoModel from {args.checkpoint} ...")
    model = (
        Sam3TrackerVideoModel.from_pretrained(
            str(args.checkpoint), attn_implementation="eager"
        )
        .cpu()
        .eval()
    )

    if args.imgsz != 1008:
        retarget_resolution(model, args.imgsz)

    P = args.imgsz // 14
    print(f"  imgsz={args.imgsz}px  feature map={P}x{P}")

    wrapper = SingleSessionBackbone(model.vision_encoder).eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)

    print("\nExporting ...")
    with torch.no_grad():
        f0, f1, f2 = wrapper(dummy)
    print(
        f"  Output shapes: fpn_0={tuple(f0.shape)} "
        f"fpn_1={tuple(f1.shape)} fpn_2={tuple(f2.shape)}"
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["pixel_values"],
            output_names=["fpn_0", "fpn_1", "fpn_2"],
        )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path}  ({size_mb:.0f} MB)")

    # ONNX > 2GB triggers external-data files (.data sidecar). Note this for the user.
    data_path = out_path.with_suffix(out_path.suffix + ".data")
    if data_path.exists():
        data_mb = data_path.stat().st_size / 1e6
        print(f"  External weights: {data_path.name}  ({data_mb:.0f} MB)")

    if args.verify:
        print("\n[verify] Comparing ONNX vs PyTorch outputs ...")
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
        o0, o1, o2 = sess.run(None, {"pixel_values": x})
        print(f"  ONNX: fpn_0={o0.shape}  fpn_1={o1.shape}  fpn_2={o2.shape}")

        pv = torch.from_numpy(x)
        with torch.inference_mode():
            vis = model.vision_encoder(pv, return_dict=True)
        for i, (onnx_out, pt_out) in enumerate(
            zip([o0, o1, o2], vis.fpn_hidden_states[:3])
        ):
            diff = np.abs(onnx_out - pt_out.numpy()).max()
            print(f"  fpn_{i} max_diff: {diff:.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
