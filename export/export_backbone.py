#!/usr/bin/env python3
"""Export Sam3TrackerVideoModel vision_encoder to 3-session ONNX backbone.

Outputs raw 256-channel FPN features, directly compatible with the existing
mask_decoder_init.onnx / mask_decoder_propagate.onnx / memory_encoder.onnx.

Sessions (with numpy BHWD->BCHW permute between Part1/Block31 and FPN):
  backbone_part1.onnx  : pixel_values (1,3,H,H) -> BHWD features (1,36,36,1024)
  backbone_block31.onnx: BHWD -> BHWD
  backbone_fpn.onnx    : BCHW -> fpn_0 (1,256,4H,4H), fpn_1 (1,256,2H,2H),
                                  fpn_2 (1,256,H,H)

The numpy permute between sessions avoids the MIGraphX Transpose+ConvTranspose
layout bug in the FPN neck.

Usage:
    python export/export_backbone.py --imgsz 504 --output-dir onnx_files
"""

from __future__ import annotations
import argparse, sys, os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=WORKSPACE / 'model' / 'sam3')
    p.add_argument('--output-dir', type=Path, default=WORKSPACE / 'onnx_files')
    p.add_argument('--imgsz',   type=int, default=504)
    p.add_argument('--n-split', type=int, default=31)
    p.add_argument('--opset',   type=int, default=17)
    p.add_argument('--verify',  action='store_true')
    return p.parse_args()


class Part1Wrapper(nn.Module):
    """pixel_values -> BHWD hidden features (embeddings + layer_norm + layers 0..n_split-1)."""
    def __init__(self, backbone, n_split):
        super().__init__()
        self.embeddings  = backbone.embeddings
        self.layer_norm  = backbone.layer_norm
        self.layers      = nn.ModuleList(backbone.layers[:n_split])
        self._patch_size = backbone.config.patch_size

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)           # (B, seq, C)
        B, seq, C = x.shape
        H = pixel_values.shape[-2] // self._patch_size
        W = pixel_values.shape[-1] // self._patch_size
        x = x.view(B, H, W, C)                     # (B, H, W, C) BHWD
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x                                    # (B, H, W, C)


class Block31Wrapper(nn.Module):
    """Single ViT block. Input/output: BHWD (B, H, W, C)."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)


class FPNWrapper(nn.Module):
    """FPN neck only (no conv_s0/s1). Outputs raw 256-ch FPN features.

    Input:  x_bchw (1, 1024, H, W)
    Output: fpn_0 (1, 256, 4H, 4H) -- feat_s0 (high res)
            fpn_1 (1, 256, 2H, 2H) -- feat_s1 (mid res)
            fpn_2 (1, 256, H,  H)  -- main image embedding
    """
    def __init__(self, neck):
        super().__init__()
        self.neck = neck

    def forward(self, x_bchw):
        fpn_outs, _ = self.neck(x_bchw)
        return fpn_outs[0], fpn_outs[1], fpn_outs[2]


def _export(wrapper, dummy, path, opset, in_names, out_names):
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, str(path),
                          opset_version=opset, dynamo=False,
                          input_names=in_names, output_names=out_names)
    print(f'  Saved: {path.name}  ({path.stat().st_size/1e6:.1f} MB)')


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoModel
    from tracker.tracker import retarget_resolution

    print(f'Loading Sam3TrackerVideoModel from {args.checkpoint} ...')
    model = Sam3TrackerVideoModel.from_pretrained(
        str(args.checkpoint), attn_implementation='eager').cpu().eval()

    if args.imgsz != 1008:
        retarget_resolution(model, args.imgsz)

    backbone = model.vision_encoder.backbone
    H = args.imgsz // 14
    print(f'  imgsz={args.imgsz}px  feature map={H}x{H}')

    # [1/3] Part1
    print('\n[1/3] Exporting backbone_part1.onnx ...')
    w1 = Part1Wrapper(backbone, args.n_split).eval()
    dummy_pv = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        out1 = w1(dummy_pv)
    print(f'  Part1: {tuple(dummy_pv.shape)} -> {tuple(out1.shape)}')
    _export(w1, (dummy_pv,), args.output_dir / 'backbone_part1.onnx',
            args.opset, ['pixel_values'], ['bhwd_features'])

    # [2/3] Block31
    print('\n[2/3] Exporting backbone_block31.onnx ...')
    w2 = Block31Wrapper(backbone.layers[args.n_split]).eval()
    dummy_bhwd = out1.detach()
    with torch.no_grad():
        out2 = w2(dummy_bhwd)
    print(f'  Block31: {tuple(dummy_bhwd.shape)} -> {tuple(out2.shape)}')
    _export(w2, (dummy_bhwd,), args.output_dir / 'backbone_block31.onnx',
            args.opset, ['x_bhwd'], ['y_bhwd'])

    # [3/3] FPN
    print('\n[3/3] Exporting backbone_fpn.onnx ...')
    w3 = FPNWrapper(model.vision_encoder.neck).eval()
    bchw = out2.detach().permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        f0, f1, f2 = w3(bchw)
    print(f'  FPN: {tuple(bchw.shape)} -> {tuple(f0.shape)}, {tuple(f1.shape)}, {tuple(f2.shape)}')
    _export(w3, (bchw,), args.output_dir / 'backbone_fpn.onnx',
            args.opset, ['x_bchw'], ['fpn_0', 'fpn_1', 'fpn_2'])

    print(f'\nExported to {args.output_dir}:')
    for name in ['backbone_part1.onnx', 'backbone_block31.onnx', 'backbone_fpn.onnx']:
        p = args.output_dir / name
        s = '✅' if p.exists() else '❌'
        print(f'  {s}  {name}  ({p.stat().st_size/1e6:.1f} MB)' if p.exists() else f'  {s}  {name}')

    if args.verify:
        print('\n[verify] Comparing ONNX vs PyTorch outputs ...')
        import onnxruntime as ort
        CPU = ['CPUExecutionProvider']
        s1 = ort.InferenceSession(str(args.output_dir / 'backbone_part1.onnx'),  providers=CPU)
        s2 = ort.InferenceSession(str(args.output_dir / 'backbone_block31.onnx'), providers=CPU)
        s3 = ort.InferenceSession(str(args.output_dir / 'backbone_fpn.onnx'),    providers=CPU)

        x = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
        o1 = s1.run(None, {'pixel_values': x})[0]
        o2 = s2.run(None, {'x_bhwd': o1})[0]
        o2c = np.transpose(o2, (0, 3, 1, 2))           # BHWD -> BCHW
        f0o, f1o, f2o = s3.run(None, {'x_bchw': o2c})
        print(f'  ONNX: fpn_0={f0o.shape}  fpn_1={f1o.shape}  fpn_2={f2o.shape}')

        pv = torch.from_numpy(x)
        with torch.inference_mode():
            vis = model.vision_encoder(pv, return_dict=True)
        pt_f0 = vis.fpn_hidden_states[0].numpy()
        pt_f1 = vis.fpn_hidden_states[1].numpy()
        pt_f2 = vis.fpn_hidden_states[2].numpy()
        print(f'  fpn_0 max_diff: {np.abs(f0o - pt_f0).max():.5f}')
        print(f'  fpn_1 max_diff: {np.abs(f1o - pt_f1).max():.5f}')
        print(f'  fpn_2 max_diff: {np.abs(f2o - pt_f2).max():.5f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
