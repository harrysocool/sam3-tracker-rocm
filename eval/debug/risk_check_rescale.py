#!/usr/bin/env python3
"""Risk check: does MIG backbone fail PT detector because of magnitude only,
or is the difference non-linear?

Test: feed PT detector with (1) raw MIG features, (2) MIG features rescaled
per-level to match PT global mean/std, (3) MIG features rescaled per-channel.
Report presence + top pred_logits in each case.

If rescale (2) or (3) brings presence back to ~PT level, the difference is
linear/scale-only — MIG-izing detector with consistent quantize_fp16 will
likely fix it. If presence stays low, the difference is non-linear and we
need a different approach.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from tracker.migraphx_runtime import MIGraphXBackbone


def rescale_global(mig: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Match overall mean/std across all elements."""
    mig_n = (mig - mig.mean()) / (mig.std() + 1e-6)
    return mig_n * pt.std() + pt.mean()


def rescale_per_channel(mig: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Match per-channel mean/std (channels = dim 1)."""
    # Reduce over (B, H, W) keeping channel
    dims = (0, 2, 3)
    mig_mean = mig.mean(dim=dims, keepdim=True)
    mig_std = mig.std(dim=dims, keepdim=True) + 1e-6
    pt_mean = pt.mean(dim=dims, keepdim=True)
    pt_std = pt.std(dim=dims, keepdim=True)
    return ((mig - mig_mean) / mig_std) * pt_std + pt_mean


def run_detector(detector, fpn_levels, pe_levels, text_inputs):
    ve = Sam3VisionEncoderOutput(
        fpn_hidden_states=tuple(fpn_levels),
        fpn_position_encoding=tuple(pe_levels),
    )
    with torch.inference_mode():
        out = detector(
            vision_embeds=ve,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )
    return out


def report(label, out):
    pres = float(torch.sigmoid(out.presence_logits[0, 0]))
    pres_logit = float(out.presence_logits[0, 0])
    pl_max = float(out.pred_logits.max())
    pl_top5 = sorted(out.pred_logits.flatten().detach().float().cpu().tolist(), reverse=True)[:5]
    print(f"  {label:<32s}  presence={pres:.4f} (logit={pres_logit:+.2f})  pred_logits.max={pl_max:+.3f}  top5={[f'{x:+.2f}' for x in pl_top5]}")


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

    # PyTorch backbone — reference
    print("\nRunning PyTorch backbone ...")
    with torch.inference_mode():
        ve_pt = detector.vision_encoder(pixel_values)
    fpn_pt = list(ve_pt.fpn_hidden_states)
    pe_pt = list(ve_pt.fpn_position_encoding)

    # MIG backbone
    print("Running MIGraphX backbone ...")
    np_in = pixel_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
    fpn_raw = mxr(np_in)
    fpn_mig = [
        torch.from_numpy(np.ascontiguousarray(t)).to(device=device, dtype=dtype)
        for t in fpn_raw
    ]
    pe_mig = [pos_enc_module(f.shape, f.device, f.dtype) for f in fpn_mig]

    print("\nFPN level stats (PT vs MIG):")
    for i in range(4):
        a, b = fpn_pt[i], fpn_mig[i]
        print(f"  fpn[{i}]: PT std={a.std().item():.4f} mean={a.mean().item():+.4f} | "
              f"MIG std={b.std().item():.4f} mean={b.mean().item():+.4f}  (ratio std={b.std().item()/a.std().item():.3f})")

    # Test 1: PT baseline
    print("\n=== Test 1: pure PyTorch (reference) ===")
    out = run_detector(detector, fpn_pt, pe_pt, text_inputs)
    report("PT backbone + PT detector", out)

    # Test 2: raw MIG
    print("\n=== Test 2: raw MIG features ===")
    out = run_detector(detector, fpn_mig, pe_mig, text_inputs)
    report("MIG raw", out)

    # Test 3: global rescale per level
    print("\n=== Test 3: MIG features rescaled per-level (global mean/std match PT) ===")
    fpn_resc_g = [rescale_global(fpn_mig[i], fpn_pt[i]) for i in range(4)]
    out = run_detector(detector, fpn_resc_g, pe_mig, text_inputs)
    report("MIG global-rescaled", out)
    for i in range(4):
        a, b = fpn_resc_g[i], fpn_pt[i]
        diff = (a - b).abs()
        print(f"    fpn[{i}] post-rescale |diff| max={diff.max().item():.3f} mean={diff.mean().item():.4f}")

    # Test 4: per-channel rescale
    print("\n=== Test 4: MIG features rescaled per-channel ===")
    fpn_resc_c = [rescale_per_channel(fpn_mig[i], fpn_pt[i]) for i in range(4)]
    out = run_detector(detector, fpn_resc_c, pe_mig, text_inputs)
    report("MIG per-channel rescaled", out)
    for i in range(4):
        a, b = fpn_resc_c[i], fpn_pt[i]
        diff = (a - b).abs()
        print(f"    fpn[{i}] post-rescale |diff| max={diff.max().item():.3f} mean={diff.mean().item():.4f}")

    # Test 5: substitute MIG features one at a time
    print("\n=== Test 5: substitute single MIG level into PT (with per-channel rescale) ===")
    for k in range(4):
        fpn_mix = list(fpn_pt)
        fpn_mix[k] = rescale_per_channel(fpn_mig[k], fpn_pt[k])
        out = run_detector(detector, fpn_mix, pe_pt, text_inputs)
        report(f"swap fpn[{k}] (per-chan rescale)", out)

    # Test 6: same but with global rescale only
    print("\n=== Test 6: substitute single MIG level (global rescale only) ===")
    for k in range(4):
        fpn_mix = list(fpn_pt)
        fpn_mix[k] = rescale_global(fpn_mig[k], fpn_pt[k])
        out = run_detector(detector, fpn_mix, pe_pt, text_inputs)
        report(f"swap fpn[{k}] (global rescale)", out)


if __name__ == "__main__":
    main()
