#!/usr/bin/env python3
"""Sanity-check: a clean nn.Module wrapper around detr_encoder produces the
same outputs as the original Sam3DetrEncoder.forward.

Goal is to bake away the Sam3DETREncoderOutput / kwargs / capture_outputs
machinery so torch.onnx.export traces a pure tensor-in/tensor-out graph.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn

from transformers import Sam3VideoModel
from transformers.models.sam3.modeling_sam3 import create_bidirectional_mask


class DetrEncoderWrapper(nn.Module):
    """Pure tensor-in/tensor-out shim around Sam3DetrEncoder.

    Inputs (matched to how Sam3Model.forward calls it for the text-prompt path):
        vision_feature: (1, 256, H, W)   = fpn_hidden_states[-2] (the 72×72 level at 1008px)
        vision_pos:     (1, 256, H, W)   = matching positional encoding
        text_features:  (1, T, 256)      = pooler_output of CLIP text encoder
        text_mask:      (1, T) bool      = padding mask (True = real token)

    Output:
        last_hidden_state: (1, H*W, 256)
    """

    def __init__(self, detr_encoder):
        super().__init__()
        self.encoder = detr_encoder

    def forward(self, vision_feature, vision_pos, text_features, text_mask):
        bsz, c, h, w = vision_feature.shape
        # Flatten the single FPN level: (1, 256, H, W) -> (1, H*W, 256)
        feat = vision_feature.flatten(2).transpose(1, 2)
        pos  = vision_pos.flatten(2).transpose(1, 2)

        prompt_cross_attn_mask = create_bidirectional_mask(
            config=self.encoder.config,
            inputs_embeds=feat,
            attention_mask=text_mask,
            encoder_hidden_states=text_features,
        )

        hidden = feat
        for layer in self.encoder.layers:
            hidden = layer(
                hidden,
                prompt_feats=text_features,
                vision_pos_encoding=pos,
                prompt_cross_attn_mask=prompt_cross_attn_mask,
            )
        return hidden  # (1, H*W, 256)


def main():
    device = torch.device("cuda")
    dtype = torch.float16
    print("Loading Sam3VideoModel ...")
    model = Sam3VideoModel.from_pretrained("/home/amd/project/sam3/model/sam3").to(device).to(dtype).eval()
    detector = model.detector_model

    # Dummy inputs at 1008px shapes
    H = W = 72
    vision_feature = torch.randn(1, 256, H, W, device=device, dtype=dtype)
    vision_pos     = torch.randn(1, 256, H, W, device=device, dtype=dtype)
    text_features  = torch.randn(1, 32, 256, device=device, dtype=dtype)
    text_mask      = torch.ones(1, 32, dtype=torch.bool, device=device)
    text_mask[0, 5:] = False  # simulate "5 real tokens, rest padding"

    # ----- Reference: call PT detr_encoder directly the way Sam3Model does -----
    with torch.inference_mode():
        ref = detector.detr_encoder(
            vision_features=[vision_feature],
            text_features=text_features,
            vision_pos_embeds=[vision_pos],
            text_mask=text_mask,
        )
    ref_lhs = ref.last_hidden_state  # (1, H*W, 256)

    # ----- Our wrapper -----
    wrapper = DetrEncoderWrapper(detector.detr_encoder).eval()
    with torch.inference_mode():
        our_lhs = wrapper(vision_feature, vision_pos, text_features, text_mask)

    diff = (ref_lhs.float() - our_lhs.float()).abs()
    print(f"\nReference: shape={tuple(ref_lhs.shape)} mean={ref_lhs.float().mean():+.4f} std={ref_lhs.float().std():.4f}")
    print(f"Ours:      shape={tuple(our_lhs.shape)} mean={our_lhs.float().mean():+.4f} std={our_lhs.float().std():.4f}")
    print(f"|diff|     max={diff.max():.5f}  mean={diff.mean():.6f}")
    if diff.max() < 1e-2:
        print("OK — wrapper matches PT reference within FP16 noise.")
    else:
        print("FAIL — wrapper disagrees with PT reference. Investigate.")


if __name__ == "__main__":
    main()
