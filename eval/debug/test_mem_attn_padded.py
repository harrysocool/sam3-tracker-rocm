#!/usr/bin/env python3
"""Numerical sanity: PT memory_attention vs padded MIG path output match.

Constructs a steady-state input (spatial=36288, ptr=N where N≤64), runs both
PT and MIG (with pointer pad to 64), checks max-diff in conditioned features.
"""
import sys
from pathlib import Path
import numpy as np
import torch

from transformers import Sam3VideoModel
from tracker.mig_memory_attention import (
    MIGMemoryAttention, EXPECTED_SPATIAL_LEN, PTR_TOKENS, HW_1008,
)

device, dtype = torch.device("cuda"), torch.float16

print("Loading Sam3VideoModel ...")
model = Sam3VideoModel.from_pretrained("/home/amd/project/sam3/model/sam3").to(device).to(dtype).eval()
trk = model.tracker_model
mem_attn_pt = trk.memory_attention

# Steady state inputs
torch.manual_seed(0)
HW = HW_1008
cur_feat = torch.randn(HW, 1, 256, device=device, dtype=dtype)
cur_pos  = torch.randn(HW, 1, 256, device=device, dtype=dtype)

for N_actual in [4, 16, 32, 56, 64]:
    spatial = torch.randn(EXPECTED_SPATIAL_LEN, 1, 64, device=device, dtype=dtype)
    spatial_pos = torch.randn(EXPECTED_SPATIAL_LEN, 1, 64, device=device, dtype=dtype)
    ptrs = torch.randn(N_actual, 1, 64, device=device, dtype=dtype)
    ptrs_pos = torch.randn(N_actual, 1, 64, device=device, dtype=dtype)
    memory_pt = torch.cat([spatial, ptrs], dim=0)
    mempos_pt = torch.cat([spatial_pos, ptrs_pos], dim=0)

    # PT reference
    with torch.inference_mode():
        out_pt = mem_attn_pt(
            current_vision_features=cur_feat,
            current_vision_position_embeddings=cur_pos,
            memory=memory_pt,
            memory_posision_embeddings=mempos_pt,
            num_object_pointer_tokens=N_actual,
        )

    # MIG path (uses shim)
    onnx_path = Path("onnx_files_1008/tracker_modules/memory_attention_fixed_S7_P64.onnx")
    shim = MIGMemoryAttention(onnx_path, mem_attn_pt) if N_actual == 4 else _shim
    if N_actual == 4: _shim = shim  # reuse session across iters
    with torch.inference_mode():
        out_mig = shim.forward(
            current_vision_features=cur_feat,
            memory=memory_pt,
            current_vision_position_embeddings=cur_pos,
            memory_posision_embeddings=mempos_pt,
            num_object_pointer_tokens=N_actual,
        )

    diff = (out_pt.float() - out_mig.float()).abs()
    print(f"  N={N_actual:3d}  PT shape={tuple(out_pt.shape)} MIG shape={tuple(out_mig.shape)} "
          f"|diff| max={diff.max():.4f} mean={diff.mean():.5f}")
