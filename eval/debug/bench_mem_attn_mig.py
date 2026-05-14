#!/usr/bin/env python3
"""Direct micro-bench of MIGMemoryAttention shim per-call latency."""
import time, statistics
from pathlib import Path
import torch
from transformers import Sam3VideoModel
from tracker.mig_memory_attention import MIGMemoryAttention, EXPECTED_SPATIAL_LEN, PTR_TOKENS, HW_1008

device, dtype = torch.device("cuda"), torch.float16

print("Loading model ...")
model = Sam3VideoModel.from_pretrained("/home/amd/project/sam3/model/sam3").to(device).to(dtype).eval()
trk = model.tracker_model
mem_attn_pt = trk.memory_attention

cur_feat = torch.randn(HW_1008, 1, 256, device=device, dtype=dtype)
cur_pos  = torch.randn(HW_1008, 1, 256, device=device, dtype=dtype)
spatial = torch.randn(EXPECTED_SPATIAL_LEN, 1, 64, device=device, dtype=dtype)
spatial_pos = torch.randn(EXPECTED_SPATIAL_LEN, 1, 64, device=device, dtype=dtype)
ptrs = torch.randn(64, 1, 64, device=device, dtype=dtype)
ptrs_pos = torch.randn(64, 1, 64, device=device, dtype=dtype)
memory = torch.cat([spatial, ptrs], dim=0)
mempos = torch.cat([spatial_pos, ptrs_pos], dim=0)

shim = MIGMemoryAttention(Path("onnx_files_1008/tracker_modules/memory_attention_fixed_S7_P64.onnx"),
                           mem_attn_pt.forward)

# Time PT
print("\n=== PT memory_attention ===")
ts = []
for i in range(7):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        out = mem_attn_pt(current_vision_features=cur_feat,
                          current_vision_position_embeddings=cur_pos,
                          memory=memory,
                          memory_posision_embeddings=mempos,
                          num_object_pointer_tokens=64)
    torch.cuda.synchronize()
    ts.append(time.perf_counter()-t0)
print(f"  iters: {[f'{t*1000:.0f}' for t in ts]}")
print(f"  mean (drop first 2): {statistics.mean(ts[2:])*1000:.1f} ms")

# Time MIG
print("\n=== MIG memory_attention (shim) ===")
ts = []
for i in range(7):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        out = shim.forward(
            current_vision_features=cur_feat,
            memory=memory,
            current_vision_position_embeddings=cur_pos,
            memory_posision_embeddings=mempos,
            num_object_pointer_tokens=64,
        )
    torch.cuda.synchronize()
    ts.append(time.perf_counter()-t0)
print(f"  iters: {[f'{t*1000:.0f}' for t in ts]}")
print(f"  mean (drop first 2): {statistics.mean(ts[2:])*1000:.1f} ms")

# Time MIG sub-steps
print("\n=== MIG sub-step breakdown (single warm call) ===")
import numpy as np
t0 = time.perf_counter()
inputs_np = {
    "current_vision_features": cur_feat.detach().float().cpu().numpy(),
    "memory": memory.detach().float().cpu().numpy(),
    "current_vis_pos_embed": cur_pos.detach().float().cpu().numpy(),
    "memory_pos_embed": mempos.detach().float().cpu().numpy(),
}
print(f"  in-prep (torch→numpy): {(time.perf_counter()-t0)*1000:.1f} ms")
torch.cuda.synchronize(); t0 = time.perf_counter()
ort_out = shim.session.run(None, inputs_np)[0]
print(f"  ORT EP run:            {(time.perf_counter()-t0)*1000:.1f} ms")
t0 = time.perf_counter()
out_4d = torch.from_numpy(ort_out).flatten(2).permute(0, 2, 1).unsqueeze(0).to(device=device, dtype=dtype)
torch.cuda.synchronize()
print(f"  out-prep (numpy→torch):{(time.perf_counter()-t0)*1000:.1f} ms")
print(f"  in size: cur_feat {cur_feat.nbytes/1e6:.1f}MB  memory {memory.nbytes/1e6:.1f}MB")
