#!/usr/bin/env python3
"""Probe memory_attention input shapes per frame in Sam3VideoModel.

Decides what fixed object-pointer count K to bake into the ONNX export.
Reports per-frame: vision shape, memory shape, num_object_pointer_tokens.
"""
from collections import Counter
from pathlib import Path
import sys, time

import cv2
import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel
from tracker.tracker import MIGraphXBackbone
from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder

device, dtype = torch.device("cuda"), torch.float16
text_prompt = sys.argv[1] if len(sys.argv) > 1 else "swan"
n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 11

print(f"Probe: text='{text_prompt}', frames={n_frames}\n")

proc = AutoProcessor.from_pretrained("/home/amd/project/sam3/model/sam3")
model = Sam3VideoModel.from_pretrained("/home/amd/project/sam3/model/sam3").to(device).to(dtype).eval()
patch_sam3_video_model_with_mig(model,
    MIGraphXBackbone(Path("onnx_files_1008/backbone_detector/single_simplified.onnx"),
                     Path("onnx_files_1008/backbone_detector/tuned.mxr")))
patch_sam3_video_model_detr_encoder(model,
    Path("onnx_files_1008/detector_modules/detr_encoder_simplified.onnx"))

trk = model.tracker_model
calls_this_frame = []
orig = trk.memory_attention.forward
def wrap(current_vision_features, memory, current_vision_position_embeddings=None,
         memory_posision_embeddings=None, num_object_pointer_tokens=0):
    calls_this_frame.append({
        "vision": tuple(current_vision_features.shape),
        "memory": tuple(memory.shape),
        "obj_ptr": int(num_object_pointer_tokens),
    })
    return orig(current_vision_features, memory, current_vision_position_embeddings,
                memory_posision_embeddings, num_object_pointer_tokens)
trk.memory_attention.forward = wrap

cap = cv2.VideoCapture("assets/demo.mp4")
frames = []
for _ in range(n_frames):
    ret, f = cap.read()
    if not ret: break
    frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
cap.release()

sess = proc.init_video_session(video=frames, inference_device=device, dtype=dtype)
proc.add_text_prompt(sess, text_prompt)

print(f"{'Frame':<8s} {'Calls':<6s} {'Vision':<20s} {'Memory':<20s} {'#ObjPtrs':<10s}")
print("-" * 70)
all_ptrs = []
all_mem_lens = []
for i in range(len(frames)):
    calls_this_frame.clear()
    with torch.inference_mode():
        _ = model(inference_session=sess, frame_idx=i)
    if not calls_this_frame:
        print(f"{i:<8d} 0     (skipped — first frame, no propagation)")
        continue
    for j, c in enumerate(calls_this_frame):
        n_calls = len(calls_this_frame) if j == 0 else ""
        print(f"{i:<8d} {str(n_calls):<6s} {str(c['vision']):<20s} {str(c['memory']):<20s} {c['obj_ptr']:<10d}")
        all_ptrs.append(c["obj_ptr"])
        all_mem_lens.append(c["memory"][0])

print()
print(f"Object pointer count distribution: {dict(Counter(all_ptrs))}")
print(f"  → max obj_ptr_tokens seen: {max(all_ptrs) if all_ptrs else 0}")
print(f"Memory total length distribution: {dict(Counter(all_mem_lens))}")
print(f"  → 7 × 5184 = 36288 (spatial-only baseline)")
print(f"  → suggested K (padding budget) ≥ {max(all_ptrs) if all_ptrs else 0}")
