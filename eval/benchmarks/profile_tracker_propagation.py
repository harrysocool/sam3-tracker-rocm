#!/usr/bin/env python3
"""Profile internals of run_tracker_propagation to choose MIG-ization targets."""
import time, statistics
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel
from tracker.migraphx_runtime import MIGraphXBackbone
from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder

device, dtype = torch.device("cuda"), torch.float16
proc = AutoProcessor.from_pretrained("/home/amd/project/sam3/model/sam3")
model = Sam3VideoModel.from_pretrained("/home/amd/project/sam3/model/sam3").to(device).to(dtype).eval()
mxr_bb = MIGraphXBackbone(Path("onnx_files_1008/backbone_detector/single_simplified.onnx"),
                           Path("onnx_files_1008/backbone_detector/tuned.mxr"))
mxr_bb.warmup(2)
patch_sam3_video_model_with_mig(model, mxr_bb)
patch_sam3_video_model_detr_encoder(model, Path("onnx_files_1008/detector_modules/detr_encoder_simplified.onnx"))

trk = model.tracker_model
timings = defaultdict(list)

def make_wrap(name, fn):
    def w(*a, **kw):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        timings[name].append(time.perf_counter() - t)
        return out
    return w

# Hook tracker sub-modules + key methods
for mod_name in ["memory_attention", "mask_decoder", "memory_encoder"]:
    if hasattr(trk, mod_name):
        m = getattr(trk, mod_name)
        m.forward = make_wrap(mod_name, m.forward)

# Hook a few important methods
for meth_name in ["_prepare_memory_conditioned_features", "_run_single_frame_inference",
                  "_encode_new_memory", "_gather_memory_frame_outputs",
                  "_build_memory_attention_inputs", "_get_object_pointers",
                  "_process_object_pointers"]:
    if hasattr(trk, meth_name):
        setattr(trk, meth_name, make_wrap(meth_name, getattr(trk, meth_name)))

# Load video frames
cap = cv2.VideoCapture("assets/blackswan.mp4")
frames = []
for _ in range(11):
    ret, f = cap.read()
    if not ret: break
    frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
cap.release()
print(f"Loaded {len(frames)} frames")

# Init session, run frame 0 (init), then propagate frames 1..N
sess = proc.init_video_session(video=frames, inference_device=device, dtype=dtype)
proc.add_text_prompt(sess, "swan")
with torch.inference_mode():
    _ = model(inference_session=sess, frame_idx=0)

# Reset timings (drop init frame)
timings.clear()

# Profile prop frames
prop_totals = []
for i in range(1, len(frames)):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model(inference_session=sess, frame_idx=i)
    torch.cuda.synchronize()
    prop_totals.append(time.perf_counter() - t0)

n = len(prop_totals)
print(f"\nMean prop frame: {statistics.mean(prop_totals)*1000:.0f} ms (over {n} frames)")
print(f"\n{'Hook':<45s} {'Calls/iter':>10s} {'Mean ms':>10s}")
print("-" * 70)
for name in ["_run_single_frame_inference", "_prepare_memory_conditioned_features",
             "_gather_memory_frame_outputs", "_build_memory_attention_inputs",
             "_get_object_pointers", "_process_object_pointers",
             "memory_attention", "mask_decoder", "memory_encoder", "_encode_new_memory"]:
    samples = timings.get(name, [])
    if not samples: continue
    if len(samples) % n == 0:
        calls = len(samples) // n
        per_iter = [sum(samples[i*calls:(i+1)*calls]) for i in range(n)]
    else:
        calls = len(samples) / n
        per_iter = samples[-n:]
    cstr = f"{calls:.1f}" if isinstance(calls, float) else str(calls)
    print(f"{name:<45s} {cstr:>10s} {statistics.mean(per_iter)*1000:>10.1f}")
