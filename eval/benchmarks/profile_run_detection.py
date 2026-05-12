#!/usr/bin/env python3
"""Profile internals of run_detection to see what's left after detr_encoder MIG."""
import time, statistics
from pathlib import Path
import torch
from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, Sam3VideoModel
from tracker.tracker import MIGraphXBackbone
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

# Hook detector sub-modules
det = model.detector_model
timings = defaultdict(list)
def make_wrap(name, fn):
    def w(*a, **kw):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        timings[name].append(time.perf_counter() - t)
        return out
    return w

for name in ["text_encoder", "text_projection", "detr_encoder", "detr_decoder", "mask_decoder", "dot_product_scoring"]:
    if hasattr(det, name):
        mod = getattr(det, name)
        mod.forward = make_wrap(name, mod.forward)

img = Image.open("assets/demo.jpg").convert("RGB")
ts = []
for i in range(7):
    sess = proc.init_video_session(video=[img], inference_device=device, dtype=dtype)
    proc.add_text_prompt(sess, "truck")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(inference_session=sess, frame_idx=0)
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)

n = 7; warmup = 2
print(f"\nMean total init: {statistics.mean(ts[warmup:])*1000:.0f} ms")
print(f"\n{'Module':<30s} {'Calls/iter':>10s} {'Mean ms':>10s}")
print("-" * 55)
for name in ["text_encoder", "text_projection", "detr_encoder", "detr_decoder", "mask_decoder", "dot_product_scoring"]:
    samples = timings.get(name, [])
    if not samples: continue
    calls = len(samples) // n
    per_iter = [sum(samples[i*calls:(i+1)*calls]) for i in range(n)][warmup:]
    print(f"{name:<30s} {calls:>10d} {statistics.mean(per_iter)*1000:>10.1f}")
