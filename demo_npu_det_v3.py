#!/usr/bin/env python3
"""SAM3 单张图片：flexml NPU detector backbone + GPU decoder"""
import sys, os, time
import numpy as np
import cv2, torch
from PIL import Image
from pathlib import Path

os.environ["XILINX_XRT"] = "/opt/xilinx/xrt"
sys.path.insert(0, "/opt/xilinx/xrt/python")
sys.path.insert(0, "/home/amd/project/sam3-tracker-rocm")
from tracker.rocm_env import apply as _apply_rocm_env; _apply_rocm_env()

import onnxruntime as ort
FMLRT_LIB = "/home/amd/miniforge3/envs/rocm7p13-sam3/lib/python3.12/site-packages/flexmlrt/lib"
VOE_LIB   = "/home/amd/miniforge3/envs/rocm7p13-sam3/lib/python3.12/site-packages/voe/lib"
os.environ["LD_LIBRARY_PATH"] = f"{FMLRT_LIB}:{VOE_LIB}:{os.environ.get('LD_LIBRARY_PATH','')}"

IMGSZ = 504

# ── NPU detector backbone ────────────────────────────────────────────────────
print("Loading NPU detector backbone...")
det_sess = ort.InferenceSession(
    "onnx_files_504/backbone_detector/single_simplified.onnx",
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file": "/home/amd/miniforge3/envs/rocm7p13-sam3/lib/python3.12/site-packages/voe-4.0-linux_x86_64/vaip_config.json",
                       "cacheDir": "npu_artifacts/voe_cache_504", "cacheKey": "backbone_detector_504_v1"}],
)
print("  ready")

# ── vision_encoder shim ──────────────────────────────────────────────────────
import torch.nn as nn
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

class NPUVisionEncoder(nn.Module):
    def __init__(self, position_encoding):
        super().__init__()
        self.position_encoding = position_encoding

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        device, dtype = pixel_values.device, pixel_values.dtype
        np_in = pixel_values.detach().float().cpu().numpy()
        t0 = time.perf_counter()
        outs = det_sess.run(None, {"pixel_values": np_in})
        print(f"  NPU backbone: {(time.perf_counter()-t0)*1000:.0f} ms")
        fpn = [torch.from_numpy(np.ascontiguousarray(o)).to(device=device, dtype=dtype) for o in outs[:4]]
        lhs = torch.from_numpy(np.ascontiguousarray(outs[4])).to(device=device, dtype=dtype)
        pe  = [self.position_encoding(t.shape, t.device, t.dtype) for t in fpn]
        return Sam3VisionEncoderOutput(
            last_hidden_state=lhs, fpn_hidden_states=tuple(fpn),
            fpn_position_encoding=tuple(pe), hidden_states=None, attentions=None,
        )

# ── 加载模型（504px config）────────────────────────────────────────────────
from transformers import Sam3VideoModel, AutoProcessor, Sam3VideoConfig

print(f"Loading Sam3VideoModel ({IMGSZ}px config)...")
checkpoint = Path("model/sam3")
processor = AutoProcessor.from_pretrained(str(checkpoint))
config = Sam3VideoConfig.from_pretrained(str(checkpoint))
config.image_size = IMGSZ
config.low_res_mask_size = 4 * IMGSZ // 14
new_size = {"height": IMGSZ, "width": IMGSZ}
new_mask = {"height": 4*IMGSZ//14, "width": 4*IMGSZ//14}
for sub in (getattr(processor, "image_processor", None),
            getattr(processor, "video_processor", None)):
    if sub is not None:
        if hasattr(sub, "size"): sub.size = new_size
        if hasattr(sub, "mask_size"): sub.mask_size = new_mask

model = (Sam3VideoModel.from_pretrained(str(checkpoint), config=config)
         .to("cuda").to(torch.float16).eval())

# patch vision_encoder → NPU
orig_pe = model.detector_model.vision_encoder.neck.position_encoding
model.detector_model.vision_encoder = NPUVisionEncoder(orig_pe).to("cuda")
print(f"  vision_encoder → NPU  image_embedding_size={model.tracker_model.prompt_encoder.image_embedding_size}")

# ── 推理 ─────────────────────────────────────────────────────────────────────
IMG  = "assets/truck.jpg"
TEXT = "truck"

img_bgr = cv2.imread(IMG)
img_504 = cv2.resize(img_bgr, (IMGSZ, IMGSZ))
img_pil = Image.fromarray(cv2.cvtColor(img_504, cv2.COLOR_BGR2RGB))

print(f"\nDetecting '{TEXT}'...")
session = processor.init_video_session(video=[img_pil], inference_device="cuda", dtype=torch.float16)
processor.add_text_prompt(session, TEXT)

t0 = time.perf_counter()
with torch.inference_mode():
    out0 = model(inference_session=session, frame_idx=0)
torch.cuda.synchronize()
print(f"Total: {(time.perf_counter()-t0)*1000:.0f} ms  objects={len(out0.object_ids)}")
for obj_id in out0.object_ids:
    print(f"  #{obj_id} score={float(out0.obj_id_to_score.get(obj_id,0)):.2f}")

# ── 画 mask ──────────────────────────────────────────────────────────────────
vis = img_504.copy()
for obj_id in out0.object_ids:
    mask = out0.obj_id_to_mask[obj_id]
    if hasattr(mask, "cpu"): mask = mask.cpu().numpy()
    m = (mask > 0).astype(np.uint8)
    if m.ndim == 3: m = m[0]
    m = cv2.resize(m, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(vis); overlay[m > 0] = [0, 220, 0]
    vis = cv2.addWeighted(vis, 1.0, overlay, 0.5, 0)
    cv2.drawContours(vis, cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0,255,0), 2)
    score = float(out0.obj_id_to_score.get(obj_id,0))
    cv2.putText(vis, f"NPU #{obj_id} {score:.2f}", (10, 30+obj_id*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

os.makedirs("results", exist_ok=True)
out_path = "results/truck_npu_detector_20260709.jpg"
cv2.imwrite(out_path, vis)
print(f"\nSaved: {out_path}")
