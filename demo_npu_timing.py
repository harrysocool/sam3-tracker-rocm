#!/usr/bin/env python3
"""单张图片 NPU 推理耗时分解"""
import sys, os, time
import numpy as np, cv2, torch
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
RUNS  = 3

# ── NPU session ───────────────────────────────────────────────────────────────
det_sess = ort.InferenceSession(
    "onnx_files_504/backbone_detector/single_simplified.onnx",
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file": "/home/amd/miniforge3/envs/rocm7p13-sam3/lib/python3.12/site-packages/voe-4.0-linux_x86_64/vaip_config.json",
                       "cacheDir": "npu_artifacts/voe_cache_504", "cacheKey": "backbone_detector_504_v1"}],
)

# ── vision encoder shim ────────────────────────────────────────────────────────
import torch.nn as nn
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

npu_times = []
class NPUVisionEncoder(nn.Module):
    def __init__(self, pe): super().__init__(); self.position_encoding = pe
    def forward(self, pixel_values, **kwargs):
        device, dtype = pixel_values.device, pixel_values.dtype
        np_in = pixel_values.detach().float().cpu().numpy()
        t0 = time.perf_counter()
        outs = det_sess.run(None, {"pixel_values": np_in})
        npu_times.append((time.perf_counter()-t0)*1000)
        fpn = [torch.from_numpy(np.ascontiguousarray(o)).to(device=device, dtype=dtype) for o in outs[:4]]
        lhs = torch.from_numpy(np.ascontiguousarray(outs[4])).to(device=device, dtype=dtype)
        pe  = [self.position_encoding(t.shape, t.device, t.dtype) for t in fpn]
        return Sam3VisionEncoderOutput(last_hidden_state=lhs, fpn_hidden_states=tuple(fpn),
                                       fpn_position_encoding=tuple(pe), hidden_states=None, attentions=None)

# ── 加载模型 ──────────────────────────────────────────────────────────────────
from transformers import Sam3VideoModel, AutoProcessor, Sam3VideoConfig

checkpoint = Path("model/sam3")
processor  = AutoProcessor.from_pretrained(str(checkpoint))
config     = Sam3VideoConfig.from_pretrained(str(checkpoint))
config.image_size = IMGSZ
config.low_res_mask_size = 4 * IMGSZ // 14
new_size = {"height": IMGSZ, "width": IMGSZ}
new_mask = {"height": 4*IMGSZ//14, "width": 4*IMGSZ//14}
for sub in (getattr(processor,"image_processor",None), getattr(processor,"video_processor",None)):
    if sub is not None:
        if hasattr(sub,"size"): sub.size = new_size
        if hasattr(sub,"mask_size"): sub.mask_size = new_mask

model = Sam3VideoModel.from_pretrained(str(checkpoint), config=config).to("cuda").to(torch.float16).eval()
orig_pe = model.detector_model.vision_encoder.neck.position_encoding
model.detector_model.vision_encoder = NPUVisionEncoder(orig_pe).to("cuda")

# ── 图片 ─────────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("assets/truck.jpg")
img_504 = cv2.resize(img_bgr, (IMGSZ, IMGSZ))
img_pil = Image.fromarray(cv2.cvtColor(img_504, cv2.COLOR_BGR2RGB))

# 建 session（包含 CLIP 等初始化）
t_sess = time.perf_counter()
session = processor.init_video_session(video=[img_pil], inference_device="cuda", dtype=torch.float16)
processor.add_text_prompt(session, "truck")
t_sess = (time.perf_counter()-t_sess)*1000

# warmup
with torch.inference_mode():
    model(inference_session=session, frame_idx=0)
session = processor.init_video_session(video=[img_pil], inference_device="cuda", dtype=torch.float16)
processor.add_text_prompt(session, "truck")
npu_times.clear()

# 计时 RUNS 次
total_times, decoder_times = [], []
for _ in range(RUNS):
    session = processor.init_video_session(video=[img_pil], inference_device="cuda", dtype=torch.float16)
    processor.add_text_prompt(session, "truck")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(inference_session=session, frame_idx=0)
    torch.cuda.synchronize()
    total_times.append((time.perf_counter()-t0)*1000)

npu_avg   = sum(npu_times[-RUNS:]) / RUNS
total_avg = sum(total_times) / RUNS
decoder_avg = total_avg - npu_avg

print(f"\n{'='*45}")
print(f"  SAM3 单张图片推理耗时分解（{RUNS}次平均）")
print(f"{'='*45}")
print(f"  NPU backbone (flexml)   {npu_avg:7.0f} ms  ({npu_avg/total_avg*100:.0f}%)")
print(f"  GPU decoder / CLIP      {decoder_avg:7.0f} ms  ({decoder_avg/total_avg*100:.0f}%)")
print(f"  {'─'*37}")
print(f"  Total (per frame)       {total_avg:7.0f} ms")
print(f"{'='*45}")
print(f"  Session init (one-time) {t_sess:7.0f} ms")
print(f"  Objects detected: {len(out.object_ids)}, score={float(out.obj_id_to_score.get(0,0)):.2f}")
