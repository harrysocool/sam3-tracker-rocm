#!/usr/bin/env python3
"""Diagnose MIG-GPU vs PT vs ONNX-CPU divergence on Sam3DetrEncoder output.

If MIG presents large diffs vs PT, recompile with --no-fp16 may help. If even
ONNX-CPU diverges, something's wrong upstream of MIG.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, Sam3VideoModel
from tracker.tracker import MIGraphXBackbone, MIGraphXSession
from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
from tracker.mig_detr_encoder import precompute_cross_attn_mask

device = torch.device("cuda")
dtype = torch.float16

print("Loading model + MIG backbone ...")
processor = AutoProcessor.from_pretrained("/home/amd/project/sam3/model/sam3")
model = Sam3VideoModel.from_pretrained("/home/amd/project/sam3/model/sam3").to(device).to(dtype).eval()

mxr_bb = MIGraphXBackbone(
    onnx_path=Path("onnx_files_1008/backbone_detector/single_simplified.onnx"),
    cache_path=Path("onnx_files_1008/backbone_detector/tuned.mxr"),
)
mxr_bb.warmup(2)
patch_sam3_video_model_with_mig(model, mxr_bb)

img = Image.open("assets/demo.jpg").convert("RGB")
session = processor.init_video_session(video=[img], inference_device=device, dtype=dtype)
text_inputs = processor.tokenizer("truck", return_tensors="pt", padding="max_length", max_length=32).to(device)

# Get vision_embeds + text_features (from real pipeline)
detector = model.detector_model
pixel_values = session.get_frame(0).unsqueeze(0)
with torch.inference_mode():
    vis = detector.get_vision_features(pixel_values=pixel_values)
    text_out = detector.get_text_features(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask, return_dict=True)
    text_features = text_out.pooler_output
text_mask = text_inputs.attention_mask.bool()
fpn = vis.fpn_hidden_states[:-1]
pos = vis.fpn_position_encoding[:-1]
vision_feature = fpn[-1]
vision_pos     = pos[-1]
print(f"vision_feature: {tuple(vision_feature.shape)} {vision_feature.dtype}")
print(f"text_features:  {tuple(text_features.shape)} {text_features.dtype}")

# 1. PT reference (call original detr_encoder)
print("\n=== PT reference detr_encoder ===")
with torch.inference_mode():
    # detr_encoder is unpatched in this diag script, so use it directly
    pt_ref = detector.detr_encoder(
        vision_features=[vision_feature],
        text_features=text_features,
        vision_pos_embeds=[vision_pos],
        text_mask=text_mask,
    )
pt_lhs = pt_ref.last_hidden_state
print(f"  PT last_hidden_state: shape={tuple(pt_lhs.shape)} mean={pt_lhs.float().mean():+.4f} std={pt_lhs.float().std():.4f}")

# 2. ONNX-CPU
print("\n=== ONNX-CPU detr_encoder ===")
import onnxruntime as ort
sess = ort.InferenceSession(
    "onnx_files_1008/detector_modules/detr_encoder_simplified.onnx",
    providers=["CPUExecutionProvider"]
)
cross_mask = precompute_cross_attn_mask(text_mask, dtype=text_features.dtype)
np_in = {
    "vision_feature":  vision_feature.detach().float().cpu().numpy(),
    "vision_pos":      vision_pos.detach().float().cpu().numpy(),
    "text_features":   text_features.detach().float().cpu().numpy(),
    "cross_attn_mask": cross_mask.detach().float().cpu().numpy(),
}
ocpu_lhs = sess.run(None, np_in)[0]
print(f"  ONNX-CPU last_hidden_state: mean={ocpu_lhs.mean():+.4f} std={ocpu_lhs.std():.4f}")
diff = (pt_lhs.float().cpu().numpy() - ocpu_lhs)
print(f"  vs PT: max diff={np.abs(diff).max():.5f}  mean={np.abs(diff).mean():.6f}")

# 3. MIG-GPU
print("\n=== MIG-GPU detr_encoder ===")
mig_sess = MIGraphXSession(
    onnx_path=Path("onnx_files_1008/detector_modules/detr_encoder_simplified.onnx"),
    cache_path=Path("onnx_files_1008/detector_modules/detr_encoder.mxr"),
    fp16=True,
    label="diag detr_encoder",
)
mig_out = mig_sess.run(None, np_in)[0]
print(f"  MIG-GPU last_hidden_state: mean={mig_out.mean():+.4f} std={mig_out.std():.4f}")
diff_mig_pt = (pt_lhs.float().cpu().numpy() - mig_out)
diff_mig_onnxcpu = (ocpu_lhs - mig_out)
print(f"  vs PT:        max diff={np.abs(diff_mig_pt).max():.5f}  mean={np.abs(diff_mig_pt).mean():.6f}")
print(f"  vs ONNX-CPU:  max diff={np.abs(diff_mig_onnxcpu).max():.5f}  mean={np.abs(diff_mig_onnxcpu).mean():.6f}")
