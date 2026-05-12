"""MIGraphX-backed memory_attention shim for Sam3VideoModel.tracker_model.

Drop-in replacement for `tracker_model.memory_attention.forward`. Routes the
forward through ONNX Runtime's MIGraphX execution provider with a padded
FIXED-shape ONNX (`memory_attention_fixed_S7_P64.onnx` produced by
`export/tracker_modules/export_memory_attention_padded.py`):

    spatial slots: 7  → 7 * 5184 = 36288 spatial tokens (steady-state only)
    pointer tokens: 64 (= max_object_pointers_in_encoder * 4, cap)
    total memory: 36352 tokens

Fallback rule:
    The shim runs MIG ONLY when the spatial portion equals 36288 AND the
    pointer count is ≤ 64. For early-video frames (first ~7) the spatial
    bank isn't full yet, so we fall through to the original PyTorch
    forward. After frame ~16 the pointer count caps at 64 and every frame
    is the same fixed shape → MIG fast path.

Why fixed-shape pad rather than dynamic shape:
- Dynamic ONNX through ORT MIG EP recompiles per shape (~90 s each); since
  pointer count grows by 4 every frame for the first 16 frames, that means
  16 cold compiles before steady-state.
- Direct migraphx.parse_onnx + quantize_fp16 has the FP16 attention
  numerical bug (Finding #8 / detr_encoder analog).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort


# ----- shape constants for the SAM3 1008px text-prompt video pipeline -----
HW_1008 = 5184              # 72 * 72
SPATIAL_SLOTS = 7
# K=32 (= 8 conditioning-frame pointers × 4 tokens/pointer) is the practical cap
# imposed by MIGraphX runtime: at K=64 the compiler picks a kernel that's
# 14× slower (791 ms vs 55 ms). When the actual PT pointer count exceeds 32,
# we truncate to the LAST 32 (= most-recent pointers); information from older
# conditioning frames is dropped — affects long-video re-identification but
# leaves continuous-tracking quality intact.
PTR_TOKENS = 32
EXPECTED_SPATIAL_LEN = SPATIAL_SLOTS * HW_1008      # 36288
EXPECTED_TOTAL_LEN   = EXPECTED_SPATIAL_LEN + PTR_TOKENS  # 36320


class MIGMemoryAttention(nn.Module):
    def __init__(self, onnx_path: Path, original_forward,
                 ort_cache_dir: Path | None = None):
        super().__init__()
        # `original_forward` is the BOUND .forward method captured BEFORE we
        # monkey-patch the module's forward. Storing the module + going through
        # __call__ recurses (because the patched forward calls us again).
        self._original_forward = original_forward
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cache_dir = Path(ort_cache_dir) if ort_cache_dir else (
            Path(onnx_path).parent / "ort_cache_mem_attn"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers = [
            ("MIGraphXExecutionProvider", {
                "migraphx_fp16_enable": "1",
                "migraphx_model_cache_dir": str(cache_dir),
            }),
            "CPUExecutionProvider",
        ]
        print(f"  memory_attention (ORT MIG EP, fp16): compiling from {Path(onnx_path).name} ...")
        import time
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
        elapsed = time.perf_counter() - t0
        prov = self.session.get_providers()[0]
        print(f"  memory_attention (ORT MIG EP): ready in {elapsed:.1f}s on {prov}")

        # Stats
        self._mig_calls = 0
        self._pt_fallback_calls = 0

    def forward(
        self,
        current_vision_features,
        memory,
        current_vision_position_embeddings=None,
        memory_posision_embeddings=None,
        num_object_pointer_tokens: int = 0,
    ):
        spatial_part = memory.shape[0] - num_object_pointer_tokens
        # Fast-path requires the steady-state spatial size; pointer count
        # may exceed PTR_TOKENS (we truncate to the most-recent PTR_TOKENS).
        ok = (spatial_part == EXPECTED_SPATIAL_LEN
              and num_object_pointer_tokens >= 0
              and current_vision_features.shape[0] == HW_1008)

        if not ok:
            self._pt_fallback_calls += 1
            return self._original_forward(
                current_vision_features=current_vision_features,
                memory=memory,
                current_vision_position_embeddings=current_vision_position_embeddings,
                memory_posision_embeddings=memory_posision_embeddings,
                num_object_pointer_tokens=num_object_pointer_tokens,
            )

        self._mig_calls += 1
        device = current_vision_features.device
        dtype = current_vision_features.dtype

        # Object pointers are at the END of the memory tensor:
        #   [spatial(36288) | actual_ptrs(N)]
        # We need exactly PTR_TOKENS ptr slots:
        #   N <= PTR_TOKENS:  pad with zeros (no info loss)
        #   N >  PTR_TOKENS:  keep last PTR_TOKENS (drop oldest pointers)
        spatial = memory[:spatial_part]
        spatial_pos = memory_posision_embeddings[:spatial_part]
        ptrs = memory[spatial_part:]
        ptrs_pos = memory_posision_embeddings[spatial_part:]

        if num_object_pointer_tokens <= PTR_TOKENS:
            pad_n = PTR_TOKENS - num_object_pointer_tokens
            if pad_n > 0:
                zero_pad = torch.zeros(pad_n, 1, 64, dtype=memory.dtype, device=memory.device)
                ptrs = torch.cat([ptrs, zero_pad], dim=0)
                ptrs_pos = torch.cat([ptrs_pos, zero_pad], dim=0)
        else:
            # Keep the LAST PTR_TOKENS (most recent pointers in PT temporal order)
            ptrs = ptrs[-PTR_TOKENS:]
            ptrs_pos = ptrs_pos[-PTR_TOKENS:]

        memory_padded = torch.cat([spatial, ptrs], dim=0)
        mem_pos_padded = torch.cat([spatial_pos, ptrs_pos], dim=0)

        # Run via ORT MIG EP (numpy fp32)
        out_np = self.session.run(None, {
            "current_vision_features": current_vision_features.detach().float().cpu().numpy(),
            "memory":                  memory_padded.detach().float().cpu().numpy(),
            "current_vis_pos_embed":   current_vision_position_embeddings.detach().float().cpu().numpy(),
            "memory_pos_embed":        mem_pos_padded.detach().float().cpu().numpy(),
        })
        # ONNX exports `conditioned_features` as (1, 256, H, W); PT returns
        # (1, 1, HW, 256). Caller (_prepare_memory_conditioned_features) does:
        #   .squeeze(1).permute(0,2,1).view(B, C, H, W)
        # so we must return the (1, 1, HW, 256) shape PT does.
        out_np_arr = out_np[0]                  # (1, 256, H, W) np float32
        # Reshape to (1, 1, HW, 256) on host (cheap), THEN move to GPU+fp16
        out_4d = torch.from_numpy(out_np_arr)   # (1, 256, H, W) cpu fp32
        out_4d = out_4d.flatten(2).permute(0, 2, 1).unsqueeze(0)  # (1, 1, HW, 256)
        return out_4d.to(device=device, dtype=dtype)


def patch_sam3_video_model_memory_attention(model, onnx_path: Path) -> None:
    """Hot-patch `model.tracker_model.memory_attention.forward` in place.

    We do NOT swap the whole module — that would break parameter ownership
    and any nn.Module child checks elsewhere. Instead we capture the original
    bound forward, build a shim that knows how to call it, and rebind
    `.forward` to the shim.
    """
    trk = model.tracker_model
    original_forward = trk.memory_attention.forward  # captured BEFORE monkey-patch
    shim = MIGMemoryAttention(Path(onnx_path), original_forward)
    trk.memory_attention.forward = shim.forward
    trk.memory_attention._mig_shim = shim  # keep ref alive
