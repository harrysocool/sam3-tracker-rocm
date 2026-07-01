"""NPU backbone service: runs bh_npu_backbone C++ binary as subprocess.

Replaces Sam3VisionEncoder for keyframe backbone computation.
Uses separate subprocess so CUDA and NPU don't conflict in the same process.

Pipeline:
  pixel_values (GPU fp16)
    -> backbone.embeddings (PyTorch GPU, ~5ms)
    -> tokens.bin (file, float32)
    -> bh_npu_backbone subprocess (NPU backbone ~36.5W standalone, ~2.35s)
    -> features.bin (file, float32)
    -> neck (PyTorch GPU, ~5ms)
    -> Sam3VisionEncoderOutput
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

_BIN_INT8    = "/home/amd/project/npu_iron/bh_npu_backbone"      # INT8 MLIR-AIE, cos=0.932, 2.35s  (future improvement)
_BIN = _BIN_BF16 = "/home/amd/project/npu_iron/bh_npu_backbone_bf16"  # BF16 MLIR-AIE, cos=0.989, 2.29s (default)
_XRT_SETUP = "source /opt/xilinx/xrt/setup.sh 2>/dev/null"
_LD_PRELOAD = (
    "/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:"
    "/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0"
)


class NPUIRONVisionEncoder(nn.Module):
    """Drop-in replacement for Sam3VisionModel that runs the ViT on NPU.

    Keeps patch embedding (backbone.embeddings) and neck on GPU for speed;
    runs the 32 ViT blocks in a fresh subprocess to avoid CUDA-NPU conflicts.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_encoding: nn.Module,
        image_size: int = 504,
        patch_size: int = 14,
        npu_bin: str = _BIN,
        omp_threads: int = 8,
    ):
        super().__init__()
        self.backbone_embed = backbone.embeddings    # patch + pos embedding
        self.backbone_layer_norm = getattr(backbone, 'layer_norm', None)
        self.neck = neck
        self.position_encoding = position_encoding
        self.height = image_size // patch_size
        self.width = image_size // patch_size
        self.npu_bin = npu_bin
        self.omp_threads = omp_threads
        self._npu_available = Path(npu_bin).exists()
        self._current_proc = None
        self._server_proc = None
        self._server_env = self._make_env(omp_threads)
        self._timing: dict[str, float] = {}
        print(f"[NPUIRONVisionEncoder] npu_bin={npu_bin} available={self._npu_available} mode=server")

    def _make_env(self, omp_threads):
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '11.5.1'
        env['LD_PRELOAD'] = _LD_PRELOAD
        env['OMP_NUM_THREADS'] = str(omp_threads)
        env['PATH'] = '/opt/xilinx/xrt/bin:' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + env.get('LD_LIBRARY_PATH', '')
        return env

    def _start_server(self):
        """Launch persistent NPU server and wait for READY magic."""
        if self._server_proc is not None:
            try: self._server_proc.kill()
            except Exception: pass
        proc = subprocess.Popen(
            [self.npu_bin],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, env=self._server_env,
        )
        self._server_proc = proc
        raw = proc.stdout.read(4)  # wait for READY magic
        if len(raw) < 4 or int.from_bytes(raw, 'little') != 0x0000BF16:
            raise RuntimeError("NPU server failed to start")
        print("[NPUIRONVisionEncoder] server ready (weights resident)")

    def shutdown(self):
        if self._server_proc is not None:
            try:
                self._server_proc.stdin.close()
                self._server_proc.wait(timeout=3)
            except Exception:
                try: self._server_proc.kill()
                except Exception: pass
            self._server_proc = None

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Sam3VisionEncoderOutput:
        device = pixel_values.device
        dtype = pixel_values.dtype
        t0 = time.perf_counter()

        # 1. Patch embedding on GPU (fast, ~5ms)
        with torch.no_grad():
            tokens = self.backbone_embed(pixel_values)  # [1, 1296, 1024]

            # Sam3ViTModel.forward() reshapes to spatial then applies layer_norm BEFORE ViT layers.
            # The C++ binary expects this pre-normed spatial input (captured via hook on layers[0]).
            tokens_spatial = tokens.view(1, self.height, self.width, 1024)  # [1, 36, 36, 1024]
            if self.backbone_layer_norm is not None:
                tokens_spatial = self.backbone_layer_norm(tokens_spatial)
            tokens = tokens_spatial.view(1, self.height * self.width, 1024)  # [1, 1296, 1024]
        self._timing['embed_ms'] = (time.perf_counter() - t0) * 1000

        if not self._npu_available:
            raise RuntimeError(f"NPU binary not found: {self.npu_bin}")

        # 2. Start persistent server if needed (weights load only once)
        t1 = time.perf_counter()
        if self._server_proc is None or self._server_proc.poll() is not None:
            self._start_server()

        # 3. Send tokens via stdin (binary: magic + float32[S*C])
        MAGIC = b'\x16\xbf\x00\x00'  # 0x0000BF16 little-endian
        tokens_flat = tokens.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        try:
            self._server_proc.stdin.write(MAGIC + tokens_flat.tobytes())
            self._server_proc.stdin.flush()
        except BrokenPipeError:
            self._start_server()
            self._server_proc.stdin.write(MAGIC + tokens_flat.tobytes())
            self._server_proc.stdin.flush()

        # 4. Read features from stdout (magic + float32[S*C])
        n_expected = self.height * self.width * 1024
        _ = self._server_proc.stdout.read(4)    # skip echo magic
        raw = self._server_proc.stdout.read(n_expected * 4)
        if len(raw) < n_expected * 4:
            raise RuntimeError(f"NPU server short read: {len(raw)}/{n_expected*4}")
        features_np = np.frombuffer(raw, dtype=np.float32).copy().reshape(1, self.height * self.width, 1024)

        self._timing['npu_ms'] = (time.perf_counter() - t1) * 1000

        # 5. Apply neck (fast PyTorch on GPU)
        # C++ binary already processed through 32 ViT blocks (pre-normed input).
        # No additional layer_norm needed here.
        lhs = torch.from_numpy(features_np).to(device=device, dtype=dtype)  # [1, 1296, 1024]

        # Reshape for neck: [1, 1296, 1024] -> [1, 1024, H, W]
        hidden_spatial = lhs.view(1, self.height, self.width, 1024).permute(0, 3, 1, 2)

        with torch.no_grad():
            fpn_hidden_states, fpn_position_encoding = self.neck(hidden_spatial)

        self._timing['neck_ms'] = (time.perf_counter() - t0) * 1000
        self._timing['total_ms'] = (time.perf_counter() - t0) * 1000

        return Sam3VisionEncoderOutput(
            last_hidden_state=lhs,
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
            hidden_states=None,
            attentions=None,
        )

    @property
    def timing(self) -> dict:
        return self._timing


def patch_sam3_with_npu_backbone(model, npu_bin: str = _BIN, omp_threads: int = 8):
    """Replace Sam3VideoModel's vision_encoder with NPUIRONVisionEncoder.

    Works with both plain Sam3VisionModel and MIGVisionEncoder (mig=True).
    When mig=True was used, MIG patches for DETR/memory_attention remain active —
    only the vision backbone is replaced with NPU.

    Args:
        model: Sam3VideoModel instance (with or without prior mig=True patching)
        npu_bin: Path to compiled bh_npu_backbone binary
        omp_threads: OMP_NUM_THREADS for C++ binary

    Returns:
        The NPUIRONVisionEncoder instance (for timing access)
    """
    enc = model.detector_model.vision_encoder

    # If MIG already replaced vision_encoder, extract backbone/neck from MIG's stored reference.
    # patch_sam3_video_model_with_mig wraps the original Sam3VisionModel inside MIGVisionEncoder
    # and stores it at enc._orig_sam3_vision_model (if we patched it to do so).
    # Simpler: just require that patch_sam3_with_npu_backbone is called BEFORE mig patches,
    # OR access backbone/neck from the detector config and reload weights.
    # Cleanest: store backbone/neck before MIG replaces enc.
    from tracker.mig_vision_encoder import MIGVisionEncoder as _MIGVE
    if isinstance(enc, _MIGVE) and hasattr(enc, '_orig_backbone'):
        backbone = enc._orig_backbone
        neck = enc._orig_neck
        print("[NPU backbone] using saved backbone/neck from MIGVisionEncoder")
    else:
        # Plain Sam3VisionModel (mig=False or first call)
        backbone = enc.backbone
        neck = enc.neck
    # Get position_encoding from neck (it has a position_encoding module)
    pos_enc = neck.position_encoding

    npu_enc = NPUIRONVisionEncoder(
        backbone=backbone,
        neck=neck,
        position_encoding=pos_enc,
        image_size=model.config.detector_config.vision_config.image_size,
        patch_size=model.config.detector_config.vision_config.backbone_config.patch_size,
        npu_bin=npu_bin,
        omp_threads=omp_threads,
    )
    model.detector_model.vision_encoder = npu_enc
    print(f"[NPU backbone] vision_encoder replaced with NPUIRONVisionEncoder")
    return npu_enc
