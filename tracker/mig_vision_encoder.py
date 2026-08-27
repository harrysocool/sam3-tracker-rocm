"""MIGraphX-backed vision_encoder shim for Sam3VideoModel.

Drop-in replacement for `Sam3VideoModel.detector_model.vision_encoder`. Routes
the forward pass through a precompiled MIGraphX `.mxr` (the detector backbone
exported with last_hidden_state — see `export/backbone/export_backbone_single.py
--backbone-source detector`). Returns a `Sam3VisionEncoderOutput` with all
fields the downstream Sam3VideoModel pipeline expects:

  - `fpn_hidden_states`: 4 FPN levels (consumed by detector path)
  - `fpn_position_encoding`: matching sine PE (cached by shape/device/dtype)
  - `last_hidden_state`: raw ViT tokens (consumed by `tracker_neck` to
                         compute tracker FPN — different weights from detector)

The position encoding module is reused from the original PyTorch
vision_encoder.neck (it has no learnable parameters; it's a sinusoidal
embedding parameterized by spatial size + dtype). Its fixed-shape outputs are
cached per encoder instance so detector calls stop evicting the tracker neck's
entries from Transformers' small function-level cache.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from .migraphx_runtime import MIGraphXBackbone


@dataclass
class _PendingVisionOutput:
    """GPU vision result plus the event and allocations that keep it valid."""

    output: Sam3VisionEncoderOutput
    event: torch.cuda.Event
    keepalive: object
    source_data_ptr: int

    def wait_on(self, stream: torch.cuda.Stream | None = None) -> Sam3VisionEncoderOutput:
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.wait_event(self.event)
        tensors = (
            self.output.last_hidden_state,
            *(self.output.fpn_hidden_states or ()),
            *(self.output.fpn_position_encoding or ()),
        )
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                tensor.record_stream(stream)
        return self.output

    def synchronize(self) -> None:
        self.event.synchronize()


def _to_torch_output(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Move a MIGraphX host output without a GPU-side FP32-to-FP16 cast.

    The compiled backbone exposes FP32 outputs. For the production FP16 path,
    casting on the host halves the transfer volume and is bit-identical to
    copying FP32 to the GPU and casting there. Keep the original conversion
    for other Torch dtypes, including bfloat16 which NumPy cannot represent.
    """
    if dtype == torch.float16:
        host = np.ascontiguousarray(array, dtype=np.float16)
        return torch.from_numpy(host).to(device=device)
    host = np.ascontiguousarray(array)
    return torch.from_numpy(host).to(device=device, dtype=dtype)


class MIGVisionEncoder(nn.Module):
    """Shim that mimics Sam3VisionModel.forward via a MIGraphX backbone.

    Args:
        mxr_backbone:    a `MIGraphXBackbone` whose underlying `.mxr` was
                         exported with `--backbone-source detector` and includes
                         the `last_hidden_state` output (5 outputs total).
        position_encoding: the sine PE module from the original PT
                         vision_encoder.neck (`detector.vision_encoder.neck.position_encoding`).
                         Has no learnable params; reused for cheap FPN PE.
    """

    def __init__(
        self,
        mxr_backbone: MIGraphXBackbone,
        position_encoding: nn.Module,
    ):
        super().__init__()
        self.mxr = mxr_backbone
        self.position_encoding = position_encoding
        # Transformers decorates Sam3SinePositionEmbedding.forward with one
        # function-level LRU of size four. The detector and tracker necks are
        # different instances with four shapes each, so their keys evict one
        # another every frame. Keep the detector's immutable mask=None values
        # per shim instance instead. Like the underlying MIGraphX program,
        # this cache assumes a single caller; cross-stream hand-off is explicit.
        self._position_encoding_cache = OrderedDict()

    def _apply(self, fn, recurse=True):
        # Cached tensors are intentionally not registered buffers. Drop them
        # when the module moves device or dtype so stale allocations cannot be
        # returned after nn.Module.to()/half()/float().
        self._position_encoding_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _get_position_encoding(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        key = (tuple(tensor.shape), device.type, device.index, tensor.dtype)
        cached = self._position_encoding_cache.pop(key, None)
        if cached is None:
            value = self.position_encoding(tensor.shape, device, tensor.dtype)
            event = None
            if device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
            cached = (value, event)
            if len(self._position_encoding_cache) >= 4:
                self._position_encoding_cache.popitem(last=False)
        self._position_encoding_cache[key] = cached

        value, event = cached
        if event is not None:
            stream = torch.cuda.current_stream(device)
            stream.wait_event(event)
            value.record_stream(stream)
        return value

    def _build_output(self, outs, device, dtype) -> Sam3VisionEncoderOutput:
        if len(outs) < 5 or outs[4] is None:
            raise RuntimeError(
                "MIGVisionEncoder requires a 5-output backbone (4 FPN + last_hidden_state). "
                "Re-export with: python export/backbone/export_backbone_single.py "
                "--backbone-source detector --output-name backbone_detector_lhs_fp32.onnx"
            )
        f0, f1, f2, f3, lhs = outs

        if self.mxr.gpu_io:
            fpn = [tensor.to(dtype=dtype) for tensor in (f0, f1, f2, f3)]
            last_hidden_state = lhs.to(dtype=dtype)
        else:
            fpn = [_to_torch_output(t, device, dtype) for t in (f0, f1, f2, f3)]
            last_hidden_state = _to_torch_output(lhs, device, dtype)
        pe = [self._get_position_encoding(tensor) for tensor in fpn]

        return Sam3VisionEncoderOutput(
            last_hidden_state=last_hidden_state,
            fpn_hidden_states=tuple(fpn),
            fpn_position_encoding=tuple(pe),
            hidden_states=None,
            attentions=None,
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Sam3VisionEncoderOutput:
        device = pixel_values.device
        dtype = pixel_values.dtype

        if self.mxr.gpu_io:
            outs = self.mxr.run_torch(pixel_values)
        else:
            # Host-I/O fallback for existing tuned.mxr artifacts.
            np_in = pixel_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
            outs = self.mxr(np_in)
        return self._build_output(outs, device, dtype)

    def _prefetch(
        self,
        pixel_values: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> _PendingVisionOutput:
        """Enqueue the GPU-resident encoder without a host-side wait."""
        if not self.mxr.gpu_io:
            raise RuntimeError("vision prefetch requires a GPU-I/O backbone")
        if getattr(self, "_prefetch_poisoned", False):
            raise RuntimeError(
                "vision prefetch is unusable after a failed asynchronous drain"
            )
        with torch.cuda.stream(stream):
            try:
                outputs, keepalive = self.mxr.enqueue_torch(pixel_values)
                result = self._build_output(
                    outputs,
                    pixel_values.device,
                    pixel_values.dtype,
                )
                event = torch.cuda.Event()
                event.record(stream)
            except BaseException:
                if "keepalive" in locals():
                    failed_keepalive = (
                        keepalive,
                        locals().get("result"),
                        locals().get("outputs"),
                    )
                    if not hasattr(self, "_failed_prefetch_keepalives"):
                        self._failed_prefetch_keepalives = []
                    self._failed_prefetch_keepalives.append(failed_keepalive)
                try:
                    stream.synchronize()
                except BaseException:
                    self._prefetch_poisoned = True
                else:
                    if "keepalive" in locals():
                        self._failed_prefetch_keepalives.remove(failed_keepalive)
                raise
        return _PendingVisionOutput(
            result,
            event,
            keepalive,
            pixel_values.data_ptr(),
        )


def patch_sam3_video_model_with_mig(model, mxr_backbone: MIGraphXBackbone) -> None:
    """In-place replace `model.detector_model.vision_encoder` with a MIG shim.

    `model` must be a `Sam3VideoModel` already loaded and moved to the desired
    device/dtype. After this, every call to `model(...)` /
    `_det_track_one_frame` runs the MIGraphX backbone for the vision encoder
    and the original PyTorch detector head, tracker_neck, and tracker_model.
    """
    pe = model.detector_model.vision_encoder.neck.position_encoding
    shim = MIGVisionEncoder(mxr_backbone, pe)
    # Move to model's device + dtype so children behave consistently
    target_dtype = next(model.detector_model.parameters()).dtype
    target_device = next(model.detector_model.parameters()).device
    shim = shim.to(device=target_device, dtype=target_dtype)
    model.detector_model.vision_encoder = shim
