"""Torch GPU tensor bindings for fixed-shape ONNX Runtime sessions.

ONNX Runtime's ROCm/MIGraphX builds expose the HIP device through the
``"cuda"`` I/O-binding device name. Binding Torch allocations directly avoids
the otherwise implicit GPU -> NumPy -> ORT -> NumPy -> GPU bridge.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def run_float32_gpu(
    session,
    inputs: Mapping[str, torch.Tensor],
    output_name: str,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Run one fixed-shape ORT output directly into a Torch GPU allocation."""
    if not inputs:
        raise ValueError("GPU I/O binding requires at least one input")

    first = next(iter(inputs.values()))
    if first.device.type != "cuda":
        raise ValueError(f"GPU I/O binding requires CUDA/HIP tensors, got {first.device}")
    device = first.device
    device_id = device.index if device.index is not None else torch.cuda.current_device()

    # The exported ONNX boundaries are FP32 even when MIGraphX quantizes the
    # graph internally. Keep these tensors alive until run_with_iobinding
    # returns because ORT holds only their raw addresses.
    bound_inputs = {
        name: tensor.detach().to(device=device, dtype=torch.float32).contiguous()
        for name, tensor in inputs.items()
    }
    output = torch.empty(tuple(output_shape), device=device, dtype=torch.float32)

    binding = session.io_binding()
    for name, tensor in bound_inputs.items():
        binding.bind_input(
            name,
            "cuda",
            device_id,
            np.float32,
            tuple(tensor.shape),
            tensor.data_ptr(),
        )
    binding.bind_output(
        output_name,
        "cuda",
        device_id,
        np.float32,
        tuple(output.shape),
        output.data_ptr(),
    )
    session.run_with_iobinding(binding)
    # MIGraphX can enqueue work on ORT's asynchronous compute stream. The
    # returned Torch tensor is consumed immediately on Torch's stream, so wait
    # for the bound output before exposing its raw allocation to the caller.
    # This is a no-op for providers that already complete synchronously.
    binding.synchronize_outputs()
    return output
