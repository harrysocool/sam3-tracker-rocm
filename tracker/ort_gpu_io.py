"""Torch GPU tensor bindings for fixed-shape ONNX Runtime sessions.

ONNX Runtime's ROCm/MIGraphX builds expose the HIP device through the
``"cuda"`` I/O-binding device name. Binding Torch allocations directly avoids
the otherwise implicit GPU -> NumPy -> ORT -> NumPy -> GPU bridge.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Mapping, Sequence

import numpy as np
import torch


_thread_state = threading.local()


class GpuIoExecutionError(RuntimeError):
    """ORT failed after a GPU-bound run may already have been submitted."""


@contextmanager
def fence_ort_inputs():
    """Fence Torch-produced ORT inputs in the current worker thread."""
    previous = getattr(_thread_state, "fence_inputs", False)
    _thread_state.fence_inputs = True
    try:
        yield
    finally:
        _thread_state.fence_inputs = previous


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
    try:
        if getattr(_thread_state, "fence_inputs", False):
            # The casts above are Torch kernels queued on the branch stream.
            # ORT owns a different stream and receives only raw pointers, so
            # it cannot infer this producer dependency. Fence only the
            # caller's stream.
            torch.cuda.current_stream(device=device).synchronize()
        session.run_with_iobinding(binding)
        # MIGraphX can enqueue work on ORT's asynchronous compute stream. The
        # returned Torch tensor is consumed immediately on Torch's stream, so
        # wait before exposing its raw allocation to the caller.
        binding.synchronize_outputs()
    except Exception as exc:
        # A failed call may already have submitted work on an ORT-owned stream.
        # Drain the device before bound tensors leave scope, and distinguish
        # this from a pre-launch binding error that callers may safely fall
        # back from.
        try:
            torch.cuda.synchronize(device=device)
        except Exception:
            pass
        raise GpuIoExecutionError("ORT GPU-I/O execution failed") from exc
    return output


__all__ = ["GpuIoExecutionError", "fence_ort_inputs", "run_float32_gpu"]
