"""GPU smoke test for the ROCm 10.1 SAM3 runtime image.

Run from the repository root with ``PYTHONPATH=/workspace`` so ``tracker`` is
importable.
"""

import os
import tempfile

import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from tracker.ort_plugin_ep import (
    create_mig_session,
    describe_ort_migraphx_stack,
    stack_errors,
)


def main() -> None:
    stack = describe_ort_migraphx_stack()
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name(0)} arches={torch.cuda.get_arch_list()}")
    print(f"stack={stack}")
    errors = stack_errors(stack)
    if errors:
        raise RuntimeError("unsupported stack: " + "; ".join(errors))
    if torch.__version__ != "2.14.0+rocm10.1.0":
        raise RuntimeError(f"unexpected Torch: {torch.__version__}")

    # The default SDPA backend needs amd-torch-device-gfx115x; it fails with
    # hipErrorInvalidImage when that wheel is missing.
    q = torch.randn((1, 4, 16, 32), device="cuda", dtype=torch.float16)
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise RuntimeError("SDPA produced non-finite output")

    x = torch.ones((2, 3), device="cuda", dtype=torch.float32)
    y = torch.full((2, 3), 2.0, device="cuda", dtype=torch.float32)
    z = torch.empty((2, 3), device="cuda", dtype=torch.float32)

    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z_info = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["z"])],
        "torch_ort_smoke",
        [x_info, y_info],
        [z_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/add.onnx"
        with open(path, "wb") as handle:
            handle.write(model.SerializeToString())
        # The plugin EP does not create its cache directory (the runtime
        # callers mkdir it). Raises if any node would fall back to CPU.
        os.makedirs(f"{tmp}/cache")
        session = create_mig_session(path, ort.SessionOptions(), f"{tmp}/cache", fp16=False)
        binding = session.io_binding()
        values = {name: ort.OrtValue.from_dlpack(t) for name, t in (("x", x), ("y", y))}
        for name, value in values.items():
            binding.bind_ortvalue_input(name, value)
        out_value = ort.OrtValue.from_dlpack(z)
        binding.bind_ortvalue_output("z", out_value)
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        torch.cuda.synchronize()
    if not torch.equal(z.cpu(), torch.full((2, 3), 3.0)):
        raise RuntimeError(f"unexpected Torch/ORT result: {z.cpu()}")
    print("SAM3 ROCm runtime smoke test: OK")


if __name__ == "__main__":
    main()
