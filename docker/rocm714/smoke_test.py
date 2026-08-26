"""GPU smoke test for the ROCm 7.14 SAM3 runtime image."""

import numpy as np
import onnxruntime as ort
import torch
from onnx import TensorProto, helper


def main() -> None:
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name(0)} arches={torch.cuda.get_arch_list()}")
    print(f"onnxruntime={ort.__version__} providers={ort.get_available_providers()}")

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

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["MIGraphXExecutionProvider"]
    )
    binding = session.io_binding()
    binding.bind_input("x", "cuda", 0, np.float32, tuple(x.shape), x.data_ptr())
    binding.bind_input("y", "cuda", 0, np.float32, tuple(y.shape), y.data_ptr())
    binding.bind_output("z", "cuda", 0, np.float32, tuple(z.shape), z.data_ptr())
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    torch.cuda.synchronize()
    if not torch.equal(z.cpu(), torch.full((2, 3), 3.0)):
        raise RuntimeError(f"unexpected Torch/ORT result: {z.cpu()}")
    print("SAM3 ROCm runtime smoke test: OK")


if __name__ == "__main__":
    main()
