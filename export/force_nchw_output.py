"""
Add Reshape-flatten → Reshape-back ops to backbone ONNX outputs.

MIGraphX GPU internally uses NHWC (channel-last) format. When offloading to CPU
via offload_copy=True, it outputs NHWC strides instead of C-contiguous NCHW.
The CPU then needs an expensive transpose (~89ms for 27.6MB at 1008px).

Fix: add Reshape(1D) → Reshape(original) at each output. This is a mathematical
no-op, but forces MIGraphX to finalize the layout conversion on GPU before
handing data to the CPU offload path.

Usage:
    python export/force_nchw_output.py \
        --input  onnx_files_1008/backbone_single_simplified.onnx \
        --output onnx_files_1008/backbone_single_nchw.onnx
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def add_contiguous_output(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    For each graph output, insert:
        original_name → Reshape(flat) → Reshape(original_shape) → new_name

    This forces MIGraphX to produce a standard C-contiguous output buffer.
    """
    new_nodes = []
    new_initializers = []

    # Map: old output name → new output name
    renames: dict[str, str] = {}

    for out in model.graph.output:
        orig_name = out.name
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if not shape or any(s <= 0 for s in shape):
            continue  # skip dynamic-shape outputs

        n_elements = 1
        for s in shape:
            n_elements *= s

        flat_name   = orig_name + "_flat__"
        cont_name   = orig_name + "_nchw__"
        shape_1d    = orig_name + "_shape_1d__"
        shape_nd    = orig_name + "_shape_nd__"

        # Initialiser: [1, N] flat shape
        init_1d = numpy_helper.from_array(
            np.array([1, n_elements], dtype=np.int64), name=shape_1d)
        new_initializers.append(init_1d)

        # Initialiser: original shape
        init_nd = numpy_helper.from_array(
            np.array(shape, dtype=np.int64), name=shape_nd)
        new_initializers.append(init_nd)

        # Reshape to [1, N]
        new_nodes.append(helper.make_node(
            "Reshape", inputs=[orig_name, shape_1d], outputs=[flat_name]))

        # Reshape back to original shape → contiguous NCHW in memory
        new_nodes.append(helper.make_node(
            "Reshape", inputs=[flat_name, shape_nd], outputs=[cont_name]))

        renames[orig_name] = cont_name

    # Append nodes and initializers
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)

    # Update graph outputs to point to new names
    for out in model.graph.output:
        if out.name in renames:
            out.name = renames[out.name]

    return model


def main():
    args = parse_args()
    model = onnx.load(str(args.input))
    print(f"Loaded:  {args.input}  ({len(model.graph.node)} nodes)")

    outputs_before = [o.name for o in model.graph.output]
    model = add_contiguous_output(model)
    outputs_after  = [o.name for o in model.graph.output]
    print(f"Outputs: {outputs_before} → {outputs_after}")
    print(f"Nodes after patch: {len(model.graph.node)}")

    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f"Saved:   {args.output}")


if __name__ == "__main__":
    main()
