#!/usr/bin/env python3
"""Validate the pruned 64-head SAM3 window FlashAttention schedule."""

import argparse
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt


HEADS = 64
SEQ = 576
DIM = 64
EXEC_ROWS = (576, 384, 384, 192)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        default="/home/amd/project/npu_iron/sam3_attn/valid_query_flash_grouped_wip_20260727",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(20260727)
    q = rng.standard_normal((HEADS, SEQ, DIM)).astype(bfloat16)
    k = rng.standard_normal((HEADS, SEQ, DIM)).astype(bfloat16)
    v = rng.standard_normal((HEADS, SEQ, DIM)).astype(bfloat16)

    xclbin_path = f"{args.artifact_root}/final.xclbin"
    insts_path = f"{args.artifact_root}/insts.bin"
    insts = np.fromfile(insts_path, dtype=np.uint8)

    direction = pyxrt.xclBOSyncDirection
    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(xclbin_path)
    uuid = device.register_xclbin(xclbin)
    context = pyxrt.hw_context(device, uuid)
    kernel = pyxrt.kernel(context, "MLIR_AIE")

    inst_bo = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    inst_bo.write(insts.tobytes(), 0)
    inst_bo.sync(direction.XCL_BO_SYNC_BO_TO_DEVICE)

    inputs = []
    for group, array in zip((3, 4, 5), (q, k, v)):
        bo = pyxrt.bo(device, array.nbytes, pyxrt.bo.host_only, kernel.group_id(group))
        bo.write(array.tobytes(), 0)
        bo.sync(direction.XCL_BO_SYNC_BO_TO_DEVICE)
        inputs.append(bo)

    output_nbytes = HEADS * SEQ * DIM * np.dtype(bfloat16).itemsize
    output_bo = pyxrt.bo(device, output_nbytes, pyxrt.bo.host_only, kernel.group_id(6))
    output_bo.write(bytes(output_nbytes), 0)
    output_bo.sync(direction.XCL_BO_SYNC_BO_TO_DEVICE)

    start = time.perf_counter()
    kernel(3, inst_bo, insts.nbytes, *inputs, output_bo).wait()
    elapsed_ms = (time.perf_counter() - start) * 1000
    output_bo.sync(direction.XCL_BO_SYNC_BO_FROM_DEVICE)
    actual = np.frombuffer(output_bo.read(output_nbytes, 0), dtype=bfloat16).reshape(
        HEADS, SEQ, DIM
    )

    rel_num = 0.0
    rel_den = 0.0
    max_abs = 0.0
    unused_max = 0.0
    scale = 1.0 / np.sqrt(DIM)
    for head in range(HEADS):
        rows = EXEC_ROWS[head // 16]
        qf = q[head, :rows].astype(np.float32)
        kf = k[head].astype(np.float32)
        vf = v[head].astype(np.float32)
        scores = qf @ kf.T * scale
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        expected = probs @ vf
        observed = actual[head, :rows].astype(np.float32)
        error = np.abs(observed - expected)
        rel_num += float(error.sum())
        rel_den += float(np.abs(expected).sum())
        max_abs = max(max_abs, float(error.max()))
        if rows < SEQ:
            unused_max = max(
                unused_max,
                float(np.max(np.abs(actual[head, rows:].astype(np.float32)))),
            )

    mean_rel_l1 = rel_num / max(rel_den, 1e-12)
    print(
        f"valid-query grouped elapsed_ms={elapsed_ms:.3f} "
        f"mean_rel_L1={mean_rel_l1:.6f} max_abs={max_abs:.6f} "
        f"unused_max={unused_max:.6f}"
    )
    if mean_rel_l1 > 0.06 or max_abs > 0.1 or unused_max != 0.0:
        print("VALID_QUERY_GROUPED=FAIL")
        return 1
    print("VALID_QUERY_GROUPED=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
