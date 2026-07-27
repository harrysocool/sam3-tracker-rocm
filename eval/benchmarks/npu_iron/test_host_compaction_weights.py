#!/usr/bin/env python3
"""Verify padded FFN weights/biases are zero before skipping padded GELU."""

from pathlib import Path

import numpy as np


C = 1024
HIDDEN = 4736
HIDDEN_PAD = 5120
ROOT = Path("/home/amd/project/npu_iron/weights/cbb")


def main():
    worst = {"W1_tail": 0.0, "b1_tail": 0.0, "W2_tail": 0.0}
    for layer in range(32):
        w1 = np.fromfile(ROOT / f"L{layer}_W1.bin", np.float32).reshape(C, HIDDEN_PAD)
        b1 = np.fromfile(ROOT / f"L{layer}_b1.bin", np.float32)
        w2 = np.fromfile(ROOT / f"L{layer}_W2.bin", np.float32).reshape(HIDDEN_PAD, C)
        worst["W1_tail"] = max(worst["W1_tail"], float(np.max(np.abs(w1[:, HIDDEN:]))))
        worst["b1_tail"] = max(worst["b1_tail"], float(np.max(np.abs(b1[HIDDEN:]))))
        worst["W2_tail"] = max(worst["W2_tail"], float(np.max(np.abs(w2[HIDDEN:, :]))))

    if any(value != 0.0 for value in worst.values()):
        raise AssertionError(f"non-zero padded FFN data: {worst}")
    print(f"host-compaction padded weights PASS {worst}")


if __name__ == "__main__":
    main()
