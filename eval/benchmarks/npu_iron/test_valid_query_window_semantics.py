#!/usr/bin/env python3
"""Prove compact window queries preserve valid-token attention outputs."""

import numpy as np


WIN = 24
GRID = 36
HEADS = 2
DIM = 8
VALID_SHAPES = ((24, 24), (24, 12), (12, 24), (12, 12))
EXEC_ROWS = (576, 384, 384, 192)


def softmax_attention(q, k, v):
    scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(DIM)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.matmul(probs, v)


def main():
    rng = np.random.default_rng(20260727)
    q = rng.normal(size=(4, HEADS, WIN * WIN, DIM)).astype(np.float32)
    k = rng.normal(size=(4, HEADS, WIN * WIN, DIM)).astype(np.float32)
    v = rng.normal(size=(4, HEADS, WIN * WIN, DIM)).astype(np.float32)
    original = softmax_attention(q, k, v)

    compact_q = np.zeros_like(q)
    valid_positions = []
    for window, (height, width) in enumerate(VALID_SHAPES):
        positions = [i * WIN + j for i in range(height) for j in range(width)]
        valid_positions.append(positions)
        compact_q[window, :, : len(positions)] = q[window][:, positions, :]

    compact = np.zeros_like(original)
    for window, rows in enumerate(EXEC_ROWS):
        compact[window, :, :rows] = softmax_attention(
            compact_q[window, :, :rows], k[window], v[window]
        )

    gathered_original = []
    gathered_compact = []
    for window, positions in enumerate(valid_positions):
        gathered_original.append(original[window][:, positions, :])
        gathered_compact.append(compact[window, :, : len(positions)])
    lhs = np.concatenate(gathered_original, axis=1)
    rhs = np.concatenate(gathered_compact, axis=1)
    max_abs = float(np.max(np.abs(lhs - rhs)))
    np.testing.assert_allclose(rhs, lhs, rtol=2e-5, atol=2e-5)
    print(f"valid-query window semantics PASS max_abs={max_abs:.8g}")


if __name__ == "__main__":
    main()
