#!/usr/bin/env python3
"""Prove projection compaction preserves padded-window attention semantics."""

import numpy as np


GRID = 36
WIN = 24
TOKENS = GRID * GRID
C = 8
HEADS = 2
D = C // HEADS


def partition(x):
    out = np.zeros((4, WIN * WIN, x.shape[-1]), dtype=x.dtype)
    for gi in range(GRID):
        for gj in range(GRID):
            win = (gi // WIN) * 2 + gj // WIN
            pos = (gi % WIN) * WIN + gj % WIN
            out[win, pos] = x[gi * GRID + gj]
    return out


def unpartition(x):
    out = np.empty((TOKENS, x.shape[-1]), dtype=x.dtype)
    for gi in range(GRID):
        for gj in range(GRID):
            win = (gi // WIN) * 2 + gj // WIN
            pos = (gi % WIN) * WIN + gj % WIN
            out[gi * GRID + gj] = x[win, pos]
    return out


def attention(qkv_no_bias, bias, cos, sin):
    qkv = qkv_no_bias + bias
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(4, WIN * WIN, HEADS, D).transpose(0, 2, 1, 3)
    k = k.reshape(4, WIN * WIN, HEADS, D).transpose(0, 2, 1, 3)
    v = v.reshape(4, WIN * WIN, HEADS, D).transpose(0, 2, 1, 3)

    for pos in range(WIN * WIN):
        for lane in range(0, D, 2):
            for tensor in (q, k):
                a = tensor[:, :, pos, lane].copy()
                b = tensor[:, :, pos, lane + 1].copy()
                tensor[:, :, pos, lane] = a * cos[pos, lane] - b * sin[pos, lane]
                tensor[:, :, pos, lane + 1] = (
                    b * cos[pos, lane + 1] + a * sin[pos, lane + 1]
                )

    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(D)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.matmul(probs, v)
    return out.transpose(0, 2, 1, 3).reshape(4, WIN * WIN, C)


def main():
    rng = np.random.default_rng(20260726)
    x = rng.normal(size=(TOKENS, C)).astype(np.float32)
    wqkv = rng.normal(scale=0.2, size=(C, 3 * C)).astype(np.float32)
    bqkv = rng.normal(scale=0.1, size=(3 * C,)).astype(np.float32)
    wo = rng.normal(scale=0.2, size=(C, C)).astype(np.float32)
    bo = rng.normal(scale=0.1, size=(C,)).astype(np.float32)
    angles = rng.normal(scale=0.2, size=(WIN * WIN, D)).astype(np.float32)
    cos, sin = np.cos(angles), np.sin(angles)

    # Original: partition zero-padded tokens, then project all 2304 rows.
    x_window = partition(x)
    qkv_original = np.matmul(x_window, wqkv)
    attn_original = attention(qkv_original, bqkv, cos, sin)
    original = unpartition(np.matmul(attn_original, wo) + bo)

    # Compact: project 1296 valid rows, materialize zero pre-bias QKV only at
    # padded window positions, gather valid attention rows, then O-project.
    qkv_valid = np.matmul(x, wqkv)
    qkv_compact = partition(qkv_valid)
    attn_compact = attention(qkv_compact, bqkv, cos, sin)
    compact = np.matmul(unpartition(attn_compact), wo) + bo

    max_abs = float(np.max(np.abs(original - compact)))
    np.testing.assert_allclose(compact, original, rtol=2e-5, atol=2e-5)
    print(f"projection-compaction semantics PASS max_abs={max_abs:.8g}")


if __name__ == "__main__":
    main()
