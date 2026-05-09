#!/usr/bin/env python3
"""
Pre-compile and cache all ORT MIGraphX sessions for a given onnx_dir.

Must be run BEFORE the tracker to populate the MIGraphX compile cache.
Run WITHOUT the backbone loaded (no GPU memory pressure) so FP16 compilations
succeed at 1008px without OOM.

Usage:
    python export/prewarm_ort_cache.py --onnx-dir onnx_files        # 504px
    python export/prewarm_ort_cache.py --onnx-dir onnx_files_1008   # 1008px
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
os.environ.setdefault("MIGRAPHX_GPU_HIP_FLAGS",
                      "-Wno-error -Wno-lifetime-safety-intra-tu-suggestions")
# Do NOT set MIGRAPHX_SKIP_BENCHMARKING here — kernel autotuning is critical:
# memory_attention: 758ms without autotuning vs 53ms with autotuning (14× difference).
# ORT sessions use their own provider options for SKIP_BENCHMARKING.
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"

import onnxruntime as ort


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx-dir", type=Path, default=Path("onnx_files"),
                   help="Directory containing exported ONNX files")
    p.add_argument("--cache-subdir", type=str, default="mxr_cache",
                   help="Cache directory name (inside onnx-dir)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Number of warmup inference runs per session")
    return p.parse_args()


def make_providers(fp16: bool, cache_dir: str):
    opts = {"migraphx_model_cache_dir": cache_dir}
    if fp16:
        opts["migraphx_fp16_enable"] = "1"
    return [("MIGraphXExecutionProvider", opts), ("CPUExecutionProvider", {})]


def warmup_session(sess: ort.InferenceSession, inputs: dict, n: int = 1):
    for i in range(n):
        sess.run(None, inputs)


def main():
    args = parse_args()
    onnx_dir = args.onnx_dir.resolve()
    cache_dir = str(onnx_dir / args.cache_subdir)
    Path(cache_dir).mkdir(exist_ok=True)

    # Infer resolution from directory name / feature map size
    # memory_attention_fixed_N7.onnx input shapes tell us HW
    ma_path = onnx_dir / "memory_attention_fixed_N7.onnx"
    if not ma_path.exists():
        raise FileNotFoundError(f"{ma_path} not found. Run export first.")

    # Peek at first input shape to determine HW
    _tmp = ort.InferenceSession(str(ma_path), providers=["CPUExecutionProvider"])
    HW = _tmp.get_inputs()[0].shape[0]   # current_vision_features: (HW, 1, 256)
    H = W = int(HW ** 0.5)
    del _tmp

    # Determine whether to use FP16 for memory_attention
    # At 1008px (HW=5184) we want FP16 — the whole point of this script is to
    # compile it here WITHOUT the backbone so there's no OOM.
    use_fp16_attn = True   # always compile FP16 (no backbone loaded here)

    print(f"Pre-warming ORT MIGraphX cache for {onnx_dir.name}")
    print(f"  Resolution: ~{H*14}px  HW={HW}  cache: {cache_dir}")
    print(f"  FP16 memory_attention: {'yes' if use_fp16_attn else 'no'}")
    print()

    # Dummy inputs
    CF = np.zeros((HW, 1, 256), dtype=np.float32)
    CP = np.zeros((HW, 1, 256), dtype=np.float32)
    MF = np.zeros((7 * HW, 1, 64), dtype=np.float32)
    MP = np.zeros((7 * HW, 1, 64), dtype=np.float32)
    MK = np.zeros((1, 1, H * 16, W * 16), dtype=np.float32)
    fpn0 = np.zeros((1, 256, H * 4, W * 4), dtype=np.float32)
    fpn1 = np.zeros((1, 256, H * 2, W * 2), dtype=np.float32)
    fpn2 = np.zeros((1, 256, H, W), dtype=np.float32)
    box  = np.array([[[[0, 0, H * 14 * 0.8, W * 14 * 0.8]]]], dtype=np.float32)
    pts  = np.zeros((1, 1, 1, 2), dtype=np.float32)
    lbl  = np.full((1, 1, 1), -1, dtype=np.int32)

    # ── 1. memory_attention: direct migraphx .mxr (FP16) ──────────────────────
    # ORT's MIGraphX EP silently falls back to CPU when backbone is in GPU memory.
    # Direct migraphx Python API avoids this: 148ms (ORT FP32) → 54ms (MIG FP16).
    ma_mxr_cache = Path(cache_dir) / "memory_attention_fp16.mxr"
    print(f"  memory_attention FP16 (direct migraphx .mxr) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        import sys
        sys.path.insert(0, "/opt/rocm-7.2.0/lib")
        import migraphx as _mxr
        if ma_mxr_cache.exists():
            _prog = _mxr.load(str(ma_mxr_cache))
            print(f"loaded existing  ({time.perf_counter()-t0:.1f}s)")
        else:
            _prog = _mxr.parse_onnx(str(ma_path))
            _mxr.quantize_fp16(_prog)
            _prog.compile(_mxr.get_target("gpu"), offload_copy=True)
            _mxr.save(_prog, str(ma_mxr_cache))
            print(f"compiled+saved  ({time.perf_counter()-t0:.1f}s)")
        # Verify with warmup
        _cf = _mxr.argument(CF); _cp = _mxr.argument(CP)
        _mf = _mxr.argument(MF); _mp = _mxr.argument(MP)
        for _ in range(args.warmup):
            _prog.run({"current_vision_features": _cf, "memory": _mf,
                       "current_vis_pos_embed": _cp, "memory_pos_embed": _mp})
    except Exception as e:
        print(f"FAILED: {e}")

    # ── 2. ORT MIGraphX sessions (dec_prop FP32, mem_enc FP16) ────────────────
    ort_sessions = [
        ("mask_decoder_propagate (ORT FP32)",
         str(onnx_dir / "mask_decoder_propagate.onnx"),
         make_providers(fp16=False, cache_dir=cache_dir),
         {"fpn_2_cond": fpn2, "fpn_0": fpn0, "fpn_1": fpn1}),

        ("memory_encoder (ORT FP16)",
         str(onnx_dir / "memory_encoder.onnx"),
         make_providers(fp16=True, cache_dir=cache_dir),
         {"vision_features": fpn2, "masks": MK}),

        ("mask_decoder_init (CPU — skipped)",
         None, None, None),
    ]

    total_t = time.perf_counter() - t0  # include ma compile time
    for label, path, providers, inputs in ort_sessions:
        if path is None:
            print(f"  {label}")
            continue
        print(f"  {label} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            sess = ort.InferenceSession(path, providers=providers)
            actual = sess.get_providers()[0]
            warmup_session(sess, inputs, args.warmup)
            elapsed = time.perf_counter() - t0
            total_t += elapsed
            print(f"done in {elapsed:.1f}s  [{actual[:10]}]")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nTotal compile time: {total_t:.0f}s")
    print(f"Cache saved to: {cache_dir}")
    print("\nTracker will now load from cache on startup (~2s per session).")


if __name__ == "__main__":
    main()
