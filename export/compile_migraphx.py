#!/usr/bin/env python3
"""Pre-compile backbone ONNX files to MIGraphX .mxr cache.

Run once; subsequent tracker runs load .mxr directly (seconds, not 680s).

Usage:
    python export/compile_migraphx.py --onnx-dir onnx_files --mxr-dir mxr_cache
"""
from __future__ import annotations
import sys, argparse, time
sys.path.insert(0, "/opt/rocm-7.2.0/lib")
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--onnx-dir', type=Path, default=WORKSPACE / 'onnx_files')
    p.add_argument('--mxr-dir',  type=Path, default=WORKSPACE / 'mxr_cache')
    p.add_argument('--fp16',     action='store_true', default=True)
    return p.parse_args()


def compile_one(onnx_path: Path, mxr_path: Path, fp16: bool = True):
    import migraphx
    print(f"  Parsing  {onnx_path.name} ...")
    t0 = time.perf_counter()
    prog = migraphx.parse_onnx(str(onnx_path))
    if fp16:
        migraphx.quantize_fp16(prog)
    print(f"  Compiling {onnx_path.name} ...")
    prog.compile(migraphx.get_target("gpu"), offload_copy=True)
    elapsed = time.perf_counter() - t0
    migraphx.save(prog, str(mxr_path))
    size_mb = mxr_path.stat().st_size / 1e6
    print(f"  Saved {mxr_path.name}  ({size_mb:.0f} MB, {elapsed:.0f}s)")
    return prog


def main():
    args = parse_args()
    args.mxr_dir.mkdir(parents=True, exist_ok=True)

    import migraphx
    print(f"MIGraphX version: {migraphx.__version__}")

    targets = [
        ("backbone_part1.onnx",          "backbone_part1.mxr"),
        ("backbone_block31.onnx",         "backbone_block31.mxr"),
        ("backbone_fpn.onnx",             "backbone_fpn.mxr"),
        ("memory_attention_fixed_N7.onnx","memory_attention_fixed_N7.mxr"),
    ]

    for onnx_name, mxr_name in targets:
        onnx_path = args.onnx_dir / onnx_name
        mxr_path  = args.mxr_dir  / mxr_name
        if not onnx_path.exists():
            print(f"\n[SKIP] {onnx_name} not found")
            continue
        print(f"\n[{onnx_name}]")
        try:
            compile_one(onnx_path, mxr_path, fp16=args.fp16)
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone. Load with: migraphx.load('mxr_cache/backbone_part1.mxr')")


if __name__ == '__main__':
    main()
