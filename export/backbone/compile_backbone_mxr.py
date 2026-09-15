#!/usr/bin/env python3
"""Compile simplified backbone ONNX to a MIGraphX .mxr cache with autotuning.

Reads backbone_<source>/single_simplified.onnx and produces either
backbone_<source>/tuned.mxr (host I/O) or tuned_gpuio.mxr (Torch GPU I/O).
Both are runtime caches that load in seconds instead of recompiling each
session; the GPU-I/O artifact is additive and never overwrites tuned.mxr.

Autotuning may take several minutes on the first compile. The resulting
.mxr is hardware-specific (gfx1151) and tied to the MIGraphX build that
produced it. The supported runtime is ROCm 7.14 / MIGraphX 2.17, selected
through docker/rocm714/run.sh; do not compile deployment caches with a host
runtime. See docker/rocm714/README.md for setup and runtime verification.

Configure SAM3_MODEL_DIR and SAM3_ONNX_DIR as described in that guide.
SAM3_ONNX_DIR must select a writable build directory containing the source
ONNX files, not an immutable baseline; it is mounted at /models/onnx_files_504.
For an additive GPU-I/O build, run from the repository root:

    SAM3_DOCKER_STRICT=0 ./docker/rocm714/run.sh \\
        python export/backbone/compile_backbone_mxr.py \\
        --imgsz 504 --backbone-source detector \\
        --onnx-dir /models/onnx_files_504 --gpu-io

Check the printed migraphx module path and the selected artifact directory
if verification fails. When changing runtime versions, rebuild in a new
artifact directory instead of overwriting existing baseline MXR files.
"""

from __future__ import annotations
import argparse
import os

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

from tracker.rocm_env import apply as _apply_rocm_env; _apply_rocm_env()
import time
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Resolution root (e.g. /models/onnx_files_504 in the container). Reads "
                        "<onnx-dir>/backbone_<source>/single_simplified.onnx, writes "
                        "<onnx-dir>/backbone_<source>/tuned.mxr or "
                        "tuned_gpuio.mxr with --gpu-io.")
    p.add_argument("--imgsz", type=int, default=504)
    p.add_argument("--backbone-source", choices=["tracker", "detector"],
                   default="tracker",
                   help="Which backbone subdir to operate on")
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Skip migraphx.quantize_fp16 (default: enabled).",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-compile output sanity check.",
    )
    p.add_argument(
        "--gpu-io",
        action="store_true",
        help="Compile with GPU-resident inputs/outputs and write tuned_gpuio.mxr.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sub_dir = args.onnx_dir / f"backbone_{args.backbone_source}"
    src = sub_dir / "single_simplified.onnx"
    dst = sub_dir / ("tuned_gpuio.mxr" if args.gpu_io else "tuned.mxr")

    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run export/backbone/simplify_backbone.py first."
        )

    # Make sure autotuning is enabled (env can disable it for fast iteration).
    os.environ.pop("MIGRAPHX_SKIP_BENCHMARKING", None)

    # Route attention subgraphs through rocMLIR-compiled kernels.
    # Validated on gfx1151: 18% faster (169ms vs 200ms/frame) with identical
    # mask quality (IoU=0.9995 vs baseline). Not the default on RDNA/gfx11;
    # must be set explicitly. Does not affect non-attention ops.
    os.environ.setdefault("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", "attention")

    # Defer the import: it touches the dynamic linker and prints whatever
    # warning the patched library emits.
    # MIGraphX Python binding lives in /opt/rocm-7.2.x/lib — add it if not
    # already on sys.path (e.g. when invoked as a subprocess without PYTHONPATH).
    import glob as _g
    _mxr_lib = (
        (os.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
        if os.environ.get("ROCM_PATH", "").rstrip("/") and os.path.isdir(os.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
        else next(
            (p for p in sorted(_g.glob("/opt/rocm-7.2.*/lib"), reverse=True)
             if os.path.isdir(p)), "/opt/rocm-7.2.0/lib"
        )
    )
    if _mxr_lib not in sys.path:
        sys.path.insert(0, _mxr_lib)
    import migraphx

    print(f"migraphx from: {migraphx.__file__}")
    print(f"Compiling {src} ...")
    print("  (autotuning enabled — first run may take several minutes)")

    t0 = time.perf_counter()
    prog = migraphx.parse_onnx(str(src))
    if not args.no_fp16:
        migraphx.quantize_fp16(prog)
    prog.compile(migraphx.get_target("gpu"), offload_copy=not args.gpu_io)
    elapsed = time.perf_counter() - t0
    print(f"  Compiled in {elapsed:.0f}s")

    migraphx.save(prog, str(dst))
    size_mb = dst.stat().st_size / 1e6
    print(f"  Saved: {dst}  ({size_mb:.0f} MB)")

    if args.skip_verify:
        return

    if args.gpu_io:
        print("\n[verify] Running GPU-resident backbone ...")
        params = {
            name: migraphx.to_gpu(migraphx.generate_argument(shape))
            for name, shape in prog.get_parameter_shapes().items()
        }
        outs = prog.run(params)
        migraphx.gpu_sync()
        for i, output in enumerate(outs):
            array = np.array(migraphx.from_gpu(output))
            if not np.isfinite(array).all():
                raise SystemExit(f"GPU output {i} contains non-finite values")
            print(f"  output_{i}: shape={array.shape} finite=True")
        print("  OK — GPU-resident inputs and outputs are valid")
        return

    print("\n[verify] Running compiled backbone, checking outputs are C-contiguous ...")
    inp = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    arg = migraphx.argument(inp)
    # Warmup so first-call kernel JIT doesn't pollute the check.
    for _ in range(3):
        prog.run({"pixel_values": arg})
    outs = prog.run({"pixel_values": arg})
    all_ok = True
    for i, o in enumerate(outs):
        a = np.array(o)
        c = a.flags.c_contiguous
        all_ok = all_ok and c
        print(f"  fpn_{i}: shape={a.shape} C_contiguous={c}")
    if not all_ok:
        raise SystemExit(
            f"Outputs are NOT C-contiguous; do not use the saved cache {dst}. "
            "Check the printed migraphx module path and use the supported "
            "ROCm 7.14 / MIGraphX 2.17 runtime through docker/rocm714/run.sh. "
            "Verify SAM3_ONNX_DIR and --onnx-dir select the intended artifact "
            "directory; do not mix host-runtime caches with container artifacts. "
            "See docker/rocm714/README.md for runtime verification and artifact "
            "setup. Rebuild in a new writable artifact directory, without "
            "overwriting baseline MXR files."
        )
    print("  OK — all outputs C-contiguous")


if __name__ == "__main__":
    main()
