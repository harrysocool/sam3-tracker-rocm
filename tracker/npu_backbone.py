"""NPU backbone for SAM3 ViT — runs 99.9% of the ViT on the AMD XDNA2 NPU
via flexmlrt (Ryzen AI SDK 1.7.1).

Compilation is done once offline (ryzenai-npu conda env):
    from tracker.npu_backbone import compile_backbone_npu
    compile_backbone_npu("onnx_files_504/backbone_tracker/single_simplified.onnx",
                         "npu_artifacts/backbone_504")

Runtime inference runs inside the rocm7p13-sam3 conda env (flexmlrt installed).

Interface is identical to MIGraphXBackbone.__call__:
    img_np: (1, 3, H, W) float32  →  (fpn_0, fpn_1, fpn_2, None)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# XRT must be initialised before importing flexmlrt
_XRT_SETUP_DONE = False


def _ensure_xrt() -> None:
    global _XRT_SETUP_DONE
    if _XRT_SETUP_DONE:
        return
    xrt = os.environ.get("XILINX_XRT", "/opt/xilinx/xrt")
    os.environ.setdefault("XILINX_XRT", xrt)
    # Extend LD_LIBRARY_PATH so flexmlrt's libflexmlrt.so finds libxrt_core.so
    voe_lib = ""
    try:
        import site
        sp = site.getsitepackages()[0]
        voe_lib = f"{sp}/voe/lib:{sp}/voe/lib64"
    except Exception:
        pass
    xrt_lib = f"{xrt}/lib"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in [xrt_lib, voe_lib, ld] if p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    _XRT_SETUP_DONE = True


class NPUBackbone:
    """SAM3 ViT backbone accelerated on AMD XDNA2 NPU via flexmlrt.

    Args:
        model_dir: Path to the compiled .flexml artifact directory
                   (output of flexml.compile / compile_backbone_npu).

    Outputs of __call__: (fpn_0, fpn_1, fpn_2, None) — same as MIGraphXBackbone.
    The 4th slot is always None (backbone_tracker/single_simplified.onnx has
    3 FPN outputs at 504px).
    """

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"NPU backbone artifacts not found: {model_dir}\n"
                f"Run compile_backbone_npu() first (ryzenai-npu conda env)."
            )

        _ensure_xrt()

        print(f"  NPU backbone: loading {model_dir.name} ...")
        t0 = time.perf_counter()

        import flexmlrt
        self._model = flexmlrt.load(str(model_dir))
        self._model.load()

        print(f"  NPU backbone: ready in {time.perf_counter()-t0:.1f}s")

    def warmup(self, n: int = 3) -> None:
        dummy = np.random.rand(1, 3, 504, 504).astype(np.float32)
        for _ in range(n):
            self._model(dummy)

    def __call__(self, img_np: np.ndarray):
        """img_np: (1, 3, H, W) float32  →  (fpn_0, fpn_1, fpn_2, None).

        Returns float32 numpy arrays. The 4th slot is None for compatibility
        with callers expecting a 4-tuple (MIGraphXBackbone API).
        """
        img_cont = np.ascontiguousarray(img_np, dtype=np.float32)
        raw = self._model(img_cont)

        # flexmlrt returns numpy arrays or torch tensors depending on input type.
        # Normalise to list of float32 numpy arrays.
        out_arrs = []
        for o in raw:
            if hasattr(o, "numpy"):          # torch tensor
                arr = o.numpy().astype(np.float32)
            else:
                arr = np.asarray(o, dtype=np.float32)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            out_arrs.append(arr)

        # Right-pad with None to at least 4 elements (MIGraphXBackbone compat).
        while len(out_arrs) < 4:
            out_arrs.append(None)
        return tuple(out_arrs)


def compile_backbone_npu(
    onnx_path: str | Path,
    output_dir: str | Path,
    device: str = "stx",
    imgsz: int = 504,
) -> None:
    """Compile SAM3 backbone ONNX for NPU. Run in ryzenai-npu conda env.

    Requires:
        - flexml installed (pip install flexml-*.whl)
        - aiecompiler in PATH (flexml_extras/bin/)
        - XILINX_XRT set

    Args:
        onnx_path:  Path to single_simplified.onnx
        output_dir: Where to save the .flexml artifacts
        device:     'stx' for Strix Halo (default), 'phx' for Phoenix
        imgsz:      Input image size (default 504)
    """
    import site
    sp = site.getsitepackages()[0]

    # Add aiecompiler to PATH if not already present
    aiec_bin = os.path.join(sp, "flexml", "flexml_extras", "bin")
    if aiec_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = aiec_bin + ":" + os.environ.get("PATH", "")

    _ensure_xrt()

    import torch
    import flexml

    onnx_path  = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.rand(1, 3, imgsz, imgsz, dtype=torch.float32)

    print(f"Compiling {onnx_path.name} for NPU (device={device}) ...")
    print(f"Output → {output_dir}")
    t0 = time.time()

    model = flexml.compile(
        model=str(onnx_path),
        inputs=[dummy],
        model_type="onnx",
        output_type="aie-exe",
        output_dir=str(output_dir),
        device=device,
        enable_f32_to_bf16_conversion=True,
    )

    elapsed = time.time() - t0
    print(f"Compilation done in {elapsed:.1f}s")

    if model is not None:
        print("Testing inference on NPU ...")
        import numpy as np
        inp = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        outs = model(inp)
        for i, o in enumerate(outs):
            print(f"  output[{i}]: shape={getattr(o,'shape','?')} dtype={getattr(o,'dtype','?')}")
        print("SUCCESS")
    else:
        print(f"Compiled (binary only, no runtime). Artifacts at: {output_dir}")
