"""MIGraphX runtime helpers — low-level building blocks used by both
``SAM3OnnxTracker`` (single-object SAM3 tracker) and the
text-prompt ``Sam3VideoModel`` MIG shims.

Contents:
  - ``_load_migraphx_module``   patched migraphx 2.16 Python binding loader
  - ``MIGraphXSession``         ORT-like wrapper around a compiled mxr program
  - ``MIGraphXBackbone``        compiled SAM3 vision encoder (3/4/5-output FPN)
  - ``preprocess_image``        BGR uint8 → float32 NCHW normalized tensor
  - ``retarget_resolution``     RoPE re-init for non-default 1008px input
  - ``MemoryBank``              FIFO of spatial memory entries for tracker

No tracker class lives here — see ``sam3_onnx_tracker.py``.
"""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# torch is imported lazily in the PyTorch backbone path only.
# MIGraphX backbone mode has zero torch dependencies.

os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")
# Suppress clang -Werror on lifetimebound warnings in MIGraphX JIT kernel compilation.
# Without this flag, newer comgr/clang versions fail with:
#   "parameter ... should be marked [[clang::lifetimebound]] [-Werror,-Wlifetime-safety-intra-tu-suggestions]"
from tracker.rocm_env import apply as _apply_rocm_env; _apply_rocm_env()


# ---------------------------------------------------------------------------
# MIGraphX direct-API helpers (backbone + memory_attention)
# ---------------------------------------------------------------------------

_MXR_BUILD_LIB = "/home/amd/project/tools/AMDMIGraphX/build_docker/lib"


def _load_migraphx_module():
    """Import the patched MIGraphX 2.16.0 Python binding."""
    import sys
    import glob as _g, os as _o
    _mxr_py_dir = (
        (_o.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
        if _o.environ.get("ROCM_PATH", "").rstrip("/") and _o.path.isdir(_o.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
        else next(
            (p for p in sorted(_g.glob("/opt/rocm-7.2.*/lib"), reverse=True)
             if _o.path.isdir(p)),
            "/opt/rocm-7.2.0/lib"
        )
    )
    if _mxr_py_dir not in sys.path:
        sys.path.insert(0, _mxr_py_dir)
    if _MXR_BUILD_LIB not in sys.path:
        sys.path.append(_MXR_BUILD_LIB)
    import migraphx
    return migraphx


class MIGraphXSession:
    """
    Thin wrapper around a migraphx compiled program that mimics the ORT session
    interface (.run(None, inputs_dict) → [output_array]).

    Used for memory_attention at 1008px where ORT's MIGraphX EP fails to keep
    the inference on GPU when the backbone is also in GPU memory.
    """

    def __init__(self, onnx_path: str | Path, cache_path: str | Path,
                 fp16: bool = True, label: str = "MIGraphX session") -> None:
        _mxr = _load_migraphx_module()
        self._mxr = _mxr

        cache_path = Path(cache_path)
        onnx_path  = Path(onnx_path)

        if cache_path.exists():
            print(f"  {label}: loading {cache_path.name} ...")
            t0 = time.perf_counter()
            self._prog = _mxr.load(str(cache_path))
            print(f"  {label}: ready in {time.perf_counter()-t0:.1f}s")
        elif onnx_path.exists():
            print(f"  {label}: compiling {onnx_path.name} ...")
            t0 = time.perf_counter()
            prog = _mxr.parse_onnx(str(onnx_path))
            if fp16:
                _mxr.quantize_fp16(prog)
            prog.compile(_mxr.get_target("gpu"), offload_copy=True)
            print(f"  {label}: compiled in {time.perf_counter()-t0:.1f}s")
            _mxr.save(prog, str(cache_path))
            self._prog = prog
        else:
            raise FileNotFoundError(f"Neither {cache_path} nor {onnx_path} found")

        # Discover ordered output names for run() return list
        self._in_names  = self._prog.get_parameter_names()

    def run(self, _output_names, inputs: dict) -> list:
        """inputs: {name: np.ndarray}  →  [output_array, ...]"""
        # np.ascontiguousarray: MIGraphX requires exact stride matching to compiled strides.
        # Backbone FPN outputs are already contiguous (converted in MIGraphXBackbone.__call__).
        # Other inputs (e.g. cur_feat from fpn2.transpose()) may be non-contiguous.
        args = {k: self._mxr.argument(np.ascontiguousarray(v)) for k, v in inputs.items()}
        outputs = self._prog.run(args)
        return [np.array(o) for o in outputs]

    def get_providers(self) -> list:
        return ["MIGraphXExecutionProvider"]


class MIGraphXBackbone:
    """
    SAM3 vision encoder compiled with patched MIGraphX 2.16.0.

    Achieves ~88ms at 504px vs PyTorch ROCm's 139ms (1.6× speedup).
    Uses backbone_<source>/single_simplified.onnx with kernel autotuning baked into
    the .mxr cache file; first-time compilation takes ~3 minutes.

    Outputs: (fpn_0, fpn_1, fpn_2, fpn_3_or_None) as float32 numpy arrays.
    fpn_3 (smallest, scale=0.5) is needed by the SAM3 detector path
    (text-prompt). Returns None for the 4th slot when running an older
    3-output backbone .mxr — the box-prompt tracker doesn't read it either way.
    """

    def __init__(self, onnx_path: str | Path, cache_path: str | Path) -> None:
        import sys
        # Prefer the ROCm lib dir where the Python binding lives; fall back to build dir.
        import glob as _g2, os as _o2
        _mxr_py_dir = (
            (_o2.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
            if _o2.environ.get("ROCM_PATH", "").rstrip("/") and _o2.path.isdir(_o2.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
            else next(
                (p for p in sorted(_g2.glob("/opt/rocm-7.2.*/lib"), reverse=True)
                 if _o2.path.isdir(p)), "/opt/rocm-7.2.0/lib"
            )
        )
        if _mxr_py_dir not in sys.path:
            sys.path.insert(0, _mxr_py_dir)
        if _MXR_BUILD_LIB not in sys.path:
            sys.path.append(_MXR_BUILD_LIB)

        import migraphx as _mxr
        self._mxr = _mxr

        cache_path = Path(cache_path)
        onnx_path  = Path(onnx_path)

        if cache_path.exists():
            print(f"  MIGraphX backbone: loading {cache_path.name} ...")
            t0 = time.perf_counter()
            self._prog = _mxr.load(str(cache_path))
            print(f"  MIGraphX backbone: ready in {time.perf_counter()-t0:.1f}s")
        elif onnx_path.exists():
            print(f"  MIGraphX backbone: compiling {onnx_path.name} with autotuning (~3 min) ...")
            t0 = time.perf_counter()
            # Temporarily lift SKIP_BENCHMARKING so autotuning selects optimal kernels
            _old = os.environ.pop("MIGRAPHX_SKIP_BENCHMARKING", None)
            prog = _mxr.parse_onnx(str(onnx_path))
            _mxr.quantize_fp16(prog)
            prog.compile(_mxr.get_target("gpu"), offload_copy=True)
            if _old is not None:
                os.environ["MIGRAPHX_SKIP_BENCHMARKING"] = _old
            print(f"  MIGraphX backbone: compiled in {time.perf_counter()-t0:.1f}s")
            _mxr.save(prog, str(cache_path))
            print(f"  MIGraphX backbone: cache saved → {cache_path}")
            self._prog = prog
        else:
            raise FileNotFoundError(
                f"MIGraphX backbone needs {cache_path} (pre-compiled) "
                f"or {onnx_path} (to compile from scratch); neither found."
            )

    def warmup(self, n: int = 3) -> None:
        # Use random normal (not zeros) and keep array reference alive across all runs.
        # MIGraphX may access input pointer asynchronously; a pre-allocated persistent
        # array prevents dangling-pointer reads if a temporary were GC'd mid-run.
        _shape = list(self._prog.get_parameter_shapes()["pixel_values"].lens())
        _data  = np.random.randn(*_shape).astype(np.float32)
        _arg   = self._mxr.argument(_data)
        for _ in range(n):
            self._prog.run({"pixel_values": _arg})

    def __call__(self, img_np: np.ndarray):
        """img_np: (1, 3, H, W) float32  →  tuple of all backbone outputs.

        Length depends on which `.mxr` was loaded:
          3 outputs: legacy box-prompt tracker (fpn_0, fpn_1, fpn_2)
          4 outputs: detector-compatible backbone with fpn_3 (scale=0.5)
          5 outputs: detector backbone + last_hidden_state (Sam3VideoModel pipeline)

        For backward compat with callers that hardcode 4-tuple unpacking, the
        return is right-padded with None to at least length 4. Callers that
        need last_hidden_state should index `outputs[4]` directly and check
        for None.
        """
        # Keep explicit references to prevent GC of data/argument before GPU finishes.
        img_cont = np.ascontiguousarray(img_np)
        arg      = self._mxr.argument(img_cont)
        outputs  = self._prog.run({"pixel_values": arg})
        # With patched MIGraphX (fix/offload-copy-contiguous-output branch),
        # backbone outputs are already C-contiguous (contiguous_kernel runs on GPU).
        # Without the patch, outputs are NHWC non-contiguous and np.ascontiguousarray
        # is needed (10ms at 504px, 94ms at 1008px CPU overhead).
        out_arrs = [np.array(o) for o in outputs]
        if out_arrs and not out_arrs[0].flags.c_contiguous:
            # Fallback: MIGraphX patch not applied, convert on CPU
            out_arrs = [np.ascontiguousarray(a) for a in out_arrs]
        # Right-pad with None so callers expecting at least 4 entries (older
        # MIGraphXBackbone API) keep working.
        while len(out_arrs) < 4:
            out_arrs.append(None)
        return tuple(out_arrs)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR uint8 → float32 NCHW normalized, resized to imgsz×imgsz."""
    img = cv2.resize(img_bgr[:, :, ::-1], (imgsz, imgsz))
    return ((img.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]


# ---------------------------------------------------------------------------
# RoPE re-targeting (needed when running at non-default 1008px resolution)
# ---------------------------------------------------------------------------

def retarget_resolution(model, new_imgsz: int) -> None:
    """Re-initialize all RoPE buffers for a different input resolution."""
    import torch
    from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoVisionRotaryEmbedding,
    )

    new_H = new_imgsz // 14
    model.config.image_size = new_imgsz
    model.config.memory_attention_rope_feat_sizes = [new_H, new_H]
    model.image_size = new_imgsz
    # prompt_encoder lives on Sam3TrackerVideoModel directly, but on
    # Sam3VideoModel it lives under .tracker_model. Be defensive.
    pe = getattr(model, "prompt_encoder", None) or          getattr(getattr(model, "tracker_model", None), "prompt_encoder", None)
    if pe is not None:
        pe.image_embedding_size = (new_H, new_H)
        pe.mask_input_size = (4 * new_H, 4 * new_H)
        pe.input_image_size = new_imgsz

    for _, mod in model.named_modules():
        dev   = getattr(mod, "rope_embeddings_cos", torch.tensor(0)).device
        dtype = getattr(mod, "rope_embeddings_cos", torch.tensor(0.0)).dtype
        if isinstance(mod, Sam3ViTRotaryEmbedding) and mod.end_x > new_H:
            mod.end_x = mod.end_y = new_H
            freqs = 1.0 / (mod.rope_theta ** (
                torch.arange(0, mod.dim, 4)[:mod.dim // 4].float() / mod.dim))
            flat = torch.arange(new_H * new_H, dtype=torch.long)
            xp = (flat % new_H).float() * mod.scale
            yp = torch.div(flat, new_H, rounding_mode="floor").float() * mod.scale
            inv = torch.cat(
                [torch.outer(xp, freqs), torch.outer(yp, freqs)], dim=-1
            ).repeat_interleave(2, dim=-1)
            mod.register_buffer("rope_embeddings_cos", inv.cos().to(dev, dtype), persistent=False)
            mod.register_buffer("rope_embeddings_sin", inv.sin().to(dev, dtype), persistent=False)
        elif isinstance(mod, Sam3TrackerVideoVisionRotaryEmbedding):
            mod.end_x = mod.end_y = new_H
            inv = mod.create_inv_freq()
            mod.register_buffer("rope_embeddings_cos", inv.cos().to(dev, dtype), persistent=False)
            mod.register_buffer("rope_embeddings_sin", inv.sin().to(dev, dtype), persistent=False)


# ---------------------------------------------------------------------------
# Memory bank
# ---------------------------------------------------------------------------

class MemoryBank:
    """FIFO of spatial memory entries with temporal positional encodings."""

    def __init__(self, temporal_pe: np.ndarray, max_slots: int = 7):
        self.temporal_pe = temporal_pe   # (max_slots, 1, 1, 64)
        self.max_slots   = max_slots
        self._entries: deque = deque(maxlen=max_slots)

    def push(self, maskmem_features: np.ndarray, maskmem_pos_enc: np.ndarray) -> None:
        self._entries.appendleft((maskmem_features, maskmem_pos_enc))

    def build_attention_inputs(self, fixed_slots: int = 0):
        if not self._entries:
            return None, None
        mem_list, pos_list = [], []
        for i, (mf, mp) in enumerate(self._entries):
            pos_list.append(mp + self.temporal_pe[i])
            mem_list.append(mf)
        if fixed_slots > 0 and len(mem_list) < fixed_slots:
            last_mf, last_pos = mem_list[0], pos_list[0]
            while len(mem_list) < fixed_slots:
                mem_list.append(last_mf)
                pos_list.append(last_pos)
        return np.concatenate(mem_list, axis=0), np.concatenate(pos_list, axis=0)

    def reset(self) -> None:
        self._entries = deque(maxlen=self.max_slots)

    def __len__(self) -> int:
        return len(self._entries)
