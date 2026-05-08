"""
SAM3OnnxTracker — mask-level video tracker using SAM3 + ONNX Runtime.

Pipeline:
  Frame 0 (init):  box/point prompt → mask_decoder_init.onnx → mask + object pointer
  Frame t (prop):  memory_attention.onnx (MIGraphX) → conditioned features
                   mask_decoder_propagate.onnx → mask + object pointer
  Every frame:     memory_encoder.onnx → memory entry → FIFO bank (max 7 frames)

Backbone: MIGraphX compiled (backbone_mxr_tuned.mxr, ~88ms at 504px) when available,
          falling back to PyTorch ROCm FP16 (139ms). Pass backbone="pytorch" to force
          PyTorch. All other modules are CPU ONNX, except memory_attention which runs
          on MIGraphX when the fixed-N7 file is present.
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
os.environ.setdefault("MIGRAPHX_GPU_HIP_FLAGS", "-Wno-error -Wno-lifetime-safety-intra-tu-suggestions")


# ---------------------------------------------------------------------------
# MIGraphX backbone (patched MIGraphX 2.16.0, ~88ms at 504px)
# ---------------------------------------------------------------------------

_MXR_BUILD_LIB = "/home/amd/project/tools/AMDMIGraphX/build_docker/lib"


class MIGraphXBackbone:
    """
    SAM3 vision encoder compiled with patched MIGraphX 2.16.0.

    Achieves ~88ms at 504px vs PyTorch ROCm's 139ms (1.6× speedup).
    Uses backbone_single_simplified.onnx with kernel autotuning baked into
    the .mxr cache file; first-time compilation takes ~3 minutes.

    Outputs: (fpn_0, fpn_1, fpn_2, None) as float32 numpy arrays.
    pos_2 is None because the ONNX does not export position encodings;
    SAM3OnnxTracker.propagate_frame() uses zeros when pos_2 is None.
    """

    def __init__(self, onnx_path: str | Path, cache_path: str | Path) -> None:
        import sys
        # Prefer the ROCm lib dir where the Python binding lives; fall back to build dir.
        _mxr_py_dir = "/opt/rocm-7.2.0/lib"
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
        """img_np: (1, 3, H, W) float32  →  (fpn_0, fpn_1, fpn_2, pos_2=None)"""
        # Keep explicit references to prevent GC of data/argument before GPU finishes.
        img_cont = np.ascontiguousarray(img_np)
        arg      = self._mxr.argument(img_cont)
        outputs  = self._prog.run({"pixel_values": arg})
        return np.array(outputs[0]), np.array(outputs[1]), np.array(outputs[2]), None


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
    pe = model.prompt_encoder
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


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SAM3OnnxTracker:
    """
    SAM3 mask-level video tracker using ONNX tracking modules.

    Args:
        checkpoint:   Path to facebook/sam3 model directory.
        onnx_dir:     Directory containing the exported ONNX files.
        imgsz:        Input resolution (default 504; use 1008 for higher quality).
        num_maskmem:  Memory bank size (default 7).
        backbone:     "auto" (default) — use MIGraphX if cache/onnx present, else PyTorch.
                      "migraphx"       — require MIGraphX backbone (88ms at 504px).
                      "pytorch"        — require PyTorch ROCm backbone (139ms at 504px).
    """

    def __init__(
        self,
        checkpoint: str | Path,
        onnx_dir: str | Path,
        imgsz: int = 504,
        num_maskmem: int = 7,
        backbone: str = "auto",
    ):
        import onnxruntime as ort

        self.imgsz         = imgsz
        self.H = self.W    = imgsz // 14
        self.HW            = self.H * self.W
        self.num_maskmem   = num_maskmem
        self.mask_mem_size = self.H * 16
        onnx_dir = Path(onnx_dir)

        _mxr_cache  = onnx_dir / "backbone_mxr_tuned.mxr"
        _mxr_onnx   = onnx_dir / "backbone_single_simplified.onnx"
        _temporal_pe = onnx_dir / "temporal_pe.npy"
        _use_mxr    = (backbone == "migraphx") or (
            backbone == "auto" and (_mxr_cache.exists() or _mxr_onnx.exists())
        )

        if _use_mxr:
            # MIGraphX backbone path: no torch dependency whatsoever.
            # temporal_pe is loaded from a pre-saved .npy file (generated once during export).
            if not _temporal_pe.exists():
                raise FileNotFoundError(
                    f"{_temporal_pe} not found. Run export/export_tracker_modules.py once "
                    f"to generate it, or copy it from another onnx_files directory."
                )
            temporal_pe = np.load(str(_temporal_pe))

            self._mxr_backbone  = MIGraphXBackbone(_mxr_onnx, _mxr_cache)
            self.vision_encoder = None
            self.device         = None

        else:
            # ---- Backbone: PyTorch ROCm GPU FP16 ----
            import torch
            from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
                Sam3TrackerVideoModel,
            )

            self._mxr_backbone = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading SAM3 from {checkpoint} (device={self.device}) ...")
            model = Sam3TrackerVideoModel.from_pretrained(
                str(checkpoint), attn_implementation="eager"
            ).to(self.device).half().eval()

            if imgsz != 1008:
                retarget_resolution(model, imgsz)

            self.vision_encoder = model.vision_encoder
            temporal_pe = model.memory_temporal_positional_encoding.detach().cpu().numpy()

            # ---- AMD TunableOp: GEMM kernel autotuner ----
            _tunableop = False
            if self.device.type == "cuda":
                try:
                    torch.cuda.tunable.enable(val=True)
                    torch.cuda.tunable.tuning_enable(val=True)
                    _tunableop = True
                except Exception:
                    pass

            # ---- Warmup (also triggers TunableOp tuning) ----
            print("  Warming up GPU kernels ...")
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=self.device, dtype=torch.float16)
            n_warmup = 8 if _tunableop else 1
            with torch.inference_mode():
                for _ in range(n_warmup):
                    self.vision_encoder(dummy, return_dict=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                if _tunableop:
                    torch.cuda.tunable.tuning_enable(val=False)
                    print("  TunableOp tuning complete.")
            print("  Warmup complete.")

        # ---- ORT session options: 8 CPU threads (optimal for mask_decoder + memory_encoder) ----
        cpu_opts = ort.SessionOptions()
        cpu_opts.intra_op_num_threads = 8
        cpu_opts.inter_op_num_threads = 1
        cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        CPU = ["CPUExecutionProvider"]
        MIG = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]

        # ---- Load ONNX modules ----
        print("  Loading ONNX tracking modules ...")
        self.dec_init = ort.InferenceSession(
            str(onnx_dir / "mask_decoder_init.onnx"), sess_options=cpu_opts, providers=CPU)
        self.dec_prop = ort.InferenceSession(
            str(onnx_dir / "mask_decoder_propagate.onnx"), sess_options=cpu_opts, providers=CPU)
        self.mem_enc  = ort.InferenceSession(
            str(onnx_dir / "memory_encoder.onnx"), sess_options=cpu_opts, providers=CPU)

        # memory_attention: prefer fixed-N7 on MIGraphX (avoids dangling reference bug)
        # Falls back to CPU if MIGraphX compilation fails (e.g. OOM when PyTorch is loaded)
        mem_attn_fixed = onnx_dir / "memory_attention_fixed_N7.onnx"
        if mem_attn_fixed.exists():
            try:
                self.mem_attn = ort.InferenceSession(str(mem_attn_fixed), providers=MIG)
                self._mem_attn_slots = num_maskmem
                print("  memory_attention: MIGraphX (fixed N=7)")
            except Exception as e:
                print(f"  memory_attention: MIGraphX failed ({str(e)[:60]}), falling back to CPU")
                self.mem_attn = ort.InferenceSession(str(mem_attn_fixed), providers=CPU)
                self._mem_attn_slots = num_maskmem
                print("  memory_attention: CPU (fixed N=7)")
        else:
            self.mem_attn = ort.InferenceSession(
                str(onnx_dir / "memory_attention.onnx"), providers=CPU)
            self._mem_attn_slots = 0
            print("  memory_attention: CPU (dynamic) — export fixed_N7 for MIGraphX speedup")

        self.memory_bank = MemoryBank(temporal_pe, max_slots=num_maskmem)
        self._frame_idx  = 0
        self._timings: dict[str, list] = {
            k: [] for k in ["backbone", "mem_attn", "dec_init", "dec_prop", "mem_enc", "total"]
        }

        # Re-warmup MIGraphX backbone after all ORT sessions are created.
        # ORT's MIGraphX EP initialization can reset GPU state, invalidating earlier warmup.
        if self._mxr_backbone is not None:
            self._mxr_backbone.warmup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backbone(self, img_np: np.ndarray):
        t0 = time.perf_counter()
        if self._mxr_backbone is not None:
            result = self._mxr_backbone(img_np)
        else:
            pv = torch.from_numpy(img_np).to(self.device).half()
            with torch.inference_mode():
                vis = self.vision_encoder(pv, return_dict=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            result = (
                vis.fpn_hidden_states[0].float().cpu().numpy(),
                vis.fpn_hidden_states[1].float().cpu().numpy(),
                vis.fpn_hidden_states[2].float().cpu().numpy(),
                (vis.fpn_position_encoding[2].float().cpu().numpy()
                 if vis.fpn_position_encoding is not None else None),
            )
        self._timings["backbone"].append(time.perf_counter() - t0)
        return result

    def _encode_memory(self, fpn_2: np.ndarray, pred_masks: np.ndarray) -> None:
        t0 = time.perf_counter()
        mask_2d = pred_masks.squeeze().astype(np.float32)
        mask_r  = cv2.resize(mask_2d, (self.mask_mem_size, self.mask_mem_size),
                             interpolation=cv2.INTER_LINEAR)[None, None]
        mf, mp = self.mem_enc.run(None, {"vision_features": fpn_2, "masks": mask_r})
        self._timings["mem_enc"].append(time.perf_counter() - t0)
        self.memory_bank.push(mf, mp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_frame(
        self,
        img_np: np.ndarray,
        box: list[float],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Initialize tracking on the first frame.

        Args:
            img_np:        Preprocessed image (1, 3, H, W) float32.
            box:           [x1, y1, x2, y2] in imgsz pixel coordinates.
            point_coords:  Optional extra clicks (1, 1, N, 2).
            point_labels:  Optional click labels  (1, 1, N).

        Returns:
            (binary_mask, score)  —  mask shape (imgsz, imgsz) bool.
        """
        t_total = time.perf_counter()
        fpn_0, fpn_1, fpn_2, _ = self._backbone(img_np)

        box_np = np.array([[[[box[0], box[1], box[2], box[3]]]]], dtype=np.float32)
        if point_coords is None:
            pts  = np.zeros((1, 1, 1, 2), dtype=np.float32)
            lbls = np.full((1, 1, 1), -1, dtype=np.int32)
        else:
            pts  = point_coords.astype(np.float32)
            lbls = point_labels.astype(np.int32)

        t0 = time.perf_counter()
        masks, _, score = self.dec_init.run(None, {
            "fpn_2": fpn_2, "fpn_0": fpn_0, "fpn_1": fpn_1,
            "input_points": pts, "input_labels": lbls, "input_boxes": box_np,
        })
        self._timings["dec_init"].append(time.perf_counter() - t0)

        binary_mask = masks.squeeze() > 0
        self._encode_memory(fpn_2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary_mask, float(score.flat[0])

    def propagate_frame(self, img_np: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Propagate tracking to the next frame using memory bank.

        Args:
            img_np:  Preprocessed image (1, 3, H, W) float32.

        Returns:
            (binary_mask, score)  —  mask shape (imgsz, imgsz) bool.
        """
        t_total = time.perf_counter()
        fpn_0, fpn_1, fpn_2, pos_2 = self._backbone(img_np)

        cur_feat = fpn_2.reshape(1, 256, self.HW).transpose(2, 0, 1)
        cur_pos  = (pos_2.reshape(1, 256, self.HW).transpose(2, 0, 1)
                    if pos_2 is not None else np.zeros_like(cur_feat))

        memory, memory_pos = self.memory_bank.build_attention_inputs(
            fixed_slots=self._mem_attn_slots)
        if memory is None:
            cond = fpn_2
        else:
            t0 = time.perf_counter()
            cond = self.mem_attn.run(None, {
                "current_vision_features": cur_feat,
                "memory":                  memory,
                "current_vis_pos_embed":   cur_pos,
                "memory_pos_embed":        memory_pos,
            })[0]
            self._timings["mem_attn"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        masks, _, score = self.dec_prop.run(None, {
            "fpn_2_cond": cond, "fpn_0": fpn_0, "fpn_1": fpn_1,
        })
        self._timings["dec_prop"].append(time.perf_counter() - t0)

        binary_mask = masks.squeeze() > 0
        self._encode_memory(fpn_2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary_mask, float(score.flat[0])

    def reset(self) -> None:
        """Reset tracker state for a new video sequence."""
        self.memory_bank.reset()
        self._frame_idx = 0

    def print_timings(self) -> None:
        """Print per-module average latency and FPS."""
        print("\nPer-frame timing (ms):")
        for name, vals in self._timings.items():
            if vals:
                avg = np.mean(vals[1:] if len(vals) > 1 else vals) * 1000
                print(f"  {name:30s}: {avg:6.1f} ms")
        n = len(self._timings["total"])
        if n > 1:
            fps = 1.0 / np.mean(list(self._timings["total"])[1:])
            print(f"  {'FPS':30s}: {fps:.2f}")
