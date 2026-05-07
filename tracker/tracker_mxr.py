"""
SAM3OnnxTrackerMXR — full MIGraphX backbone via ORT session cache.

Uses three ORT MIGraphX sessions for the vision encoder backbone,
eliminating PyTorch backbone latency. ORT auto-compiles on first run
and loads from disk cache on subsequent runs (~2-7s vs 680s cold-start).

Pipeline:
  pixel_values → backbone_part1.onnx (MIGraphX) → bhwd_features
               → backbone_block31.onnx (MIGraphX) → y_bhwd
               → [numpy BHWD→BCHW permute]
               → backbone_fpn.onnx (MIGraphX) → fpn_0, fpn_1, fpn_2
  fpn_* → mask_decoder_init / propagate (ONNX CPU)
  fpn_2 → memory_attention_fixed_N7.onnx (MIGraphX)
        → mask_decoder_propagate → memory_encoder

Prerequisites:
  Run `python export/export_backbone.py --imgsz 504 --output-dir onnx_files`
  once to generate backbone ONNX files. ORT session cache is built
  automatically on first inference (~10 min total), then reused.
"""

from __future__ import annotations
import os, sys, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")
os.environ.setdefault("MIGRAPHX_GPU_HIP_FLAGS",
                      "-Wno-error -Wno-lifetime-safety-intra-tu-suggestions")

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
from tracker.tracker import MemoryBank, preprocess_image, _MEAN, _STD


def _mig_providers(cache_dir: str):
    return [
        ("MIGraphXExecutionProvider", {"migraphx_model_cache_dir": cache_dir}),
        ("CPUExecutionProvider", {}),
    ]


class SAM3OnnxTrackerMXR:
    """SAM3 tracker with full MIGraphX backbone (ORT sessions, persistent cache)."""

    def __init__(
        self,
        onnx_dir:    str | Path,
        mxr_cache_dir: str | Path,
        pos2_npy:    str | Path,
        temporal_pe_npy: str | Path,
        imgsz:       int = 504,
        num_maskmem: int = 7,
    ):
        self.imgsz         = imgsz
        self.H = self.W    = imgsz // 14
        self.HW            = self.H * self.W
        self.num_maskmem   = num_maskmem
        self.mask_mem_size = self.H * 16

        onnx_dir   = Path(onnx_dir)
        cache_dir  = str(mxr_cache_dir)
        mig        = _mig_providers(cache_dir)

        cpu_opts = ort.SessionOptions()
        cpu_opts.intra_op_num_threads = 8
        CPU = ["CPUExecutionProvider"]

        print("Loading backbone (MIGraphX, with ORT cache)...")
        t0 = time.perf_counter()
        self._bb_part1  = ort.InferenceSession(str(onnx_dir/"backbone_part1.onnx"),  providers=mig)
        self._bb_block31= ort.InferenceSession(str(onnx_dir/"backbone_block31.onnx"),providers=mig)
        self._bb_fpn    = ort.InferenceSession(str(onnx_dir/"backbone_fpn.onnx"),    providers=mig)
        print(f"  backbone ready in {time.perf_counter()-t0:.1f}s"
              f"  [{self._bb_part1.get_providers()[0]}]")

        print("Loading tracking modules...")
        self._mem_attn = ort.InferenceSession(
            str(onnx_dir/"memory_attention_fixed_N7.onnx"), providers=mig)
        print(f"  memory_attention [{self._mem_attn.get_providers()[0]}]")

        self._dec_init = ort.InferenceSession(
            str(onnx_dir/"mask_decoder_init.onnx"), sess_options=cpu_opts, providers=CPU)
        self._dec_prop = ort.InferenceSession(
            str(onnx_dir/"mask_decoder_propagate.onnx"), sess_options=cpu_opts, providers=CPU)
        self._mem_enc  = ort.InferenceSession(
            str(onnx_dir/"memory_encoder.onnx"), sess_options=cpu_opts, providers=CPU)

        self._pos2 = np.load(str(pos2_npy)).astype(np.float32)
        temporal_pe = np.load(str(temporal_pe_npy)).astype(np.float32)
        self.memory_bank = MemoryBank(temporal_pe, max_slots=num_maskmem)
        self._frame_idx  = 0
        self._timings    = {k: [] for k in
                            ["backbone","mem_attn","dec_init","dec_prop","mem_enc","total"]}

    def reset(self):
        self.memory_bank.reset()
        self._frame_idx = 0

    def _backbone(self, img_np: np.ndarray):
        t0 = time.perf_counter()
        o1, = self._bb_part1.run(None,  {"pixel_values": img_np})
        o2, = self._bb_block31.run(None, {"x_bhwd": o1})
        bchw = np.ascontiguousarray(o2.transpose(0, 3, 1, 2))
        f0, f1, f2 = self._bb_fpn.run(None, {"x_bchw": bchw})
        self._timings["backbone"].append(time.perf_counter() - t0)
        return f0, f1, f2, self._pos2

    def _encode_memory(self, fpn_2, pred_masks):
        t0 = time.perf_counter()
        mask_2d = pred_masks.squeeze().astype(np.float32)
        mask_r  = cv2.resize(mask_2d, (self.mask_mem_size, self.mask_mem_size),
                             interpolation=cv2.INTER_LINEAR)[None, None]
        mf, mp  = self._mem_enc.run(None, {"vision_features": fpn_2, "masks": mask_r})
        self._timings["mem_enc"].append(time.perf_counter() - t0)
        self.memory_bank.push(mf, mp)

    def init_frame(self, img_np: np.ndarray, box: list[float]):
        t_total = time.perf_counter()
        f0, f1, f2, _ = self._backbone(img_np)
        box_np = np.array([[[[box[0], box[1], box[2], box[3]]]]], dtype=np.float32)
        pts    = np.zeros((1, 1, 1, 2), dtype=np.float32)
        lbls   = np.full((1, 1, 1), -1, dtype=np.int32)
        t0 = time.perf_counter()
        masks, _, score = self._dec_init.run(None, {
            "fpn_2": f2, "fpn_0": f0, "fpn_1": f1,
            "input_points": pts, "input_labels": lbls, "input_boxes": box_np,
        })
        self._timings["dec_init"].append(time.perf_counter() - t0)
        binary = masks.squeeze() > 0
        self._encode_memory(f2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary, float(score.flat[0])

    def propagate_frame(self, img_np: np.ndarray):
        t_total = time.perf_counter()
        f0, f1, f2, pos2 = self._backbone(img_np)
        memory, memory_pos = self.memory_bank.build_attention_inputs(
            fixed_slots=self.num_maskmem)
        t0 = time.perf_counter()
        cond = self._mem_attn.run(None, {
            "current_vision_features": f2.reshape(1, 256, self.HW).transpose(2, 0, 1),
            "memory":                  memory,
            "current_vis_pos_embed":   pos2.reshape(1, 256, self.HW).transpose(2, 0, 1),
            "memory_pos_embed":        memory_pos,
        })
        self._timings["mem_attn"].append(time.perf_counter() - t0)
        cond_fpn2 = cond[0]
        t0 = time.perf_counter()
        masks, _, score = self._dec_prop.run(None, {
            "fpn_2_cond": cond_fpn2, "fpn_0": f0, "fpn_1": f1,
        })
        self._timings["dec_prop"].append(time.perf_counter() - t0)
        binary = masks.squeeze() > 0
        self._encode_memory(f2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary, float(score.flat[0])

    def print_timings(self, warmup: int = 0):
        import statistics as st
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  SAM3 MXR Tracker — {self.imgsz}px (ORT MIGraphX backbone)")
        print(sep)
        def row(label, key):
            v = self._timings[key][warmup:]
            if not v: return
            m, s = st.mean(v)*1000, st.stdev(v)*1000 if len(v)>1 else 0
            print(f"  {label:<40}  {m:>7.1f} ± {s:>4.1f} ms")
        row("backbone  [MIGraphX x3]", "backbone")
        row("memory_attention  [MIGraphX]", "mem_attn")
        row("mask_decoder_propagate  [ONNX CPU]", "dec_prop")
        row("memory_encoder  [ONNX CPU]", "mem_enc")
        tot = self._timings["total"][warmup:]
        if tot:
            m = st.mean(tot)*1000
            print(f"  {'─'*58}")
            print(f"  {'Total':<42}  {m:>7.1f} ms  →  {1000/m:.2f} FPS")
        print(sep + "\n")
