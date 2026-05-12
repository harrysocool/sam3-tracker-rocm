#!/usr/bin/env python3
"""Microbenchmark MIGVisionEncoder sub-steps."""
import time, statistics, sys
from pathlib import Path
import numpy as np
import torch

from tracker.tracker import MIGraphXBackbone

device = torch.device("cuda")
dtype = torch.float16
ITERS = 10

mxr = MIGraphXBackbone(
    onnx_path=Path("onnx_files_1008/backbone_detector/single_simplified.onnx"),
    cache_path=Path("onnx_files_1008/backbone_detector/tuned.mxr"),
)
mxr.warmup(n=3)

pixel_values = torch.randn(1, 3, 1008, 1008, device=device, dtype=dtype)

def t(label, fn, n=ITERS):
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"  {label:<48s} {statistics.mean(times[2:])*1000:>8.2f} ms")
    return out

print(f"Input: {pixel_values.shape} {pixel_values.dtype} on {pixel_values.device}\n")

print("=== Per sub-step (mean of last 8 of 10 runs) ===\n")

# 1. Input conversion: torch.cuda.fp16 → np.float32 cpu
def step_in():
    return pixel_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
np_in = t("[in] torch.cuda.fp16 → np.float32 cpu", step_in)

# 2. MIG run
def step_mig():
    return mxr(np_in)
outs = t("[mig] mxr(np_in) — pure compute", step_mig)
f0, f1, f2, f3, lhs = outs
print(f"      MIG output shapes: {[o.shape if o is not None else None for o in outs]}")
print(f"      total output bytes: {sum(o.nbytes for o in outs if o is not None)/1e6:.1f} MB\n")

# 3. Output conversion variants
def step_out_current():
    fpn = [torch.from_numpy(np.ascontiguousarray(t)).to(device=device, dtype=dtype)
           for t in (f0, f1, f2, f3)]
    lhs_t = torch.from_numpy(np.ascontiguousarray(lhs)).to(device=device, dtype=dtype)
    return fpn, lhs_t
t("[out current] np.fp32 cpu → torch.cuda.fp16", step_out_current)

def step_out_fp16_first():
    # Cast to fp16 in numpy first (halves transfer)
    fpn = [torch.from_numpy(t.astype(np.float16, copy=False)).to(device=device, non_blocking=True)
           for t in (f0, f1, f2, f3)]
    lhs_t = torch.from_numpy(lhs.astype(np.float16, copy=False)).to(device=device, non_blocking=True)
    torch.cuda.synchronize()
    return fpn, lhs_t
t("[out fp16-first] cast np→fp16 then transfer", step_out_fp16_first)

# Persistent pinned buffers
print("\n=== Persistent buffer reuse ===\n")
pinned = {}
gpu = {}
shapes_dtypes = [("f0", f0.shape, np.float16), ("f1", f1.shape, np.float16),
                 ("f2", f2.shape, np.float16), ("f3", f3.shape, np.float16),
                 ("lhs", lhs.shape, np.float16)]
for name, shape, dt in shapes_dtypes:
    pinned[name] = torch.zeros(shape, dtype=torch.float16, pin_memory=True).numpy()
    gpu[name] = torch.empty(shape, dtype=torch.float16, device=device)

def step_out_pinned():
    pinned["f0"][...] = f0.astype(np.float16, copy=False)
    pinned["f1"][...] = f1.astype(np.float16, copy=False)
    pinned["f2"][...] = f2.astype(np.float16, copy=False)
    pinned["f3"][...] = f3.astype(np.float16, copy=False)
    pinned["lhs"][...] = lhs.astype(np.float16, copy=False)
    for name in ("f0", "f1", "f2", "f3", "lhs"):
        gpu[name].copy_(torch.from_numpy(pinned[name]), non_blocking=True)
    torch.cuda.synchronize()
    return [gpu[n] for n in ("f0", "f1", "f2", "f3")], gpu["lhs"]
t("[out pinned + reuse-gpu] copy_(non_blocking=True)", step_out_pinned)

# End-to-end variants
print("\n=== End-to-end forward variants ===\n")

def fwd_current():
    np_in = pixel_values.detach().float().cpu().numpy()
    outs = mxr(np_in)
    f0, f1, f2, f3, lhs = outs
    fpn = [torch.from_numpy(np.ascontiguousarray(t)).to(device=device, dtype=dtype)
           for t in (f0, f1, f2, f3)]
    lhs_t = torch.from_numpy(np.ascontiguousarray(lhs)).to(device=device, dtype=dtype)
    return fpn, lhs_t
t("[full current]", fwd_current)

def fwd_fp16_first():
    np_in = pixel_values.detach().float().cpu().numpy()
    outs = mxr(np_in)
    f0, f1, f2, f3, lhs = outs
    fpn = [torch.from_numpy(t.astype(np.float16, copy=False)).to(device=device, non_blocking=True)
           for t in (f0, f1, f2, f3)]
    lhs_t = torch.from_numpy(lhs.astype(np.float16, copy=False)).to(device=device, non_blocking=True)
    torch.cuda.synchronize()
    return fpn, lhs_t
t("[full fp16-first]", fwd_fp16_first)

def fwd_pinned_full():
    np_in = pixel_values.detach().float().cpu().numpy()
    outs = mxr(np_in)
    f0v, f1v, f2v, f3v, lhsv = outs
    pinned["f0"][...] = f0v.astype(np.float16, copy=False)
    pinned["f1"][...] = f1v.astype(np.float16, copy=False)
    pinned["f2"][...] = f2v.astype(np.float16, copy=False)
    pinned["f3"][...] = f3v.astype(np.float16, copy=False)
    pinned["lhs"][...] = lhsv.astype(np.float16, copy=False)
    for name in ("f0", "f1", "f2", "f3", "lhs"):
        gpu[name].copy_(torch.from_numpy(pinned[name]), non_blocking=True)
    torch.cuda.synchronize()
    return [gpu[n] for n in ("f0", "f1", "f2", "f3")], gpu["lhs"]
t("[full pinned + reuse]", fwd_pinned_full)
