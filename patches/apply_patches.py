#!/usr/bin/env python3
"""Apply ROCm compatibility patches to .local_deps/transformers.

Run after cloning DART and setting up the conda environment.
These patches enable fill_holes (scipy), greedy NMS (PyTorch), and
update the warning messages to reflect the scipy/PyTorch fallback paths.
"""
import argparse
import os
import sys
from pathlib import Path

DEFAULT_DART = Path(__file__).resolve().parent.parent.parent / "sam3" / "repo" / "DART"

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dart-root", type=Path, default=None,
               help="Path to the DART repo root (default: ../sam3/repo/DART relative to project)")
args = p.parse_args()

DART_ROOT = Path(os.environ.get("SAM3_DART_ROOT", "")).expanduser() if os.environ.get("SAM3_DART_ROOT") else (args.dart_root or DEFAULT_DART)
MODELING = DART_ROOT / ".local_deps/transformers/models/sam3_video/modeling_sam3_video.py"

if not MODELING.exists():
    print(f"[patches] WARNING: modeling_sam3_video.py not found at {MODELING}")
    print("[patches] Skipping ROCm patches — re-run manually: python patches/apply_patches.py --dart-root <path>")
    sys.exit(0)  # soft exit so setup.sh continues

src = MODELING.read_text()

# ── Patch 1: warning message ────────────────────────────────────────────────
OLD1 = (
    '        logger.warning_once(\n'
    '            "kernels library is not installed. NMS post-processing, hole filling, and sprinkle removal will be skipped. "\n'
    '            "Install it with `pip install kernels` for better mask quality."\n'
    '        )'
)
NEW1 = (
    '        logger.warning_once(\n'
    '            "kernels library is not installed or has no build for this platform. "\n'
    '            "Falling back to scipy.ndimage for hole filling and sprinkle removal "\n'
    '            "(NMS post-processing skipped). Install `pip install kernels` with a "\n'
    '            "matching CUDA build for better performance."\n'
    '        )'
)
if OLD1 in src:
    src = src.replace(OLD1, NEW1)
    print("  ✓ Patch 1: warning message")
elif NEW1 in src:
    print("  - Patch 1: already applied")
else:
    print("  ✗ Patch 1: FAILED (source changed?)")

# ── Patch 2: scipy connected components ────────────────────────────────────
OLD2 = (
    '    if not cv_utils_kernel:\n'
    '        # Fallback: return dummy labels and counts that won\'t trigger filtering\n'
    '        labels = torch.zeros_like(mask, dtype=torch.int32)\n'
    '        counts = torch.full_like(mask, fill_value=mask.shape[2] * mask.shape[3] + 1, dtype=torch.int32)\n'
    '        return labels, counts'
)
NEW2 = (
    '    if not cv_utils_kernel:\n'
    '        try:\n'
    '            import scipy.ndimage\n'
    '            import numpy as np\n'
    '            labels = torch.zeros_like(mask, dtype=torch.int32)\n'
    '            counts = torch.zeros_like(mask, dtype=torch.int32)\n'
    '            np_mask = mask.cpu().numpy()\n'
    '            B, C = np_mask.shape[:2]\n'
    '            for b in range(B):\n'
    '                for c in range(C):\n'
    '                    lbl, n = scipy.ndimage.label(np_mask[b, c])\n'
    '                    lbl_t = torch.from_numpy(lbl.astype(np.int32))\n'
    '                    labels[b, c] = lbl_t\n'
    '                    if n > 0:\n'
    '                        areas = torch.bincount(lbl_t.flatten(), minlength=n + 1)\n'
    '                        counts[b, c] = areas[lbl_t]\n'
    '                    else:\n'
    '                        counts[b, c] = H * W + 1\n'
    '            return labels.to(mask.device), counts.to(mask.device)\n'
    '        except Exception:\n'
    '            pass\n'
    '        labels = torch.zeros_like(mask, dtype=torch.int32)\n'
    '        counts = torch.full_like(mask, fill_value=H * W + 1, dtype=torch.int32)\n'
    '        return labels, counts'
)
if OLD2 in src:
    src = src.replace(OLD2, NEW2)
    print("  ✓ Patch 2: scipy connected components")
elif "scipy.ndimage" in src:
    print("  - Patch 2: already applied")
else:
    print("  ✗ Patch 2: FAILED")

# ── Patch 3: greedy NMS + nms_masks fallback ───────────────────────────────
HELPER = (
    '\ndef _greedy_nms(ious: torch.Tensor, probs: torch.Tensor, iou_threshold: float) -> torch.Tensor:\n'
    '    """Pure-PyTorch greedy NMS on a precomputed (N, N) IoU matrix."""\n'
    '    order = probs.argsort(descending=True)\n'
    '    suppressed = torch.zeros(len(probs), dtype=torch.bool, device=probs.device)\n'
    '    kept = []\n'
    '    for idx in order:\n'
    '        if suppressed[idx]:\n'
    '            continue\n'
    '        kept.append(idx)\n'
    '        suppressed = suppressed | (ious[idx] > iou_threshold)\n'
    '    if not kept:\n'
    '        return torch.zeros(0, dtype=torch.long, device=probs.device)\n'
    '    return torch.stack(kept)\n'
    '\n'
    '\n'
)
OLD3 = (
    '    if not cv_utils_kernel:\n'
    '        return is_valid  # Fallback: keep all valid detections without NMS\n'
    '\n'
    '    try:\n'
    '        kept_inds = cv_utils_kernel.generic_nms(ious, probs, iou_threshold, use_iou_matrix=True)\n'
    '    except Exception as e:\n'
    '        logger.warning_once(f"Failed to run NMS using kernels library: {e}. NMS post-processing will be skipped.")\n'
    '        return is_valid  # Fallback: keep all valid detections without NMS'
)
NEW3 = (
    '    if not cv_utils_kernel:\n'
    '        kept_inds = _greedy_nms(ious, probs, iou_threshold)\n'
    '    else:\n'
    '        try:\n'
    '            kept_inds = cv_utils_kernel.generic_nms(ious, probs, iou_threshold, use_iou_matrix=True)\n'
    '        except Exception as e:\n'
    '            logger.warning_once(f"Failed to run NMS using kernels library: {e}. Falling back to PyTorch NMS.")\n'
    '            kept_inds = _greedy_nms(ious, probs, iou_threshold)'
)
if "def _greedy_nms" not in src and OLD3 in src:
    src = HELPER.join(src.split("def nms_masks(", 1))
    src = src.replace(OLD3, NEW3)
    print("  ✓ Patch 3: PyTorch greedy NMS")
elif "_greedy_nms" in src:
    print("  - Patch 3: already applied")
else:
    print("  ✗ Patch 3: FAILED")

MODELING.write_text(src)
print("\nDone.")
