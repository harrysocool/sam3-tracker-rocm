"""ROCm environment defaults for gfx1151 (Strix Halo / Ryzen AI Max+ 395).

Call apply() at the top of any script that touches MIGraphX or the ROCm GPU
stack. Uses os.environ.setdefault — explicit user exports always take precedence.

Update the values here when upgrading the ROCm stack or targeting a new GPU.
Script-specific flags (MIGRAPHX_MLIR_USE_SPECIFIC_OPS, MIGRAPHX_SKIP_BENCHMARKING)
are intentionally NOT here — their logic varies per script.
"""
import os

# ---------------------------------------------------------------------------
# Global constants — one place to update for ROCm stack / GPU changes
# ---------------------------------------------------------------------------
HSA_OVERRIDE_GFX_VERSION = "11.5.1"
MIGRAPHX_GPU_HIP_FLAGS = "-Wno-error -Wno-lifetime-safety-intra-tu-suggestions"
TRANSFORMERS_OFFLINE = "1"

_DEFAULTS = {
    "HSA_OVERRIDE_GFX_VERSION": HSA_OVERRIDE_GFX_VERSION,
    "MIGRAPHX_GPU_HIP_FLAGS": MIGRAPHX_GPU_HIP_FLAGS,
    "TRANSFORMERS_OFFLINE": TRANSFORMERS_OFFLINE,
}


def apply() -> None:
    """Apply ROCm env defaults. No-op for vars the user has already exported."""
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, v)
