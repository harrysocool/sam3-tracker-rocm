"""Resolve a gfx1151 PyTorch wheel against the system ROCm installation.

AMD's gfx1151 PyTorch wheel is currently built against ROCm 7.13 and normally
preloads ROCm libraries from Python wheels. The container supplies compatible
ROCm 7.14 libraries under /opt/rocm. Skipping that preload avoids importing
gfx942-only libraries from the builder image.
"""

from pathlib import Path
import glob

__version__ = "7.14.0"


def initialize_process(**_kwargs):
    """Let the dynamic linker resolve ROCm libraries from LD_LIBRARY_PATH."""


def find_libraries(*shortnames: str) -> list[Path]:
    """Resolve requested ROCm shared libraries below /opt/rocm."""
    paths: list[Path] = []
    for shortname in shortnames:
        matches = sorted(glob.glob(f"/opt/rocm/lib/lib{shortname}.so*"))
        if not matches:
            matches = sorted(
                glob.glob(f"/opt/rocm/**/lib{shortname}.so*", recursive=True)
            )
        if not matches:
            raise FileNotFoundError(f"ROCm library not found: {shortname}")
        paths.append(Path(matches[0]))
    return paths
