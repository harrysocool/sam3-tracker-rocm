"""Resolve the Torch wheel against the system ROCm installation.

AMD's ROCm 10.1 PyTorch wheel normally preloads ROCm libraries from Python
wheels. The container supplies the same-version ROCm 10.1 libraries under
/opt/rocm. Skipping that preload avoids importing a second, wheel-supplied ROCm
stack.
"""

from pathlib import Path
import glob

__version__ = "10.1.0"


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
