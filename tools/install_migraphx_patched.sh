#!/bin/bash
# Install patched MIGraphX (`2.15+patches`, internally labelled 2.16.0-pre)
# from a CMake build into the stock ROCm 7.2 install tree, replacing the
# 2.15.0 system libraries. The stock libs are preserved as `*.bak`.
#
# See docs/build_migraphx_patched.md for the full build flow and rationale.
#
# Usage:
#   sudo bash tools/install_migraphx_patched.sh
#   sudo BUILD=/path/to/AMDMIGraphX/build_docker bash tools/install_migraphx_patched.sh
#   sudo ROCM=/opt/rocm-7.2.0 bash tools/install_migraphx_patched.sh
set -euo pipefail

BUILD="${BUILD:-./build_docker}"
ROCM="${ROCM:-/opt/rocm-7.2.0}"

if [[ ! -d "$BUILD/lib" ]]; then
    echo "ERROR: $BUILD/lib not found. Set BUILD=/path/to/AMDMIGraphX/build_docker." >&2
    exit 1
fi
if [[ ! -d "$ROCM/lib" ]]; then
    echo "ERROR: $ROCM/lib not found. Set ROCM=/opt/rocm-7.2.0 (or wherever ROCm 7.2 APT installed)." >&2
    exit 1
fi

echo "=== Backing up stock MIGraphX 2.15.0 libs (idempotent) ==="
backup_if_missing() {
    local f="$1"
    if [[ -f "$f" && ! -f "$f.bak" ]]; then
        cp "$f" "$f.bak"
        echo "  backed up: $(basename "$f")"
    fi
}
backup_if_missing "$ROCM/lib/libmigraphx_c.so.3.0.70200"
for lib in libmigraphx libmigraphx_gpu libmigraphx_device libmigraphx_onnx libmigraphx_tf; do
    backup_if_missing "$ROCM/lib/migraphx/lib/${lib}.so.2015000.0.70200"
done

echo
echo "=== Installing patched core C API ==="
cp "$BUILD/lib/libmigraphx_c.so.3.0" "$ROCM/lib/libmigraphx_c.so.3.0.2016000"
ln -sf libmigraphx_c.so.3.0.2016000 "$ROCM/lib/libmigraphx_c.so.3.0.70200"

echo
echo "=== Installing patched internal libs to $ROCM/lib/migraphx/lib ==="
for lib in libmigraphx libmigraphx_gpu libmigraphx_device libmigraphx_onnx libmigraphx_tf; do
    src="$BUILD/lib/${lib}.so.2016000.0"
    if [[ -f "$src" ]]; then
        cp "$src" "$ROCM/lib/migraphx/lib/"
        # Re-point the .2015000 symlinks the system loader uses
        (cd "$ROCM/lib/migraphx/lib" && \
            ln -sf "${lib}.so.2016000.0" "${lib}.so.2015000.0.70200")
        echo "  installed: ${lib}.so.2016000.0"
    else
        echo "  WARNING: $src not found, skipping"
    fi
done

echo
echo "=== Installing Python 3.12 binding ==="
cp "$BUILD/lib/migraphx.cpython-312-x86_64-linux-gnu.so" "$ROCM/lib/"

# The shared py loaders are also rebuilt; install side-by-side as .new so the
# stock py 3.13 binding (used by other tools on the system) keeps working.
for f in libmigraphx_py.so libmigraphx_py_3.10.so; do
    if [[ -f "$BUILD/lib/$f" ]]; then
        cp "$BUILD/lib/$f" "$ROCM/lib/$f.new"
    fi
done

ldconfig

echo
echo "=== Done ==="
echo "Verify with:"
echo "  python3 -c \"import migraphx; print('MIGraphX from:', migraphx.__file__)\""
echo "Expected path: $ROCM/lib/migraphx.cpython-312-x86_64-linux-gnu.so"
echo
echo "To rollback, see docs/build_migraphx_patched.md (Rollback section)."
