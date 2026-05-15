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
#   sudo ROCM=/opt/rocm-7.2.3 bash tools/install_migraphx_patched.sh  # override if needed
set -euo pipefail

BUILD="${BUILD:-./build_docker}"

# Auto-detect ROCm install path (supports 7.2.0, 7.2.1, 7.2.2, 7.2.3 …).
# Override with:  sudo ROCM=/opt/rocm-7.2.3 bash install_migraphx_patched.sh
if [[ -z "${ROCM:-}" ]]; then
    # Prefer the path reported by the APT package itself
    _dpkg_path="$(dpkg -L migraphx 2>/dev/null | grep 'lib/libmigraphx\.so$' | head -1 | sed 's|/lib/libmigraphx\.so||')"
    if [[ -n "$_dpkg_path" && -d "$_dpkg_path/lib" ]]; then
        ROCM="$_dpkg_path"
    else
        # Fall back: scan /opt/rocm-7.2.* descending (pick newest patch)
        for _p in $(ls -d /opt/rocm-7.2.* 2>/dev/null | sort -rV); do
            [[ -d "$_p/lib" ]] && ROCM="$_p" && break
        done
        # Last resort
        ROCM="${ROCM:-/opt/rocm-7.2.0}"
    fi
fi
echo "  ROCm path: $ROCM"

if [[ ! -d "$BUILD/lib" ]]; then
    echo "ERROR: $BUILD/lib not found. Set BUILD=/path/to/AMDMIGraphX/build_docker." >&2
    exit 1
fi
if [[ ! -d "$ROCM/lib" ]]; then
    echo "ERROR: $ROCM/lib not found. Set ROCM= to the ROCm 7.2.x install prefix." >&2
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
# libmigraphx_cpu is new in 2.16+ (no stock 2.15 file to back up).
for lib in libmigraphx libmigraphx_gpu libmigraphx_device libmigraphx_onnx libmigraphx_tf libmigraphx_ref; do
    backup_if_missing "$ROCM/lib/migraphx/lib/${lib}.so.2015000.0.70200"
done

echo
echo "=== Installing patched core C API ==="
cp "$BUILD/lib/libmigraphx_c.so.3.0" "$ROCM/lib/libmigraphx_c.so.3.0.2016000"
ln -sf libmigraphx_c.so.3.0.2016000 "$ROCM/lib/libmigraphx_c.so.3.0.70200"

echo
echo "=== Installing patched internal libs to $ROCM/lib/migraphx/lib ==="
# libmigraphx_ref and libmigraphx_cpu are required by the Python binding's
# DT_NEEDED — `import migraphx` fails immediately without them.
for lib in libmigraphx libmigraphx_gpu libmigraphx_device libmigraphx_onnx libmigraphx_tf libmigraphx_ref libmigraphx_cpu; do
    src="$BUILD/lib/${lib}.so.2016000.0"
    if [[ -f "$src" ]]; then
        cp "$src" "$ROCM/lib/migraphx/lib/"
        # Re-point the stock-version symlink only if there was a stock backup
        # (skip for libmigraphx_cpu which is new in 2.16+).
        if [[ -e "$ROCM/lib/migraphx/lib/${lib}.so.2015000.0.70200.bak" ]]; then
            (cd "$ROCM/lib/migraphx/lib" && \
                ln -sf "${lib}.so.2016000.0" "${lib}.so.2015000.0.70200")
        fi
        echo "  installed: ${lib}.so.2016000.0"
    else
        echo "  WARNING: $src not found, skipping"
    fi
done

echo
echo "=== Installing bundled vendor libs (libdnnl, libomp) ==="
# The migraphx Python binding and libmigraphx_cpu.so DT_NEEDED libdnnl.so.1
# (OneDNN) and libomp.so. Stock ROCm 7.2 APT does NOT ship these — the
# patched MIGraphX build provides them. Skip silently if not in BUILD.
for f in libdnnl.so.1 libomp.so; do
    if [[ -f "$BUILD/lib/$f" ]]; then
        cp "$BUILD/lib/$f" "$ROCM/lib/migraphx/lib/"
        echo "  installed: $f"
    fi
done

echo
echo "=== Registering migraphx lib path with dynamic linker ==="
# /opt/rocm-7.2.0/lib/migraphx/lib/ is not on the default ld search path.
# Without this, libmigraphx_tf.so.2016000 (and friends) cannot be located
# at runtime — the Python binding fails with ImportError.
LD_CONF="/etc/ld.so.conf.d/rocm-migraphx-2016.conf"
if [[ ! -f "$LD_CONF" ]] || ! grep -q "$ROCM/lib/migraphx/lib" "$LD_CONF"; then
    echo "$ROCM/lib/migraphx/lib" > "$LD_CONF"
    echo "  wrote: $LD_CONF"
else
    echo "  $LD_CONF already includes $ROCM/lib/migraphx/lib"
fi

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
