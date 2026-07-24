#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_DYNAMIC_K_WIP:-0}" != "1" ]]; then
  echo "Refusing to build unvalidated dynamic-K backbone." >&2
  echo "Set ALLOW_DYNAMIC_K_WIP=1 only for controlled development." >&2
  exit 2
fi

source /opt/xilinx/xrt/setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/bh_shared_dynamic_k.cpp"
OUT="${1:-/home/amd/project/npu_iron/bh_shared_dynamic_k}"

g++ -O3 -march=native -mavx512f -mavx512bf16 \
  -ffast-math -funroll-loops -fopenmp -std=c++17 \
  "$SRC" -o "$OUT" \
  -I/opt/xilinx/xrt/include \
  -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil

echo "Built: $OUT"
sha256sum "$SRC" "$OUT"
