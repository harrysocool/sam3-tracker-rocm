#!/usr/bin/env bash
set -euo pipefail

source /opt/xilinx/xrt/setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "${ALLOW_DYNAMIC_K_V5_WIP:-0}" != "1" ]]; then
  echo "Refusing to build unvalidated dynamic-K v5 backbone without ALLOW_DYNAMIC_K_V5_WIP=1" >&2
  exit 2
fi

SRC="$SCRIPT_DIR/bh_phase2_dynamic_v5_wip.cpp"
OUT="${1:-/home/amd/project/npu_iron/bh_phase2_dynamic_v5_wip}"

g++ -O3 -march=native -mavx512f -mavx512bf16 \
  -ffast-math -funroll-loops -fopenmp -std=c++17 \
  "$SRC" -o "$OUT" \
  -I/opt/xilinx/xrt/include \
  -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil

echo "Built: $OUT"
sha256sum "$SRC" "$OUT"
