#!/usr/bin/env bash
# Build the instrumented SAM3 IRON backbone profiler.
set -euo pipefail

source /opt/xilinx/xrt/setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/bh_flash_profile.cpp"
OUT="${1:-/home/amd/project/npu_iron/bh_npu_backbone_flash_profile}"

g++ -O3 -march=native -mavx512f -mavx512bf16 \
  -ffast-math -funroll-loops -fopenmp -std=c++17 \
  "$SRC" -o "$OUT" \
  -I/opt/xilinx/xrt/include \
  -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil

echo "Built: $OUT"
sha256sum "$SRC" "$OUT"
