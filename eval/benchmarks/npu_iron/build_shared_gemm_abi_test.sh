#!/usr/bin/env bash
set -euo pipefail

source /opt/xilinx/xrt/setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/shared_gemm_abi_test.cpp"
OUT="${1:-/home/amd/project/npu_iron/shared_gemm_abi_test_v5}"

g++ -O2 -std=c++17 "$SRC" -o "$OUT" \
  -I/opt/xilinx/xrt/include \
  -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil

echo "Built: $OUT"
sha256sum "$SRC" "$OUT"
