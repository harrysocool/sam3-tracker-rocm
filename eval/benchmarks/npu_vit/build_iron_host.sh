#!/usr/bin/env bash
set -euo pipefail

npu_root=${NPU_IRON_ROOT:-/home/amd/project/npu_iron}
release=${SAM3_IRON_RELEASE:-$npu_root/releases/sam3-vit-p14-m1536-power-v1}
block=$npu_root/mlir-aie-atb-20260727/programming_examples/ml/block_datatypes
output=${1:-/tmp/sam3-vit-p14-m1536.rebuilt}

[[ ! -e $output ]] || { echo "refusing to overwrite: $output" >&2; exit 2; }
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

g++ -O3 -march=native -mavx512f -mavx512bf16 \
  -ffast-math -funroll-loops -fopenmp -std=c++17 \
  "$release/src/backbone_p14_m1536.cpp" \
  -o "$output" \
  -I/opt/xilinx/xrt/include -I"$block" \
  -I"$block/gemm_asymmetric_tile_buffering" \
  -L/opt/xilinx/xrt/lib -lxrt_coreutil

sha256sum "$release/src/backbone_p14_m1536.cpp" "$output"
echo "IRON_HOST_BUILD=PASS output=$output"
