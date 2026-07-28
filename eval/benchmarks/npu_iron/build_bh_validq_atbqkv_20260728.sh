#!/usr/bin/env bash
set -euo pipefail

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

repo=/home/amd/project/sam3-tracker-rocm
src=$repo/eval/benchmarks/npu_iron/bh_validq_atbqkv_20260728.cpp
out=/home/amd/project/npu_iron/bh_validq_atbqkv_20260728
block=/home/amd/project/npu_iron/mlir-aie-atb-20260727/programming_examples/ml/block_datatypes

g++ -O3 -march=native -mavx512f -mavx512bf16 \
  -ffast-math -funroll-loops -fopenmp -std=c++17 \
  "$src" -o "$out" \
  -I/opt/xilinx/xrt/include \
  -I"$block" \
  -I"$block/gemm_asymmetric_tile_buffering" \
  -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil

sha256sum "$src" "$out"
echo "ATB_QKV_BACKBONE_BUILD=PASS"
