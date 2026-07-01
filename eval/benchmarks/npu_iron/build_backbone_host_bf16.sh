#!/bin/bash
# Build the C++ SAM3 ViT BF16 backbone binary.
# Prerequisites:
#   - Stage B+C complete: /home/amd/project/npu_iron/weights/cbb/ exists
#   - Stage A complete:   /home/amd/project/npu_iron/sam3_attn/ exists
#   - XRT installed:      source /opt/xilinx/xrt/setup.sh
set -e
source /opt/xilinx/xrt/setup.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/amd/project/npu_iron/bh_npu_backbone_bf16

g++ -O3 -march=native -mavx512f -mavx512bf16 -ffast-math -funroll-loops -fopenmp -std=c++17 \
    "$SCRIPT_DIR/backbone_host_bf16_20260617.cpp" -o "$OUT" \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil

echo "Built: $OUT"
echo "Smoke test..."
OMP_NUM_THREADS=16 "$OUT"
