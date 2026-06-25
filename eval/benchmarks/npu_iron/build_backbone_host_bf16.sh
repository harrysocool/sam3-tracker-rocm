#!/bin/bash
# Build + run the C++ SAM3 ViT backbone host (504px).
# Requires: weights exported to /tmp/cbb (export_weights_raw.py), XRT setup.
source /opt/xilinx/xrt/setup.sh
g++ -O3 -march=native -mavx512f -mavx512bf16 -ffast-math -funroll-loops -fopenmp -std=c++17 \
  backbone_host_cpp_20260617.cpp -o backbone_host \
  -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil
OMP_NUM_THREADS=16 ./backbone_host
