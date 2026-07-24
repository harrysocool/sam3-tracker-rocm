#!/usr/bin/env bash
set -euo pipefail

source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate mlir-air-build
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

ROOT=/home/amd/project/npu_iron
MLIR_AIE=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/mlir_aie
AIR=$ROOT/mlir-air/install
export PEANO_INSTALL_DIR=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/llvm-aie
export PYTHONPATH=$AIR/python:$MLIR_AIE/python:/opt/xilinx/xrt/python:${PYTHONPATH:-}
export PATH=$AIR/bin:$MLIR_AIE/bin:$PATH
export LD_LIBRARY_PATH=$MLIR_AIE/lib:$AIR/lib:/opt/xilinx/xrt/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}

SRC=$ROOT/mlir-air/programming_examples/matrix_multiplication/bf16
OUT=$ROOT/sam3_attn/shared_gemm_candidate_m32n64
LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/shared_gemm_build_20260724
mkdir -p "$OUT" "$LOG"
cd "$SRC"

make -f Makefile compile-kernel \
  TILE_M=32 TILE_N=64 TILE_K_L1=64 \
  PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR AIE_TARGET=aie2p \
  >"$LOG/compile_kernel.log" 2>&1

gen() {
  local name=$1 m=$2 k=$3 n=$4
  python3 run.py \
    --herd-m 8 --herd-n 4 \
    --m "$m" --k "$k" --n "$n" \
    --tile-m 32 --tile-k-l2 256 --tile-k-l1 64 --tile-n 64 \
    --output-dtype f32 --arch aie2p --direct-codegen \
    --compile-mode compile-and-xclbin \
    >"$LOG/$name.log" 2>&1
  test -s air.xclbin
  test -s air.insts.bin
  mkdir -p "$OUT/$name"
  mv air.xclbin "$OUT/$name/final.xclbin"
  mv air.insts.bin "$OUT/$name/insts.bin"
  sha256sum "$OUT/$name/final.xclbin" "$OUT/$name/insts.bin"
}

gen qkv_w 2304 1024 3072
gen qkv_g 1536 1024 3072
gen o_w   2304 1024 1024
gen o_g   1536 1024 1024
gen ffn1  1536 1024 5120
gen ffn2  1536 5120 1024

echo SHARED_GEMM_CANDIDATES_DONE
