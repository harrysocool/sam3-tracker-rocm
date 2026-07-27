#!/usr/bin/env bash
set -euo pipefail

source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate mlir-air-build
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

ROOT=/home/amd/project/npu_iron
SRC=$ROOT/mlir-air/programming_examples/matrix_multiplication/bf16
MLIR_AIE=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/mlir_aie
AIR=$ROOT/mlir-air/install
OUT=$ROOT/sam3_attn/compact_ffn_candidates_20260726
LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/compact_ffn_candidates_20260726_build
QUAR=/home/amd/project/9_to_delete/compact_ffn_generator_stale_20260726

if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "Refusing to overwrite existing output/log: $OUT / $LOG" >&2
  exit 2
fi
mkdir -p "$OUT" "$LOG" "$QUAR"

export PEANO_INSTALL_DIR=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/llvm-aie
export PYTHONPATH=$AIR/python:$MLIR_AIE/python:/opt/xilinx/xrt/python:${PYTHONPATH:-}
export PATH=$AIR/bin:$MLIR_AIE/bin:$PATH
export LD_LIBRARY_PATH=$MLIR_AIE/lib:$AIR/lib:/opt/xilinx/xrt/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}

cd "$SRC"

quarantine_stale() {
  local tag=$1 path target
  for path in air_project air.xclbin air.insts.bin; do
    if [[ -e "$path" ]]; then
      target=$QUAR/${tag}_${path//\//_}
      if [[ -e "$target" ]]; then
        target=${target}_$(date +%H%M%S%N)
      fi
      mv "$path" "$target"
    fi
  done
}

compile_kernel() {
  local tile_m=$1
  make -f Makefile compile-kernel \
    TILE_M="$tile_m" TILE_N=64 TILE_K_L1=64 \
    PEANO_INSTALL_DIR="$PEANO_INSTALL_DIR" AIE_TARGET=aie2p \
    >"$LOG/compile_kernel_m${tile_m}.log" 2>&1
}

generate() {
  local name=$1 m=$2 k=$3 n=$4 tile_m=$5 herd_m=$6
  local target=$OUT/$name
  quarantine_stale "$name"
  python3 run.py \
    --herd-m "$herd_m" --herd-n 4 \
    --m "$m" --k "$k" --n "$n" \
    --tile-m "$tile_m" --tile-k-l2 256 --tile-k-l1 64 --tile-n 64 \
    --output-dtype f32 --arch aie2p --direct-codegen \
    --compile-mode compile-and-xclbin \
    >"$LOG/$name.log" 2>&1
  test -s air.xclbin
  test -s air.insts.bin
  test -d air_project
  mkdir -p "$target"
  mv air.xclbin "$target/final.xclbin"
  mv air.insts.bin "$target/insts.bin"
  mv air_project "$target/air_project"
  printf '%s m=%s k=%s n=%s tile_m=%s herd=%sx4\n' \
    "$name" "$m" "$k" "$n" "$tile_m" "$herd_m" >"$target/config.txt"
  sha256sum "$target/final.xclbin" "$target/insts.bin"
}

compile_kernel 32
generate ffn1_m1536_h4864 1536 1024 4864 32 8
generate ffn2_m1536_h4864 1536 4864 1024 32 8

compile_kernel 16
generate ffn1_m1408_h4864 1408 1024 4864 16 8
generate ffn2_m1408_h4864 1408 4864 1024 16 8

compile_kernel 32
generate ffn1_m1344_h4864 1344 1024 4864 32 6
generate ffn2_m1344_h4864 1344 4864 1024 32 6

echo COMPACT_FFN_CANDIDATES_DONE
