#!/usr/bin/env bash
set -euo pipefail

source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate mlir-air-build
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

ROOT=/home/amd/project/npu_iron
SRC=$ROOT/mlir-air/programming_examples/flash_attention/kernel_fusion_based
BUILD=$SRC/build_peano
OUT=$ROOT/sam3_attn/valid_query_flash_shapes_20260726
LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/valid_query_flash_shapes_20260726_build
QUAR=/home/amd/project/9_to_delete/valid_query_flash_stale_20260726

if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "Refusing to overwrite existing output/log: $OUT / $LOG" >&2
  exit 2
fi
mkdir -p "$OUT" "$LOG" "$QUAR" "$BUILD"

MLIR_AIE=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/mlir_aie
AIR=$ROOT/mlir-air/install
export PEANO_INSTALL_DIR=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/llvm-aie
export PYTHONPATH=$AIR/python:$MLIR_AIE/python:/opt/xilinx/xrt/python:${PYTHONPATH:-}
export PATH=$AIR/bin:$MLIR_AIE/bin:$PATH
export LD_LIBRARY_PATH=$MLIR_AIE/lib:$AIR/lib:/opt/xilinx/xrt/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

cd "$SRC"
make -f Makefile compile-kernel \
  LK=576 LQ=576 LKP=64 LQP=144 DK=64 DV=64 \
  NUM_Q_TILES=3 PEANO_INSTALL_DIR="$PEANO_INSTALL_DIR" \
  >"$LOG/compile_kernel_q48.log" 2>&1

quarantine_stale() {
  local tag=$1 path target
  for path in air.xclbin air.insts.bin air.elf air_project; do
    if [[ -e "$BUILD/$path" ]]; then
      target=$QUAR/${tag}_${path//\//_}
      if [[ -e "$target" ]]; then
        target=${target}_$(date +%H%M%S%N)
      fi
      mv "$BUILD/$path" "$target"
    fi
  done
}

generate() {
  local name=$1 lq=$2
  local target=$OUT/$name
  quarantine_stale "$name"
  (
    cd "$BUILD"
    python3 "$SRC/attn_npu2.py" \
      --lk 576 --lkp 64 --lq "$lq" --lqp 144 --dk 64 --dv 64 \
      --num-heads 16 --num-kv-heads 16 \
      --num-q-tiles 3 --num-cascade-stages 3 \
      --compile-mode compile-only --output-format xclbin \
      >"$LOG/$name.log" 2>&1
  )
  test -s "$BUILD/air.xclbin"
  test -s "$BUILD/air.insts.bin"
  test -d "$BUILD/air_project"
  mkdir -p "$target"
  mv "$BUILD/air.xclbin" "$target/final.xclbin"
  mv "$BUILD/air.insts.bin" "$target/insts.bin"
  mv "$BUILD/air_project" "$target/air_project"
  if [[ -e "$BUILD/air.elf" ]]; then mv "$BUILD/air.elf" "$target/final.elf"; fi
  printf 'LQ=%s LK=576 LQP=144 tile_q=48 heads=16\n' "$lq" >"$target/config.txt"
  sha256sum "$target/final.xclbin" "$target/insts.bin"
}

generate q576 576
generate q288 288
generate q144 144

echo VALID_QUERY_FLASH_SHAPES_DONE
