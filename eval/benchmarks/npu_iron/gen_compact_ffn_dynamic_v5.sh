#!/usr/bin/env bash
set -euo pipefail

source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate mlir-air-build
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

ROOT=/home/amd/project/npu_iron
TRACKER=/home/amd/project/sam3-tracker-rocm
PATCHER=$TRACKER/eval/benchmarks/npu_iron/patch_dynamic_k_rtp.py
CANDIDATE=$ROOT/sam3_attn/compact_ffn_candidates_20260726
OUT=$ROOT/sam3_attn/compact_ffn_dynamic_rtp_v5
LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/compact_ffn_dynamic_rtp_v5_build

if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "Refusing to overwrite existing output/log: $OUT / $LOG" >&2
  exit 2
fi
mkdir -p "$OUT" "$LOG"

export PEANO_INSTALL_DIR=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/llvm-aie
export PATH=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/mlir_aie/bin:$PATH

generate() {
  local source_name=$1 output_name=$2 k_tiles=$3
  local source=$CANDIDATE/$source_name/air_project/input_with_addresses.mlir
  local target=$OUT/$output_name
  mkdir -p "$target"
  python3 "$PATCHER" "$source" "$target/patched.mlir" --k-tiles "$k_tiles"
  (
    cd "$target"
    aiecc.py --alloc-scheme=basic-sequential \
      --aie-generate-xclbin --no-compile-host \
      --xclbin-name=final.xclbin \
      --no-xchesscc --no-xbridge --peano="$PEANO_INSTALL_DIR" \
      --aie-generate-npu-insts --npu-insts-name=insts.bin \
      patched.mlir >"$LOG/$output_name.log" 2>&1
  )
  test -s "$target/final.xclbin"
  test -s "$target/insts.bin"
  sha256sum "$target/final.xclbin" "$target/insts.bin"
}

generate ffn1_m1536_h4864 ffn1 4
generate ffn2_m1536_h4864 ffn2 19

echo COMPACT_FFN_DYNAMIC_V5_DONE
