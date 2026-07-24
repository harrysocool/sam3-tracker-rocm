#!/usr/bin/env bash
set -euo pipefail

source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate mlir-air-build
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

ROOT=/home/amd/project/npu_iron
TRACKER=/home/amd/project/sam3-tracker-rocm
PATCHER=$TRACKER/eval/benchmarks/npu_iron/patch_dynamic_k_rtp.py
CANDIDATE=$ROOT/sam3_attn/shared_gemm_candidate_m32n64
OUT="${1:-$ROOT/sam3_attn/shared_gemm_dynamic_rtp_complete}"
LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/shared_gemm_dynamic_rtp_build

if [ -e "$OUT" ]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT" "$LOG"

export PEANO_INSTALL_DIR=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/llvm-aie
export PATH=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/mlir_aie/bin:$PATH

gen() {
  local name=$1 k_tiles=$2
  local source=$CANDIDATE/$name/air_project/input_with_addresses.mlir
  local target=$OUT/$name
  mkdir -p "$target"
  python3 "$PATCHER" "$source" "$target/patched.mlir" --k-tiles "$k_tiles"
  (
    cd "$target"
    aiecc.py --alloc-scheme=basic-sequential \
      --aie-generate-xclbin --no-compile-host \
      --xclbin-name=final.xclbin \
      --no-xchesscc --no-xbridge --peano="$PEANO_INSTALL_DIR" \
      --aie-generate-npu-insts --npu-insts-name=insts.bin \
      patched.mlir >"$LOG/$name.log" 2>&1
  )
  test -s "$target/final.xclbin"
  test -s "$target/insts.bin"
  sha256sum "$target/final.xclbin" "$target/insts.bin"
}

gen qkv_w 4
gen qkv_g 4
gen o_w   4
gen o_g   4
gen ffn1  4
gen ffn2  20

echo SHARED_GEMM_DYNAMIC_RTP_DONE
