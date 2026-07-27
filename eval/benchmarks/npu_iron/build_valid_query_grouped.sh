#!/usr/bin/env bash
set -euo pipefail

source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate mlir-air-build
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

ROOT=/home/amd/project/npu_iron
TRACKER=/home/amd/project/sam3-tracker-rocm
BASE=$ROOT/sam3_attn/valid_query_uniform64_base_20260727
OUT="${1:-$ROOT/sam3_attn/valid_query_flash_grouped_wip_20260727}"
LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/valid_query_grouped_rebuild_20260727.log
PATCHER=$TRACKER/eval/benchmarks/npu_iron/prune_valid_query_runtime.py

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 2
fi
[[ -f "$BASE/air_project/input_with_addresses.mlir" ]] || {
  echo "Missing uniform 64-head placed MLIR base" >&2
  exit 2
}
[[ -f "$BASE/attn_npu2.o" ]] || {
  echo "Missing validated q64 core object" >&2
  exit 2
}

mkdir -p "$OUT"
python3 "$PATCHER" "$BASE/air_project/input_with_addresses.mlir" "$OUT/patched.mlir"
cp "$BASE/attn_npu2.o" "$OUT/attn_npu2.o"

export PEANO_INSTALL_DIR=/home/amd/miniforge3/envs/mlir-air-build/lib/python3.12/site-packages/llvm-aie
(
  cd "$OUT"
  aiecc.py --alloc-scheme=basic-sequential \
    --aie-generate-xclbin --no-compile-host \
    --xclbin-name=final.xclbin \
    --no-xchesscc --no-xbridge --peano="$PEANO_INSTALL_DIR" \
    --aie-generate-npu-insts --npu-insts-name=insts.bin \
    patched.mlir >"$LOG" 2>&1
)

test -s "$OUT/final.xclbin"
test -s "$OUT/insts.bin"
sha256sum "$OUT/final.xclbin" "$OUT/insts.bin"
echo VALID_QUERY_GROUPED_BUILD_DONE
