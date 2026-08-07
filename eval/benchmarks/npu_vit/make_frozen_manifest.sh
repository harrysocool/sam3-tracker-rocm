#!/usr/bin/env bash
set -euo pipefail

repo=${SAM3_TRACKER_ROOT:-/home/amd/project/sam3-tracker-rocm}
npu_root=${NPU_IRON_ROOT:-/home/amd/project/npu_iron}
recovery=${AMDXDNA_RECOVERY_ROOT:-/home/amd/project/amdxdna-recovery}
release=${SAM3_IRON_RELEASE:-$npu_root/releases/sam3-vit-p14-m1536-power-v1}
manifest=$repo/eval/benchmarks/npu_vit/FROZEN_ARTIFACTS.sha256
sizes=$repo/eval/benchmarks/npu_vit/FROZEN_ARTIFACTS.tsv
tmp_manifest=/tmp/sam3-vit-xdna2-frozen-artifacts.sha256
tmp_sizes=/tmp/sam3-vit-xdna2-frozen-artifacts.tsv

cd "$repo"

files=(
  assets/truck.jpg
  model/sam3/LICENSE
  model/sam3/config.json
  model/sam3/configuration.json
  model/sam3/merges.txt
  model/sam3/processor_config.json
  model/sam3/special_tokens_map.json
  model/sam3/tokenizer.json
  model/sam3/tokenizer_config.json
  model/sam3/vocab.json
  /home/amd/project/3_model/sam3/model.safetensors
  onnx_files_504/backbone_detector/single_simplified.onnx
  npu_artifacts/voe_cache_504/backbone_detector_504_v1/backbone_detector_504_v1.rai
  eval/benchmarks/npu_vit/README.md
  eval/benchmarks/npu_vit/STATUS.md
  eval/benchmarks/npu_vit/benchmark_sam3_vit.py
  eval/benchmarks/npu_vit/build_iron_host.sh
  eval/benchmarks/npu_vit/compile_flexml_cache.py
  eval/benchmarks/npu_vit/compile_flexml_cache.sh
  eval/benchmarks/npu_vit/make_frozen_manifest.sh
  eval/benchmarks/npu_vit/run_benchmark.sh
  eval/benchmarks/npu_vit/run_flexml_attended_tdr.sh
  eval/benchmarks/npu_vit/run_mask_validation.sh
  eval/benchmarks/npu_vit/validate_sam3_mask.py
  eval/benchmarks/npu_vit/verify_benchmark_artifacts.sh
  eval/benchmarks/npu_vit/verify_frozen_state.sh
  eval/benchmarks/npu_vit/reference_results/flexml_20260806.json
  eval/benchmarks/npu_vit/reference_results/flexml_mask_20260806.json
  eval/benchmarks/npu_vit/reference_results/iron_20260806.json
  eval/benchmarks/npu_vit/reference_results/iron_mask_20260806.json
  eval/benchmarks/npu_vit/reference_results/masks/flexml_20260806/difference.png
  eval/benchmarks/npu_vit/reference_results/masks/iron_20260806/difference.png
  analysis/sam3_vit_npu_benchmark_closeout_20260806.md
  "$release/MANIFEST.sha256"
  "$release/WEIGHTS.sha256"
  "$release/bin/sam3-vit-p14-m1536"
  "$recovery/dist/amdxdna-pmfix-validated.ko"
)

for path in "${files[@]}"; do
  [[ -f $path ]] || { echo "missing frozen artifact: $path" >&2; exit 2; }
done

sha256sum "${files[@]}" >"$tmp_manifest"
{
  printf 'size_bytes\tpath\n'
  for path in "${files[@]}"; do
    printf '%s\t%s\n' "$(stat -c %s "$path")" "$path"
  done
} >"$tmp_sizes"

install -m 0644 "$tmp_manifest" "$manifest"
install -m 0644 "$tmp_sizes" "$sizes"

echo "manifest=$manifest"
echo "sizes=$sizes"
echo "entries=${#files[@]}"
echo "FROZEN_MANIFEST=PASS"
