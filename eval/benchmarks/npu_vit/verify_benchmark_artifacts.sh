#!/usr/bin/env bash
set -euo pipefail

tracker=${SAM3_TRACKER_ROOT:-/home/amd/project/sam3-tracker-rocm}
npu_root=${NPU_IRON_ROOT:-/home/amd/project/npu_iron}
release=${SAM3_IRON_RELEASE:-$npu_root/releases/sam3-vit-p14-m1536-power-v1}

flexml_onnx=$tracker/onnx_files_504/backbone_detector/single_simplified.onnx
flexml_rai=$tracker/npu_artifacts/voe_cache_504/backbone_detector_504_v1/backbone_detector_504_v1.rai

expected_onnx=f0e5f73d254d83b20421e4d40f16c9f1727a02126c99d1fd465d3caf67721003
expected_rai=0272945ed2254ac9b6196a1348600024a26023e5d4d340490ede42aea1f9b5b5
expected_iron=a53b5b83f77f6c87beadbd33bf52b669729e9d002ba9f361910aa8d13bc98f1a

check_one(){
  local expected=$1 path=$2 actual
  [[ -f $path ]] || { echo "missing: $path" >&2; exit 2; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ $actual == "$expected" ]] || {
    echo "checksum mismatch: $path expected=$expected actual=$actual" >&2
    exit 3
  }
  echo "OK $actual $path"
}

check_one "$expected_onnx" "$flexml_onnx"
check_one "$expected_rai" "$flexml_rai"
check_one "$expected_iron" "$release/bin/sam3-vit-p14-m1536"
bash "$release/scripts/verify_release.sh" >/dev/null
echo "SAM3_VIT_ARTIFACT_VERIFY=PASS"
