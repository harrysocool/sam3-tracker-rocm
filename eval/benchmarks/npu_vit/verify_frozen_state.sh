#!/usr/bin/env bash
set -euo pipefail

repo=${SAM3_TRACKER_ROOT:-/home/amd/project/sam3-tracker-rocm}
npu_root=${NPU_IRON_ROOT:-/home/amd/project/npu_iron}
release=${SAM3_IRON_RELEASE:-$npu_root/releases/sam3-vit-p14-m1536-power-v1}

cd "$repo"
sha256sum -c eval/benchmarks/npu_vit/FROZEN_ARTIFACTS.sha256
bash "$release/scripts/verify_release.sh" >/dev/null

python3 - <<'PY'
import json
from pathlib import Path

root = Path("eval/benchmarks/npu_vit/reference_results")
for route in ("iron", "flexml"):
    perf = json.loads((root / f"{route}_20260806.json").read_text())
    mask = json.loads((root / f"{route}_mask_20260806.json").read_text())
    assert perf["route"] == route
    assert mask["route"] == route
    assert mask["passed"] is True
    comparison = mask["comparison"]
    assert comparison["reference_objects"] > 0
    assert comparison["reference_objects"] == comparison["candidate_objects"]
    assert comparison["matched_objects"] == comparison["reference_objects"]
    assert comparison["mask_iou_min"] >= mask["thresholds"]["min_mask_iou"]
    print(
        f"FROZEN_RESULT route={route} "
        f"p50_ms={perf['latency']['p50_ms']:.3f} "
        f"mask_iou={comparison['mask_iou_min']:.6f} PASS"
    )
PY

echo "SAM3_VIT_FROZEN_STATE=PASS"
