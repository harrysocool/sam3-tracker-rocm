#!/usr/bin/env bash
set -euo pipefail

POWER_CONTROL=/sys/bus/pci/devices/0000:c6:00.1/power/control
REPO=/home/amd/project/sam3-tracker-rocm
BINARY=${NPU_BENCH_BINARY:-/home/amd/project/npu_iron/bh_validq_stallprof_20260727}
LOG=${NPU_BENCH_LOG:-/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/validq_stallprof_power_on_30f_20260727.log}

die() {
  echo "STALL_PM_COMPARISON=FAIL: $*" >&2
  exit 1
}

[[ -r "$POWER_CONTROL" && -x "$BINARY" ]] || die "missing power control or profiler binary"
if ps -eo stat,wchan:36,cmd | awk \
  '$1 ~ /^D/ && /amdxdna/ {found=1} END {exit found ? 0 : 1}'; then
  die "an amdxdna task is already in D-state"
fi

original=$(cat "$POWER_CONTROL")
restore_power_control() {
  echo "$original" | sudo tee "$POWER_CONTROL" >/dev/null || true
  echo "power_control_restored=$(cat "$POWER_CONTROL")"
}
trap restore_power_control EXIT

echo on | sudo tee "$POWER_CONTROL" >/dev/null
[[ $(cat "$POWER_CONTROL") == on ]] || die "failed to force runtime PM on"
echo "power_control=on"

source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate rocm7p13-sam3
cd "$REPO"

set +e
timeout --signal=TERM --kill-after=5s 120s \
  python eval/benchmarks/npu_iron/bench_npu_single_image.py \
    --image assets/truck.jpg --imgsz 504 --warmup 1 --runs 30 \
    --npu-bin "$BINARY" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "benchmark_rc=$rc"
[[ "$rc" == 0 ]] || die "benchmark failed or timed out"
if ps -eo stat,wchan:36,cmd | awk \
  '$1 ~ /^D/ && /amdxdna/ {found=1} END {exit found ? 0 : 1}'; then
  die "an amdxdna task entered D-state"
fi

echo "log=$LOG"
echo "STALL_PM_COMPARISON=PASS"
