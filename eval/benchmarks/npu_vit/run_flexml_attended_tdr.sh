#!/usr/bin/env bash
set -euo pipefail

repo=${SAM3_TRACKER_ROOT:-/home/amd/project/sam3-tracker-rocm}
recovery=${AMDXDNA_RECOVERY_ROOT:-/home/amd/project/amdxdna-recovery}
module=${AMDXDNA_RECOVERY_MODULE:-$recovery/dist/amdxdna-pmfix-validated.ko}
expected_module_sha=a302f61ed45f3bf6a053c967055d5b9d591d0ec96273caf208137a15d8a61124
target_tdr_ms=${FLEXML_TDR_TIMEOUT_MS:-10000}
normal_tdr_ms=2000
result=${FLEXML_BENCH_RESULT:-results/npu_vit_benchmark/flexml_20260806.json}
log=${FLEXML_BENCH_LOG:-results/npu_vit_benchmark/flexml_attended_20260806.log}
mask_result=${FLEXML_MASK_RESULT:-results/npu_vit_benchmark/flexml_mask_20260806.json}
mask_visuals=${FLEXML_MASK_VISUALS:-results/npu_vit_benchmark/masks/flexml_20260806}

die(){ echo "FLEXML_ATTENDED_BENCH=FAIL: $*" >&2; exit 1; }

[[ -t 0 ]] || die "run this script from an attended interactive terminal"
[[ $target_tdr_ms =~ ^[0-9]+$ && $target_tdr_ms -ge 5000 ]] || \
  die "FLEXML_TDR_TIMEOUT_MS must be an integer >= 5000"
[[ -f $module ]] || die "validated recovery module is missing: $module"
actual_sha=$(sha256sum "$module" | awk '{print $1}')
[[ $actual_sha == "$expected_module_sha" ]] || \
  die "recovery module SHA mismatch: $actual_sha"

dstate(){
  ps -eo stat,wchan:36,comm,args | awk \
    '$1 ~ /^D/ && /amdxdna/ {found=1} END {exit found ? 0 : 1}'
}

use_count(){
  awk '$1=="amdxdna"{print $3}' /proc/modules 2>/dev/null
}

require_idle(){
  dstate && die "an amdxdna task is in D-state; do not unload the driver"
  local count
  count=$(use_count)
  [[ -n $count && $count == 0 ]] || die "amdxdna use count is not zero: ${count:-missing}"
}

read_tdr(){ sudo cat /sys/module/amdxdna/parameters/tdr_timeout_ms; }
read_dump_only(){ sudo cat /sys/module/amdxdna/parameters/tdr_dump_only; }

xrt_check(){
  source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
  timeout 15s xrt-smi examine -r aie-partitions
}

load_profile(){
  local timeout_ms=$1
  require_idle
  sudo modprobe -r amdxdna
  if ! sudo insmod "$module" tdr_timeout_ms="$timeout_ms" tdr_dump_only=0; then
    echo "profile load failed; attempting to restore the installed module" >&2
    sudo modprobe amdxdna || true
    return 1
  fi
  [[ $(read_tdr) == "$timeout_ms" ]] || return 1
  case $(read_dump_only) in N|0) ;; *) return 1 ;; esac
  [[ $(use_count) == 0 ]] || return 1
  xrt_check >/dev/null
  echo "AMDXDNA_PROFILE=PASS tdr_timeout_ms=$timeout_ms"
}

wait_idle(){
  local count
  for _ in {1..10}; do
    count=$(use_count)
    [[ $count == 0 ]] && return 0
    sleep 1
  done
  echo "amdxdna did not become idle; use_count=${count:-missing}" >&2
  return 1
}

restore_profile(){
  echo "Restoring normal ${normal_tdr_ms}ms TDR profile..."
  if dstate; then
    echo "D-state detected; refusing module unload. Preserve the machine for attended recovery." >&2
    return 20
  fi
  wait_idle || return 21
  if [[ $(read_tdr 2>/dev/null || true) == "$normal_tdr_ms" ]]; then
    echo "AMDXDNA_RESTORE=PASS already_normal"
    return 0
  fi
  sudo modprobe -r amdxdna || return 22
  sudo insmod "$module" tdr_timeout_ms="$normal_tdr_ms" tdr_dump_only=0 || return 23
  [[ $(read_tdr) == "$normal_tdr_ms" ]] || return 24
  case $(read_dump_only) in N|0) ;; *) return 25 ;; esac
  xrt_check >/dev/null || return 26
  echo "AMDXDNA_RESTORE=PASS tdr_timeout_ms=$normal_tdr_ms"
}

on_exit(){
  local original_rc=$? restore_rc=0
  trap - EXIT INT TERM
  set +e
  restore_profile
  restore_rc=$?
  set -e
  if [[ $original_rc -ne 0 ]]; then exit "$original_rc"; fi
  exit "$restore_rc"
}

require_idle
xrt_check >/dev/null
mkdir -p "$repo/results/npu_vit_benchmark"
cd "$repo"

echo "The driver will be reloaded twice. Keep physical recovery available."
sudo -v
[[ $(read_tdr) == "$normal_tdr_ms" ]] || \
  die "expected initial tdr_timeout_ms=$normal_tdr_ms, got $(read_tdr)"

trap on_exit EXIT INT TERM
start_epoch=$(date +%s)
load_profile "$target_tdr_ms"

set -o pipefail
if [[ ${FLEXML_MASK_ONLY:-0} != 1 ]]; then
  bash eval/benchmarks/npu_vit/run_benchmark.sh flexml \
    --warmup 1 --runs 3 --output "$result" 2>&1 | tee "$log"
fi

bash eval/benchmarks/npu_vit/run_mask_validation.sh flexml \
  --prompt truck --output "$mask_result" --visual-dir "$mask_visuals" \
  2>&1 | tee -a "$log"

if journalctl -k -b --since "@$start_epoch" --no-pager 2>/dev/null | \
  grep -qE 'DRM scheduler timeout|aie2_sched_job_timedout'; then
  die "a scheduler timeout occurred during the attended flexml run"
fi

require_idle
xrt_check >/dev/null
echo "FLEXML_ATTENDED_BENCH=PASS result=$repo/$result mask=$repo/$mask_result log=$repo/$log"
