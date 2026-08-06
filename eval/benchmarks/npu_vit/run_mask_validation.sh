#!/usr/bin/env bash
set -euo pipefail

route=${1:?usage: run_mask_validation.sh flexml|iron [arguments...]}
shift

repo=${SAM3_TRACKER_ROOT:-/home/amd/project/sam3-tracker-rocm}
python=${SAM3_BENCH_PYTHON:-/home/amd/miniforge3/envs/rocm7p13-sam3/bin/python}
site=/home/amd/miniforge3/envs/rocm7p13-sam3/lib/python3.12/site-packages
script=$repo/eval/benchmarks/npu_vit/validate_sam3_mask.py

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$site/flexmlrt/lib:$site/voe/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"

if ps -eo stat=,comm=,wchan= | awk '$1 ~ /^D/ && ($2 ~ /amdxdna/ || $3 ~ /amdxdna/) {found=1} END {exit !found}'; then
  echo "refusing to start: amdxdna-related D-state task is present" >&2
  exit 3
fi

timeout 15s xrt-smi examine -r aie-partitions >/dev/null
cd "$repo"

case "$route" in
  flexml)
    tdr_ms=$(journalctl -k -b --no-pager 2>/dev/null | \
      sed -n 's/.*TDR enabled, timeout \([0-9][0-9]*\) ms.*/\1/p' | tail -n 1)
    if [[ -n $tdr_ms && $tdr_ms -lt 5000 ]]; then
      echo "flexml mask preflight failed: amdxdna TDR is ${tdr_ms}ms" >&2
      exit 4
    fi
    command=("$python" "$script" --route flexml "$@")
    ;;
  iron)
    command=("$python" "$script" --route iron "$@")
    ;;
  *)
    echo "unknown route: $route" >&2
    exit 2
    ;;
esac

"${command[@]}" || {
  rc=$?
  echo "SAM3_MASK_VALIDATION=FAIL route=$route rc=$rc" >&2
  exit "$rc"
}

timeout 15s xrt-smi examine -r aie-partitions >/dev/null
echo "SAM3_MASK_VALIDATION=PASS route=$route"
