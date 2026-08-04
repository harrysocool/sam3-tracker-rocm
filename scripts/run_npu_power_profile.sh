#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 COMMAND [ARGS...]" >&2
  exit 2
fi

service=amdxdna-performance-mode.service
power_control=/sys/bus/pci/devices/0000:c6:00.1/power/control
default_bin=/home/amd/project/npu_iron/releases/sam3-vit-p14-m1536-power-v1/bin/sam3-vit-p14-m1536

check_dstate() {
  local found
  found=$(ps -eo stat,wchan:32,comm,args | awk '$1 ~ /^D/ && /amdxdna/ {print}')
  [[ -z $found ]] || {
    printf '%s\n' "$found" >&2
    echo "NPU_POWER_PROFILE=ABORT_DSTATE" >&2
    exit 3
  }
}

cleanup() {
  sudo systemctl stop "$service" || true
  echo -n "power_control_after_cleanup="
  cat "$power_control" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

check_dstate
[[ -x ${SAM3_NPU_BIN:-$default_bin} ]] || {
  echo "missing NPU binary: ${SAM3_NPU_BIN:-$default_bin}" >&2
  exit 2
}

sudo -v
sudo systemctl start "$service"
[[ $(systemctl is-active "$service") == active ]]
[[ $(cat "$power_control") == on ]]

export SAM3_NPU_BIN=${SAM3_NPU_BIN:-$default_bin}
export SAM3_NPU_OMP_THREADS=${SAM3_NPU_OMP_THREADS:-1}
echo "npu_bin=$SAM3_NPU_BIN"
echo "omp_threads=$SAM3_NPU_OMP_THREADS"
echo "power_control=$(cat "$power_control")"
