#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NPU_BENCH_BINARY=/home/amd/project/npu_iron/bh_validq_hostfused_20260727
export NPU_BENCH_LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/validq_hostfused_power_on_30f_20260727.log
exec "$SCRIPT_DIR/run_stall_pm_control_comparison.sh"
