#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BINARY=${PROJCOMPACT_BINARY:-/home/amd/project/npu_iron/bh_projcompact_20260726}
EXPECTED_SHA=${PROJCOMPACT_EXPECTED_SHA:-4d57c89b57faddd62c57a0467d5ac6b0a06b93989f105ed305e1eb839198d1f1}
PROBE=/home/amd/project/amdxdna-recovery/scripts/run_tdr_fault_probe.py
VERIFY=/home/amd/project/amdxdna-recovery/scripts/verify_recovery.sh
LOG=${PROJCOMPACT_LOG:-/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_1f_20260726.log}

die() {
  echo "PROJCOMPACT_GATE=FAIL: $*" >&2
  exit 1
}

[[ -x "$BINARY" ]] || die "candidate binary is missing"
actual_sha=$(sha256sum "$BINARY" | awk '{print $1}')
[[ "$actual_sha" == "$EXPECTED_SHA" ]] || die "binary SHA mismatch: $actual_sha"
[[ -x "$PROBE" ]] || die "single-frame probe is missing"
[[ -x "$VERIFY" ]] || die "driver verification script is missing"

PARAM_DIR=/sys/module/amdxdna/parameters
[[ -e "$PARAM_DIR/tdr_timeout_ms" ]] || die "TDR production candidate is not loaded"
[[ -e "$PARAM_DIR/tdr_dump_only" ]] || die "TDR recovery parameter is missing"
timeout_value=$(sudo cat "$PARAM_DIR/tdr_timeout_ms")
dump_only_value=$(sudo cat "$PARAM_DIR/tdr_dump_only")
[[ "$timeout_value" == "2000" ]] || die "unexpected TDR timeout: $timeout_value"
[[ "$dump_only_value" == "N" || "$dump_only_value" == "0" ]] || \
  die "TDR recovery mode is disabled"

usage_count=$(awk '$1 == "amdxdna" {print $3}' /proc/modules)
[[ "$usage_count" == "0" ]] || die "amdxdna is already in use ($usage_count)"
if ps -eo stat,wchan:36,cmd | awk \
  '$1 ~ /^D/ && /amdxdna/ {found=1} END {exit found ? 0 : 1}'; then
  die "an amdxdna task is already in D-state"
fi

python3 "$SCRIPT_DIR/test_projcompact_semantics.py" || die "CPU semantic proof failed"

set +e
python3 "$PROBE" --binary "$BINARY" --timeout 60 2>&1 | tee "$LOG"
probe_rc=${PIPESTATUS[0]}
set -e
echo "probe_rc=$probe_rc"

"$VERIFY" || die "driver recovery verification failed"
[[ "$probe_rc" == "0" ]] || die "candidate single-frame probe failed"

if ps -eo stat,wchan:36,cmd | awk \
  '$1 ~ /^D/ && /amdxdna/ {found=1} END {exit found ? 0 : 1}'; then
  die "an amdxdna task entered D-state"
fi

echo
echo "candidate_sha256=$actual_sha"
echo "log=$LOG"
echo "PROJCOMPACT_GATE=PASS"
