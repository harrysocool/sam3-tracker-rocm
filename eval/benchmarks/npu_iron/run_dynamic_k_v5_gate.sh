#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARTIFACT_ROOT=/home/amd/project/npu_iron/sam3_attn/shared_gemm_dynamic_rtp_v5
COMPACT_ROOT=/home/amd/project/npu_iron/sam3_attn/compact_ffn_dynamic_rtp_v5
BINARY=/home/amd/project/npu_iron/shared_gemm_abi_test_v5
EXPECTED_SHA=296210a2226ad1378a4e75cf8b977a42e254ab6db72886be03622703e5268d2c
VERIFY=/home/amd/project/amdxdna-recovery/scripts/verify_recovery.sh
LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/dynamic_k_v5_gate.log

die() {
  echo "DYNAMIC_K_V5_GATE=FAIL: $*" >&2
  exit 1
}

[[ -x "$BINARY" ]] || die "ABI probe binary is missing"
actual_sha=$(sha256sum "$BINARY" | awk '{print $1}')
[[ "$actual_sha" == "$EXPECTED_SHA" ]] || die "ABI probe SHA mismatch: $actual_sha"
python3 "$SCRIPT_DIR/check_dynamic_k_artifacts.py" "$ARTIFACT_ROOT" || \
  die "static artifact check failed"

PARAM_DIR=/sys/module/amdxdna/parameters
[[ -e "$PARAM_DIR/tdr_timeout_ms" && -e "$PARAM_DIR/tdr_dump_only" ]] || \
  die "load the temporary production TDR module before this gate"
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

set +e
timeout --signal=TERM --kill-after=5s 120s "$BINARY" \
  --artifact-root "$ARTIFACT_ROOT" --compact-root "$COMPACT_ROOT" \
  2>&1 | tee "$LOG"
probe_rc=${PIPESTATUS[0]}
set -e
echo "probe_rc=$probe_rc"

"$VERIFY" || die "driver recovery verification failed"
[[ "$probe_rc" == "0" ]] || die "dynamic-K boundary sequence failed or timed out"

if ps -eo stat,wchan:36,cmd | awk \
  '$1 ~ /^D/ && /amdxdna/ {found=1} END {exit found ? 0 : 1}'; then
  die "an amdxdna task entered D-state"
fi

echo
echo "artifact_root=$ARTIFACT_ROOT"
echo "probe_sha256=$actual_sha"
echo "log=$LOG"
echo "DYNAMIC_K_V5_GATE=PASS"
