#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python3 "$SCRIPT_DIR/test_host_compaction_weights.py"

export PROJCOMPACT_BINARY=/home/amd/project/npu_iron/bh_projcompact_hostopt_20260726
export PROJCOMPACT_EXPECTED_SHA=36de8cc5e49b34707de0fb27148cb13380c28cbba6eb8c452a80269bf4d910de
export PROJCOMPACT_LOG=/home/amd/project/9_to_delete/git_cleanup_20260721/manifests/projcompact_hostopt_1f_20260726.log
exec "$SCRIPT_DIR/run_projcompact_gate.sh"
