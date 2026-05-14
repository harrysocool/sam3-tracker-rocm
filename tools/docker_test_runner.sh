#!/usr/bin/env bash
# docker_test_runner.sh — Docker clean-environment new-user setup test
#
# Simulates a fresh Ubuntu 24.04 user cloning and running the full
# setup → build → demo pipeline. Used to catch regressions in setup.sh
# and export/build.py before releases.
#
# Usage:
#   ./tools/docker_test_runner.sh                    # full test
#   ./tools/docker_test_runner.sh --no-text          # skip text-prompt build (~30 min)
#   ./tools/docker_test_runner.sh --container NAME   # reuse existing container
#   ./tools/docker_test_runner.sh --clean            # remove container after test
#
# Requirements:
#   - Docker running on this machine
#   - GPU devices /dev/kfd and /dev/dri accessible
#   - Model weights available at MODEL_CACHE_DIR (to skip download)
#
# The script:
#   1. Starts a fresh ubuntu:24.04 container with GPU passthrough
#   2. Installs sudo (not in base image, unlike real Ubuntu desktop)
#   3. Runs the in-container bootstrap script which:
#      a. Installs miniforge
#      b. git clones the repo (dev branch)
#      c. Runs ./setup.sh
#      d. Runs export/build.py --pipeline box --imgsz 504
#      e. (optional) export/build.py --pipeline text --imgsz 504
#      f. Runs all 3 demos and verifies output
#   4. Reports pass/fail with timing for each step

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL="${SAM3_REPO_URL:-https://github.com/harrysocool/sam3-tracker-rocm.git}"
REPO_BRANCH="${SAM3_BRANCH:-dev}"
MODEL_CACHE_DIR="${SAM3_MODEL_CACHE:-/home/amd/project/sam3/model}"
CONTAINER_NAME="${DOCKER_CONTAINER:-sam3_clean_test}"
IMAGE="${DOCKER_IMAGE:-ubuntu:24.04}"
BUILD_TEXT=true
REMOVE_AFTER=false
LOG_DIR="${SAM3_LOG_DIR:-/tmp/sam3_docker_test_logs}"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-text)        BUILD_TEXT=false ;;
        --clean)          REMOVE_AFTER=true ;;
        --container)      CONTAINER_NAME="$2"; shift ;;
        --branch)         REPO_BRANCH="$2"; shift ;;
        --model-cache)    MODEL_CACHE_DIR="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
fail() { echo -e "${RED}  ✗ $*${NC}"; }

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/test_${TS}.log"
exec > >(tee "$LOG") 2>&1

echo "═══════════════════════════════════════════════════"
echo "  SAM3 Docker Clean-Environment Test"
echo "  Branch: $REPO_BRANCH  Image: $IMAGE"
echo "  $(date)"
echo "═══════════════════════════════════════════════════"
echo ""

T_TOTAL=$(date +%s)

# ── Step 0: GPU device IDs ────────────────────────────────────────────────────
VIDEO_GID=$(stat -c '%g' /dev/dri/card1 2>/dev/null || echo 44)
RENDER_GID=$(stat -c '%g' /dev/kfd 2>/dev/null || echo 992)

# ── Step 1: Start container ───────────────────────────────────────────────────
echo "[Step 1] Starting Docker container: $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --name "$CONTAINER_NAME" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add "$VIDEO_GID" \
    --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined \
    --network=host \
    -v "${MODEL_CACHE_DIR}:/model_cache:ro" \
    "$IMAGE" \
    bash -c 'sleep infinity'
ok "Container started"

# Install sudo (not in ubuntu base image, present on real desktop)
docker exec "$CONTAINER_NAME" bash -c 'apt-get update -qq && apt-get install -y sudo 2>/dev/null'
ok "sudo installed"

# ── Step 2: Write in-container bootstrap script ───────────────────────────────
echo ""
echo "[Step 2] Preparing bootstrap script"

BUILD_TEXT_FLAG=$( $BUILD_TEXT && echo "true" || echo "false" )

cat > /tmp/sam3_bootstrap.sh << BOOTSTRAP
#!/bin/bash
set -euo pipefail
LOG=/tmp/sam3_test.log
exec > >(tee -a \$LOG) 2>&1

ts() { echo "[\$(date '+%H:%M:%S')] \$*"; }
T0=\$(date +%s)
elapsed() { echo "\$(( \$(date +%s) - T0 ))s"; }
PASS=0; FAIL=0
check() {
    local label=\$1; shift
    if "\$@" >> /tmp/step_out.log 2>&1; then
        echo "  ✓ \$label"
        PASS=\$(( PASS + 1 ))
    else
        echo "  ✗ FAILED: \$label"
        tail -5 /tmp/step_out.log
        FAIL=\$(( FAIL + 1 ))
    fi
}

ts "=== SAM3 New-User Test (Branch: ${REPO_BRANCH}) ==="

# ── Prereq: miniforge ──────────────────────────────────────────────────────
ts "[1/6] Install miniforge"
apt-get update -qq && apt-get install -y -qq curl git wget 2>/dev/null
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
ts "  miniforge done (\$(elapsed))"

# ── Stage 1: clone + setup.sh ──────────────────────────────────────────────
ts "[2/6] Clone + setup.sh"
cd /workspace
git clone -b ${REPO_BRANCH} ${REPO_URL} sam3-tracker-rocm
cd sam3-tracker-rocm

# Symlink model weights from cache
mkdir -p model/sam3
for f in /model_cache/sam3/*; do
    fname=\$(basename \$f)
    [ ! -e "model/sam3/\$fname" ] && ln -sf "\$f" "model/sam3/\$fname"
done

./setup.sh
ts "  setup.sh done (\$(elapsed))"

# ── Stage 2: activate + build ─────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sam3-tracker

ts "[3/6] Build box-prompt @504"
check "build box" python export/build.py --pipeline box --imgsz 504
ts "  box build done (\$(elapsed))"

if [ "${BUILD_TEXT_FLAG}" = "true" ]; then
    ts "[4/6] Build text-prompt @504"
    check "build text" python export/build.py --pipeline text --imgsz 504
    ts "  text build done (\$(elapsed))"
else
    ts "[4/6] Skipping text-prompt build (--no-text)"
fi

# ── Stage 3: demos ────────────────────────────────────────────────────────
ts "[5/6] Demo: box-prompt"
check "demo box" python demo.py --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --image assets/truck.jpg --box 85,281,1710,850 --output /tmp/out_box.jpg

ts "[6/6] Demos: text-prompt"
check "demo text PT" python demo_text.py --checkpoint model/sam3 \
    --image assets/truck.jpg --text "truck" --output /tmp/out_text_pt.jpg

if [ "${BUILD_TEXT_FLAG}" = "true" ]; then
    export HSA_OVERRIDE_GFX_VERSION=11.5.1
    export PYTHONPATH=/opt/rocm-7.2.0/lib:\$PYTHONPATH
    check "demo text MIG" env LD_PRELOAD=/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0 \
        python demo_text.py --checkpoint model/sam3 \
        --onnx-dir onnx_files_504 --imgsz 504 --mig \
        --image assets/truck.jpg --text "truck" --output /tmp/out_text_mig.jpg
fi

TOTAL=\$(( \$(date +%s) - T0 ))
echo ""
echo "═══════════════════════════════════════════════════"
echo "  RESULT: PASS=\$PASS  FAIL=\$FAIL  TIME=\${TOTAL}s (\$(( TOTAL/60 ))m\$(( TOTAL%60 ))s)"
echo "═══════════════════════════════════════════════════"

[ \$FAIL -eq 0 ]
BOOTSTRAP

chmod +x /tmp/sam3_bootstrap.sh
docker cp /tmp/sam3_bootstrap.sh "$CONTAINER_NAME":/tmp/sam3_bootstrap.sh
docker exec "$CONTAINER_NAME" bash -c 'mkdir -p /workspace'
ok "Bootstrap script ready"

# ── Step 3: Run test ──────────────────────────────────────────────────────────
echo ""
echo "[Step 3] Running full test (this takes ~30-90 min)..."
echo "  Log: $LOG"
echo ""

if docker exec "$CONTAINER_NAME" bash /tmp/sam3_bootstrap.sh; then
    ok "ALL TESTS PASSED"
    RESULT=0
else
    fail "SOME TESTS FAILED"
    RESULT=1
fi

# ── Step 4: Collect outputs ───────────────────────────────────────────────────
echo ""
echo "[Step 4] Collecting outputs"
for f in out_box.jpg out_text_pt.jpg out_text_mig.jpg sam3_test.log; do
    docker cp "$CONTAINER_NAME:/tmp/$f" "$LOG_DIR/${TS}_${f}" 2>/dev/null && \
        ok "Saved: $LOG_DIR/${TS}_${f}" || \
        warn "Not found: $f (may be expected if build was skipped)"
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
if $REMOVE_AFTER; then
    docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
    ok "Container removed"
else
    warn "Container '$CONTAINER_NAME' still running (use --clean to remove)"
fi

TOTAL_S=$(( $(date +%s) - T_TOTAL ))
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Total wall time: ${TOTAL_S}s ($(( TOTAL_S/60 ))m$(( TOTAL_S%60 ))s)"
echo "  Full log: $LOG"
echo "═══════════════════════════════════════════════════"
exit $RESULT
