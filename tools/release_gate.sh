#!/usr/bin/env bash
# Run the complete source-release qualification flow and retain its evidence.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_IMAGE="sam3-gpu714-ort1242-mgx217-gfx1151:0.2.0-rc4-local"
CANONICAL_CHECKPOINT_SHA256="6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"
CANONICAL_VIDEO_SHA256="aaf37f0db4eba8d0058fd48b03391a742ded0d7f7db747378bec8459229476b9"
PYTEST_REQUIREMENT="pytest==9.0.3"
MIN_CACHED_FREE_GIB=15
MIN_NO_CACHE_FREE_GIB=30
MAX_CANONICAL_MEAN_MS=100.0
MAX_SOAK_MEAN_MS=100.0
MIN_MASK_MEAN_IOU=0.99
MIN_MASK_IOU=0.98

CHECKPOINT=""
OUTPUT=""
IMAGE="${SAM3_DOCKER_IMAGE:-${DEFAULT_IMAGE}}"
MGX_ARCHIVE=""
ORT_WHEEL=""
PREFLIGHT_ONLY=false
NO_CACHE=false
CURRENT_STAGE="argument parsing"
OUTPUT_READY=false

usage() {
    cat <<'EOF'
Usage:
  tools/release_gate.sh \
    --checkpoint DIR \
    --output NEW_DIR \
    [OPTIONS]

Required inputs:
  --checkpoint DIR       Canonical SAM3 model directory
  --output NEW_DIR       New evidence directory outside the Git checkout

Options:
  --image TAG                Runtime image tag (default: pinned rc4-local tag)
  --migraphx-archive FILE    Local MIGraphX release archive for runtime assembly
  --ort-wheel FILE           Local ONNX Runtime wheel for runtime assembly
  --no-cache                Force a full Docker image rebuild; default uses cache
  --preflight-only           Validate inputs without creating output or running work
  -h, --help

The gate is intentionally fixed. It requires:
  * clean dev or release/rcN source at the versioned commit;
  * EC Performance mode and 120/140/120 W power-policy attestations;
  * at least 15 GiB free with an existing image, or 30 GiB for a full rebuild;
  * a clean runtime/model build and strict full+hybrid smoke;
  * the complete pytest suite in the target runtime plus host wrapper tests;
  * manifest/SHA validation, PT-vs-MIG mask regression;
  * three canonical-250 runs and one soak-1000 run.

Box-prompt artifact generation and DAVIS validation are outside the rc1 gate.

It never changes EC/SMU settings, deletes an existing output directory, merges,
tags, or publishes anything. A failed run leaves logs and RELEASE_GATE_FAILED.
EOF
}

die() {
    printf 'release gate: %s\n' "$*" >&2
    exit 2
}

value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) value "$@"; CHECKPOINT="$2"; shift 2 ;;
        --output) value "$@"; OUTPUT="$2"; shift 2 ;;
        --image) value "$@"; IMAGE="$2"; shift 2 ;;
        --migraphx-archive) value "$@"; MGX_ARCHIVE="$2"; shift 2 ;;
        --ort-wheel) value "$@"; ORT_WHEEL="$2"; shift 2 ;;
        --no-cache) NO_CACHE=true; shift ;;
        --preflight-only) PREFLIGHT_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

for command in git docker python3 readlink sha256sum df awk tee; do
    command -v "${command}" >/dev/null 2>&1 || die "required command not found: ${command}"
done

[[ -n "${CHECKPOINT}" ]] || die "--checkpoint is required"
[[ -n "${OUTPUT}" ]] || die "--output is required"
[[ -d "${CHECKPOINT}" && -f "${CHECKPOINT}/model.safetensors" ]] || \
    die "--checkpoint must contain model.safetensors"
[[ -z "${MGX_ARCHIVE}" || -f "${MGX_ARCHIVE}" ]] || \
    die "--migraphx-archive must be a file"
[[ -z "${ORT_WHEEL}" || -f "${ORT_WHEEL}" ]] || \
    die "--ort-wheel must be a file"

CHECKPOINT="$(readlink -f -- "${CHECKPOINT}")"
OUTPUT="$(readlink -m -- "${OUTPUT}")"

case "${OUTPUT}" in
    "${ROOT}"|"${ROOT}"/*) die "--output must be outside the Git checkout" ;;
esac
[[ ! -e "${OUTPUT}" ]] || die "--output already exists; final gates require a new directory"

branch="$(git -C "${ROOT}" symbolic-ref --quiet --short HEAD)" || \
    die "detached HEAD is not allowed"
version="$(< "${ROOT}/VERSION")"
[[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$ ]] || \
    die "VERSION is not a release-candidate version: ${version}"
expected_release_branch="release/rc${version##*-rc}"
case "${branch}" in
    dev|"${expected_release_branch}") ;;
    *) die "run from dev or ${expected_release_branch}, got ${branch}" ;;
esac
[[ -f "${ROOT}/docs/releases/${version}.md" ]] || \
    die "missing release notes: docs/releases/${version}.md"
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]] || \
    die "source checkout is dirty"
git -C "${ROOT}" diff --check
git -C "${ROOT}" diff --check v0.2.0-rc6..HEAD

commit="$(git -C "${ROOT}" rev-parse HEAD)"
if [[ "${branch}" == dev ]]; then
    origin_dev="$(git -C "${ROOT}" rev-parse origin/dev 2>/dev/null || true)"
    [[ -n "${origin_dev}" && "${commit}" == "${origin_dev}" ]] || \
        die "dev must exactly match origin/dev before qualification"
fi
if git -C "${ROOT}" rev-parse --verify --quiet "refs/tags/v${version}" >/dev/null; then
    die "tag v${version} already exists"
fi

checkpoint_sha="$(sha256sum "${CHECKPOINT}/model.safetensors" | awk '{print $1}')"
[[ "${checkpoint_sha}" == "${CANONICAL_CHECKPOINT_SHA256}" ]] || \
    die "checkpoint SHA256 is not the canonical validation checkpoint"
video_sha="$(sha256sum "${ROOT}/assets/blackswan.mp4" | awk '{print $1}')"
[[ "${video_sha}" == "${CANONICAL_VIDEO_SHA256}" ]] || \
    die "assets/blackswan.mp4 SHA256 does not match the canonical input"

ec_source="SAM3_EC_POWER_MODE"
ec_mode="${SAM3_EC_POWER_MODE:-}"
if [[ -z "${ec_mode}" ]]; then
    ec_source="${SAM3_EC_POWER_MODE_PATH:-/sys/class/ec_su_axb35/apu/power_mode}"
    if [[ -r "${ec_source}" ]]; then
        ec_mode="$(< "${ec_source}")"
    fi
fi
ec_mode="${ec_mode,,}"
[[ "${ec_mode}" == performance ]] || \
    die "EC Performance mode is required; got ${ec_mode:-unknown} via ${ec_source}"

[[ -n "${SAM3_BUILD_HOST_ID:-}" ]] || \
    die "set SAM3_BUILD_HOST_ID to a non-sensitive machine label"
for specification in \
    "SAM3_STAPM_LIMIT_W:120" \
    "SAM3_FAST_PPT_LIMIT_W:140" \
    "SAM3_SLOW_PPT_LIMIT_W:120"; do
    name="${specification%%:*}"
    expected="${specification##*:}"
    actual="${!name:-}"
    [[ -n "${actual}" ]] || die "set ${name}=${expected} after verifying the power policy"
    python3 - "${name}" "${actual}" "${expected}" <<'PY'
import math
import sys

name, raw, expected = sys.argv[1:]
try:
    value = float(raw)
except ValueError as exc:
    raise SystemExit(f"release gate: {name} must be numeric, got {raw!r}") from exc
if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(
        f"release gate: {name} must be {expected} W, got {raw!r}"
    )
PY
done

[[ -e /dev/kfd && -d /dev/dri ]] || die "ROCm devices /dev/kfd and /dev/dri are required"
docker info >/dev/null
docker buildx version >/dev/null

if [[ "${NO_CACHE}" == true ]]; then
    min_free_gib=${MIN_NO_CACHE_FREE_GIB}
    runtime_build_mode="no-cache"
elif docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    min_free_gib=${MIN_CACHED_FREE_GIB}
    runtime_build_mode="cached"
else
    min_free_gib=${MIN_NO_CACHE_FREE_GIB}
    runtime_build_mode="cache-allowed-image-missing"
fi
required_kib=$((min_free_gib * 1024 * 1024))
check_free_space() {
    local label="$1"
    local path="$2"
    while [[ ! -e "${path}" ]]; do
        local parent
        parent="$(dirname -- "${path}")"
        [[ "${parent}" != "${path}" ]] || die "cannot locate filesystem for ${label}"
        path="${parent}"
    done
    local available
    available="$(df -Pk "${path}" | awk 'NR==2 {print $4}')"
    (( available >= required_kib )) || \
        die "at least ${min_free_gib} GiB free is required on ${label} (${path})"
    printf '%s' "$((available / 1024 / 1024))"
}
output_free_gib="$(check_free_space "the output filesystem" "$(dirname -- "${OUTPUT}")")"
docker_root="$(docker info --format '{{.DockerRootDir}}')"
docker_free_gib="$(check_free_space "the Docker filesystem" "${docker_root}")"

printf 'Preflight PASS: %s at %s (%s)\n' "${version}" "${commit}" "${branch}"
printf '  EC=%s; STAPM/Fast/Slow PPT=120/140/120 W\n' "${ec_mode}"
printf '  free space: output=%s GiB; Docker=%s GiB\n' \
    "${output_free_gib}" "${docker_free_gib}"
printf '  runtime build mode: %s\n' "${runtime_build_mode}"
if [[ "${PREFLIGHT_ONLY}" == true ]]; then
    exit 0
fi

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/canonical"
OUTPUT_READY=true
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

on_exit() {
    local status=$?
    if (( status != 0 )) && [[ "${OUTPUT_READY}" == true ]]; then
        {
            printf 'status=FAIL\n'
            printf 'stage=%s\n' "${CURRENT_STAGE}"
            printf 'exit_code=%s\n' "${status}"
            printf 'commit=%s\n' "${commit}"
            printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > "${OUTPUT}/RELEASE_GATE_FAILED"
    fi
}
trap on_exit EXIT

run_stage() {
    local name="$1"
    shift
    CURRENT_STAGE="${name}"
    printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}"
    set +e
    "$@" 2>&1 | tee "${OUTPUT}/logs/${name}.log"
    local statuses=("${PIPESTATUS[@]}")
    set -e
    (( statuses[0] == 0 )) || return "${statuses[0]}"
    return "${statuses[1]}"
}

assert_source_unchanged() {
    [[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${commit}" ]] || \
        die "source commit changed during the gate"
    [[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]] || \
        die "source checkout became dirty during the gate"
}

write_context() {
    local image_id="${1:-}"
    python3 - "${OUTPUT}/gate-context.json" "${version}" "${commit}" \
        "${branch}" "${IMAGE}" "${image_id}" "${checkpoint_sha}" \
        "${video_sha}" "${STARTED_AT}" "${runtime_build_mode}" <<'PY'
import json
import os
import sys

(path, version, commit, branch, image_ref, image_id, checkpoint_sha,
 video_sha, started_at, runtime_build_mode) = sys.argv[1:]
payload = {
    "schema": 1,
    "version": version,
    "source_commit": commit,
    "source_branch": branch,
    "source_dirty": False,
    "runtime_image_ref": image_ref,
    "runtime_image_id": image_id or None,
    "runtime_build_mode": runtime_build_mode,
    "checkpoint_sha256": checkpoint_sha,
    "canonical_video_sha256": video_sha,
    "build_host_id": os.environ["SAM3_BUILD_HOST_ID"],
    "power_policy": {
        "ec_mode": "performance",
        "stapm_limit_w": 120.0,
        "fast_ppt_limit_w": 140.0,
        "slow_ppt_limit_w": 120.0,
        "source": "environment_attestation",
    },
    "started_at": started_at,
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

write_context ""

build_output="${OUTPUT}/model-build"
runner=(
    "${ROOT}/tools/docker_test_runner.sh"
    --checkpoint "${CHECKPOINT}"
    --output "${build_output}"
    --image "${IMAGE}"
)
[[ -z "${MGX_ARCHIVE}" ]] || runner+=(--migraphx-archive "${MGX_ARCHIVE}")
[[ -z "${ORT_WHEEL}" ]] || runner+=(--ort-wheel "${ORT_WHEEL}")
[[ "${NO_CACHE}" != true ]] || runner+=(--no-cache)
run_stage clean-build "${runner[@]}"
assert_source_unchanged

artifact_root="${build_output}/onnx_files_504"
runtime=(env
    "SAM3_DOCKER_IMAGE=${IMAGE}"
    "SAM3_MODEL_DIR=${CHECKPOINT}"
    "SAM3_ONNX_DIR=${artifact_root}"
    "SAM3_OUTPUT_DIR=${OUTPUT}"
    "${ROOT}/docker/rocm714/run.sh"
)
image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
write_context "${image_id}"

run_container_unit_tests() {
    "${runtime[@]}" bash -lc '
set -euo pipefail
python -m pip install --disable-pip-version-check --no-cache-dir \
    --only-binary=:all: --target /tmp/sam3-pytest-deps pytest==9.0.3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="/tmp/sam3-pytest-deps:${PYTHONPATH}" \
    python -m pytest -q --ignore=tests/test_run_wrapper.py \
    --basetemp /output/pytest-container-tmp \
    -o cache_dir=/output/pytest-container-cache
'
}
run_stage unit-tests-container run_container_unit_tests

run_host_wrapper_tests() {
    local test_env
    local status
    test_env="$(mktemp -d)" || return
    python3 -m venv "${test_env}/venv" || {
        status=$?
        rm -rf -- "${test_env}"
        return "${status}"
    }
    "${test_env}/venv/bin/python" -m pip install \
        --disable-pip-version-check --no-cache-dir --only-binary=:all: \
        "${PYTEST_REQUIREMENT}" || {
        status=$?
        rm -rf -- "${test_env}"
        return "${status}"
    }
    status=0
    SAM3_TEST_IMAGE="${IMAGE}" "${test_env}/venv/bin/python" -m pytest -q \
        "${ROOT}/tests/test_run_wrapper.py" \
        --basetemp "${OUTPUT}/pytest-host-tmp" \
        -o "cache_dir=${OUTPUT}/pytest-host-cache" || status=$?
    rm -rf -- "${test_env}"
    return "${status}"
}
run_stage unit-tests-wrapper run_host_wrapper_tests
assert_source_unchanged

verify_manifest() {
    (
        cd "${artifact_root}"
        sha256sum --check ARTIFACT_MANIFEST.sha256 &&
            sha256sum --check SHA256SUMS
    ) || return
    python3 - "${artifact_root}/ARTIFACT_MANIFEST.json" "${commit}" \
        "${version}" "${checkpoint_sha}" "${image_id}" <<'PY'
import json
import sys

path, commit, version, checkpoint_sha, image_id = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    manifest = json.load(stream)
errors = []
if manifest.get("schema") != 2:
    errors.append("manifest schema is not 2")
source = manifest.get("source", {})
if source.get("commit") != commit or source.get("dirty") is not False:
    errors.append("source identity does not match the clean gate commit")
if source.get("version") != version:
    errors.append("source version mismatch")
if manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha:
    errors.append("checkpoint SHA256 mismatch")
build = manifest.get("build", {})
if build.get("image_id") != image_id:
    errors.append("runtime image ID mismatch")
if build.get("ec_power_mode") != "performance":
    errors.append("artifact build did not record EC Performance mode")
stack = manifest.get("stack", {})
if stack.get("onnxruntime") != "1.24.2":
    errors.append("artifact runtime did not use ONNX Runtime 1.24.2")
if not str(stack.get("migraphx", "")).startswith("2.17"):
    errors.append("artifact runtime did not use MIGraphX 2.17")
providers = stack.get("providers") or []
if not providers or providers[0] != "MIGraphXExecutionProvider":
    errors.append("MIGraphXExecutionProvider was not primary")
if stack.get("gpu_arch") != "gfx1151":
    errors.append(f"expected gfx1151, got {stack.get('gpu_arch')!r}")
if errors:
    raise SystemExit("manifest validation failed: " + "; ".join(errors))
print(f"manifest identity PASS: {len(manifest.get('files', []))} files")
PY
}
run_stage manifest-sha256 verify_manifest

run_stage mask-regression env SAM3_DOCKER_STRICT=1 "${runtime[@]}" \
    python eval/datasets/mask_diff_pt_vs_mig.py \
    --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
    --video assets/blackswan.mp4 --text swan --imgsz 504 --max-frames 30 \
    --parallel-tail --out /output/mask-regression.json

validate_mask() {
    python3 - "${OUTPUT}/mask-regression.json" <<PY
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
errors = []
if result.get("n_frames") != 30:
    errors.append(f"expected 30 frames, got {result.get('n_frames')!r}")
if result.get("iou_mean", -1) < ${MIN_MASK_MEAN_IOU}:
    errors.append(f"mean IoU {result.get('iou_mean')!r} < ${MIN_MASK_MEAN_IOU}")
if result.get("iou_min", -1) < ${MIN_MASK_IOU}:
    errors.append(f"minimum IoU {result.get('iou_min')!r} < ${MIN_MASK_IOU}")
memory = result.get("memory_attention", {})
if memory.get("pytorch_fallback_calls") != 0:
    errors.append(f"PyTorch fallback count is {memory.get('pytorch_fallback_calls')!r}")
if errors:
    raise SystemExit("mask regression failed: " + "; ".join(errors))
print(
    f"mask regression PASS: mean={result['iou_mean']:.6f}, "
    f"min={result['iou_min']:.6f}"
)
PY
}
run_stage mask-thresholds validate_mask

validate_canonical() {
    local result="$1"
    local profile="$2"
    local maximum="$3"
    python3 - "${result}" "${profile}" "${maximum}" "${commit}" <<'PY'
import json
import math
import sys

path, profile, maximum, commit = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    result = json.load(stream)
errors = list(result.get("validation", {}).get("errors", []))
if result.get("validation", {}).get("passed") is not True:
    errors.append("harness validation did not pass")
if result.get("config", {}).get("profile") != profile:
    errors.append("profile mismatch")
if result.get("config", {}).get("tracker_memory") != {
    "num_maskmem": 7, "max_cond_frame_num": 4
}:
    errors.append("tracker memory policy is not S7/C4")
if result.get("memory_attention", {}).get("pytorch_fallback_calls") != 0:
    errors.append("memory attention used PyTorch fallback")
if result.get("artifact", {}).get("source", {}).get("commit") != commit:
    errors.append("artifact source commit mismatch")
power = result.get("power_policy", {})
expected_power = {
    "stapm_limit_w": 120.0,
    "fast_ppt_limit_w": 140.0,
    "slow_ppt_limit_w": 120.0,
}
if power.get("complete") is not True:
    errors.append("power policy is incomplete")
for name, expected in expected_power.items():
    value = power.get(name)
    if value is None or not math.isclose(float(value), expected, abs_tol=1e-9):
        errors.append(f"{name} is {value!r}, expected {expected}")
mean = result.get("warm_service_ms", {}).get("mean")
if mean is None or float(mean) > float(maximum):
    errors.append(f"mean service latency {mean!r} exceeds {maximum} ms")
if errors:
    raise SystemExit(f"{profile} failed: " + "; ".join(dict.fromkeys(errors)))
print(f"{profile} PASS: mean service={float(mean):.3f} ms")
PY
}

for run in 1 2 3; do
    result="${OUTPUT}/canonical/250-r${run}.json"
    run_stage "canonical-250-r${run}" env SAM3_DOCKER_STRICT=1 "${runtime[@]}" \
        python eval/benchmarks/benchmark_latest_frame_canonical.py \
        --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
        --profile canonical-250 --out "/output/canonical/250-r${run}.json"
    run_stage "canonical-250-r${run}-threshold" \
        validate_canonical "${result}" canonical-250 "${MAX_CANONICAL_MEAN_MS}"
done

run_stage soak-1000 env SAM3_DOCKER_STRICT=1 "${runtime[@]}" \
    python eval/benchmarks/benchmark_latest_frame_canonical.py \
    --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
    --profile soak-1000 --out /output/canonical/1000.json
run_stage soak-1000-threshold validate_canonical \
    "${OUTPUT}/canonical/1000.json" soak-1000 "${MAX_SOAK_MEAN_MS}"

assert_source_unchanged

CURRENT_STAGE="summary"
python3 - "${OUTPUT}" "${version}" "${commit}" "${branch}" \
    "${image_id}" "${STARTED_AT}" "${runtime_build_mode}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
version, commit, branch, image_id, started_at, runtime_build_mode = sys.argv[2:]

def load(relative):
    with (root / relative).open(encoding="utf-8") as stream:
        return json.load(stream)

def digest(relative):
    value = hashlib.sha256()
    with (root / relative).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

mask = load("mask-regression.json")
canonical = []
for run in range(1, 4):
    relative = f"canonical/250-r{run}.json"
    result = load(relative)
    canonical.append({
        "run": run,
        "mean_service_ms": result["warm_service_ms"]["mean"],
        "p95_service_ms": result["warm_service_ms"]["p95"],
        "output_frames": result["output_frames"],
        "sha256": digest(relative),
    })
soak = load("canonical/1000.json")
manifest_relative = "model-build/onnx_files_504/ARTIFACT_MANIFEST.json"
summary = {
    "schema": 1,
    "status": "PASS",
    "version": version,
    "source_commit": commit,
    "source_branch": branch,
    "runtime_image_id": image_id,
    "runtime_build_mode": runtime_build_mode,
    "started_at": started_at,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "thresholds": {
        "canonical_mean_service_ms_max": 100.0,
        "soak_mean_service_ms_max": 100.0,
        "mask_mean_iou_min": 0.99,
        "mask_iou_min": 0.98,
    },
    "mask_regression": {
        "mean_iou": mask["iou_mean"],
        "minimum_iou": mask["iou_min"],
        "sha256": digest("mask-regression.json"),
    },
    "canonical_250": canonical,
    "soak_1000": {
        "mean_service_ms": soak["warm_service_ms"]["mean"],
        "p95_service_ms": soak["warm_service_ms"]["p95"],
        "output_frames": soak["output_frames"],
        "sha256": digest("canonical/1000.json"),
    },
    "artifact_manifest_sha256": digest(manifest_relative),
}
with (root / "RELEASE_GATE_PASS.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

rm -f -- "${OUTPUT}/RELEASE_GATE_FAILED"
trap - EXIT
printf '\nRELEASE GATE PASS: %s\nEvidence: %s\n' "${version}" "${OUTPUT}"
