#!/usr/bin/env bash
# Reproduce the original release flow with the current stack:
# precompiled runtime dependencies -> local model compilation -> demos/smoke.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=""
OUTPUT=""
IMAGE="sam3-gpu714-ort1242-mgx217-gfx1151:0.2.0-rc4-local"
FRAMES=12
MGX_ARCHIVE=""
ORT_WHEEL=""
RESUME=false

usage() {
    cat <<'EOF'
Usage: tools/docker_test_runner.sh --checkpoint DIR --output NEW_DIR [OPTIONS]

Options:
  --migraphx-archive FILE  Use a local release tar instead of downloading it
  --ort-wheel FILE         Use a local release wheel instead of downloading it
  --image TAG              Local assembled image tag
  --frames N               Final full/hybrid smoke frames (default: 12)
  --resume                 Reuse an existing output after an interrupted build
  -h, --help

The runner assembles Docker from released binaries, compiles SAM3 model graphs
locally, prewarms ORT caches, then runs an offline/read-only full+hybrid smoke.
It never compiles rocMLIR, MIGraphX, ONNX Runtime, or PyTorch.
EOF
}

die() { echo "clean-environment test: $*" >&2; exit 2; }
value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) value "$@"; CHECKPOINT="$2"; shift 2 ;;
        --output) value "$@"; OUTPUT="$2"; shift 2 ;;
        --migraphx-archive) value "$@"; MGX_ARCHIVE="$2"; shift 2 ;;
        --ort-wheel) value "$@"; ORT_WHEEL="$2"; shift 2 ;;
        --image) value "$@"; IMAGE="$2"; shift 2 ;;
        --frames) value "$@"; FRAMES="$2"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -d "${CHECKPOINT}" && -f "${CHECKPOINT}/model.safetensors" ]] || \
    die "--checkpoint must contain model.safetensors"
[[ -n "${OUTPUT}" ]] || die "--output is required"
if [[ -e "${OUTPUT}" && "${RESUME}" != true ]]; then
    die "--output already exists; pass --resume to continue its model build"
fi
[[ ! -e "${OUTPUT}" || -d "${OUTPUT}" ]] || die "--output must be a directory"
[[ "${FRAMES}" =~ ^[0-9]+$ && "${FRAMES}" -ge 3 ]] || die "--frames must be at least 3"

CHECKPOINT="$(readlink -f -- "${CHECKPOINT}")"
OUTPUT="$(readlink -m -- "${OUTPUT}")"
mkdir -p "${OUTPUT}/onnx_files_504"
exec > >(tee "${OUTPUT}/clean-build.log") 2>&1

build_env=("RUNTIME_IMAGE=${IMAGE}")
[[ -z "${MGX_ARCHIVE}" ]] || build_env+=("MIGRAPHX_ARCHIVE=$(readlink -f -- "${MGX_ARCHIVE}")")
[[ -z "${ORT_WHEEL}" ]] || build_env+=("ORT_WHEEL_PATH=$(readlink -f -- "${ORT_WHEEL}")")
env "${build_env[@]}" "${ROOT}/docker/rocm714/build.sh" --no-cache

runtime=(env
    "SAM3_DOCKER_IMAGE=${IMAGE}"
    "SAM3_MODEL_DIR=${CHECKPOINT}"
    "SAM3_ONNX_DIR=${OUTPUT}/onnx_files_504"
    "SAM3_OUTPUT_DIR=${OUTPUT}"
    "${ROOT}/docker/rocm714/run.sh"
)

"${runtime[@]}" python export/build_text_prompt_mig.py \
    --imgsz 504 --checkpoint /models/sam3 --onnx-root /models \
    --performance-build

# Re-run the short writable smoke as an independent lifecycle check. The model
# build has already populated DETR and memory caches before writing its manifest.
"${runtime[@]}" python tools/smoke_live_release.py \
    --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
    --video assets/blackswan.mp4 --text swan --frames 3 --mode full \
    --output /output/prewarm-smoke.json

# Refresh the manifest after writable prewarm so any generated ORT cache is
# included before the artifact root is tested read-only.
"${runtime[@]}" python export/write_artifact_manifest.py \
    --root /models/onnx_files_504 --checkpoint /models/sam3/model.safetensors \
    --imgsz 504 --ptr-tokens 64 --max-spatial-slots 10

SAM3_DOCKER_STRICT=1 "${runtime[@]}" python tools/smoke_live_release.py \
    --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
    --video assets/blackswan.mp4 --text swan --frames "${FRAMES}" --mode both \
    --output /output/installation-smoke.json

python3 - "${OUTPUT}/installation-smoke.json" <<'PY'
import json, sys
if not json.load(open(sys.argv[1], encoding="utf-8")).get("passed"):
    raise SystemExit("installation smoke did not pass")
PY
echo "PASS: ${OUTPUT}/installation-smoke.json"
