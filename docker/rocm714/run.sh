#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SAM3_DOCKER_IMAGE:-sam3-gpu714-ort1242-mgx217-gfx1151:torch211}"
MODEL_DIR="${SAM3_MODEL_DIR:-${ROOT}/model/sam3}"
ONNX_DIR="${SAM3_ONNX_DIR:-${ROOT}/onnx_files_504}"

if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
    echo "ROCm devices /dev/kfd and /dev/dri are required" >&2
    exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Set SAM3_MODEL_DIR to an existing SAM3 model directory" >&2
    exit 1
fi
mkdir -p "${ONNX_DIR}"

gpu_args=(--device=/dev/kfd --device=/dev/dri --group-add "$(stat -c '%g' /dev/kfd)")
if [[ -e /dev/dri/renderD128 ]]; then
    gpu_args+=(--group-add "$(stat -c '%g' /dev/dri/renderD128)")
fi

mount_args=(
    -v "${ROOT}:/workspace"
    -v "${MODEL_DIR}:/models/sam3:ro"
    -v "${ONNX_DIR}:/models/onnx_files_504"
)

# Preserve the repository's supported external-artifact layout. Docker does
# not follow absolute symlinks inside a bind mount, so bind their resolved
# targets at the same absolute paths used by the links.
weight_link="${MODEL_DIR}/model.safetensors"
if [[ -L "${weight_link}" ]]; then
    raw_target="$(readlink "${weight_link}")"
    resolved_target="$(readlink -f "${weight_link}" 2>/dev/null || true)"
    if [[ "${raw_target}" = /* && -f "${resolved_target}" ]]; then
        mount_args+=(
            -v "$(dirname "${resolved_target}"):$(dirname "${raw_target}"):ro"
        )
    fi
fi
for subdir in backbone_detector backbone_tracker detector_modules tracker_modules; do
    link_path="${ONNX_DIR}/${subdir}"
    if [[ -L "${link_path}" ]]; then
        raw_target="$(readlink "${link_path}")"
        resolved_target="$(readlink -f "${link_path}" 2>/dev/null || true)"
        if [[ "${raw_target}" = /* && -d "${resolved_target}" ]]; then
            mount_args+=(-v "${resolved_target}:${raw_target}")
        fi
    fi
done

tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
    tty_args+=(-t)
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run --rm "${tty_args[@]}" --network host --ipc=host \
    "${gpu_args[@]}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e PYTHONPATH=/workspace:/opt/migraphx-develop/lib \
    "${mount_args[@]}" \
    -w /workspace \
    "${IMAGE}" "$@"
