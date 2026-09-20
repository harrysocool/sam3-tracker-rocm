#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./setup.sh --runtime [--no-smoke]
  ./setup.sh --models CHECKPOINT_DIR [BUILD_ARGS...]

--runtime downloads verified precompiled MIGraphX/ORT/Torch dependencies and
assembles the Docker image. It never compiles the runtime stack.

--models runs the model export/MXR compilation inside that image. The checkpoint
is a separate licensed input. Model outputs default to
results/build-<version>/onnx_files_504/; override SAM3_MODEL_BUILD_ROOT to use
another new or resumable build directory. This is a performance build and
requires EC performance mode to be visible in sysfs or declared through
SAM3_EC_POWER_MODE=performance after checking the BIOS.
EOF
}

case "${1:-}" in
    -h|--help|'') usage ;;
    --runtime)
        shift
        exec "${ROOT}/docker/rocm714/build.sh" "$@"
        ;;
    --models)
        [[ $# -ge 2 ]] || { echo "setup: --models requires CHECKPOINT_DIR" >&2; exit 2; }
        checkpoint="$2"
        shift 2
        version="$(< "${ROOT}/VERSION")"
        build_root="${SAM3_MODEL_BUILD_ROOT:-${ROOT}/results/build-${version}}"
        mkdir -p "${build_root}/onnx_files_504"
        SAM3_MODEL_DIR="${checkpoint}" SAM3_ONNX_DIR="${build_root}/onnx_files_504" \
            exec "${ROOT}/docker/rocm714/run.sh" python export/build_text_prompt_mig.py \
                --imgsz 504 --checkpoint /models/sam3 --onnx-root /models \
                --performance-build "$@"
        ;;
    *) echo "setup: unknown argument: $1" >&2; exit 2 ;;
esac
