#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${SAM3_DOCKER_BUILD_ROOT:-${HOME}/.cache/sam3-rocm714-build}"
GPU_ARCH="${GPU_ARCH:-gfx1151}"
ROCM_VERSION="${ROCM_VERSION:-7.14}"
JOBS="${JOBS:-16}"

MIGRAPHX_REPO="${MIGRAPHX_REPO:-https://github.com/ROCm/AMDMIGraphX.git}"
MIGRAPHX_COMMIT="${MIGRAPHX_COMMIT:-9f1a138e77f4738d82a065d225836b3b337950ce}"
ORT_REPO="${ORT_REPO:-https://github.com/microsoft/onnxruntime.git}"
ORT_COMMIT="${ORT_COMMIT:-058787ceead760166e3c50a0a4cba8a833a6f53f}"

BUILDER_IMAGE="${BUILDER_IMAGE:-sam3-migraphx-builder:${MIGRAPHX_COMMIT:0:7}-${GPU_ARCH}}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-sam3-gpu714-ort1242-mgx217-gfx1151:torch211}"

MGX_SRC="${BUILD_ROOT}/AMDMIGraphX"
MGX_OUT="${BUILD_ROOT}/migraphx-out"
ORT_SRC="${BUILD_ROOT}/onnxruntime"
ORT_OUT="${BUILD_ROOT}/onnxruntime-out"

checkout_commit() {
    local repo="$1" dst="$2" commit="$3"
    if [[ ! -d "${dst}/.git" ]]; then
        git clone --filter=blob:none --no-checkout "${repo}" "${dst}"
    fi
    git -C "${dst}" fetch --depth 1 origin "${commit}"
    git -C "${dst}" checkout --detach FETCH_HEAD
}

gpu_args=()
if [[ -e /dev/kfd && -d /dev/dri ]]; then
    gpu_args+=(--device=/dev/kfd --device=/dev/dri)
    gpu_args+=(--group-add "$(stat -c '%g' /dev/kfd)")
    if [[ -e /dev/dri/renderD128 ]]; then
        gpu_args+=(--group-add "$(stat -c '%g' /dev/dri/renderD128)")
    fi
fi

mkdir -p "${BUILD_ROOT}" "${MGX_OUT}" "${ORT_OUT}/wheels"
checkout_commit "${MIGRAPHX_REPO}" "${MGX_SRC}" "${MIGRAPHX_COMMIT}"
checkout_commit "${ORT_REPO}" "${ORT_SRC}" "${ORT_COMMIT}"

DOCKER_BUILDKIT=1 docker build \
    --build-arg "ROCM_VERSION=${ROCM_VERSION}" \
    --build-arg "GPU_ARCH=${GPU_ARCH}" \
    -t "${BUILDER_IMAGE}" "${MGX_SRC}"

docker run --rm "${gpu_args[@]}" --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "${MGX_SRC}:/src:ro" -v "${MGX_OUT}:/work" -w /work \
    "${BUILDER_IMAGE}" bash -lc "
        cmake -S /src -B /work/build \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
          -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
          -DCMAKE_INSTALL_PREFIX=/work/install \
          -DCMAKE_PREFIX_PATH='/usr/local;/opt/rocm' \
          -DBUILD_SHARED_LIBS=ON \
          -DMIGRAPHX_ENABLE_MLIR=ON \
          -DMIGRAPHX_USE_HIPBLASLT=ON \
          -DGPU_TARGETS=${GPU_ARCH}
        cmake --build /work/build --target install -j${JOBS}
    "

docker run --rm "${gpu_args[@]}" --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e CC=/usr/bin/gcc -e CXX=/usr/bin/g++ \
    -e CMAKE_PREFIX_PATH=/mgx/install:/usr/local:/opt/rocm \
    -e LD_LIBRARY_PATH=/mgx/install/lib:/mgx/install/lib/migraphx/lib:/usr/local/lib:/opt/rocm/lib \
    -v "${ORT_SRC}:/src:ro" -v "${ORT_OUT}:/work" -v "${MGX_OUT}:/mgx:ro" -w /src \
    "${BUILDER_IMAGE}" python3 tools/ci_build/build.py \
      --build_dir /work/build --config Release --update --build \
      --skip_tests --skip_submodule_sync --build_wheel --parallel "${JOBS}" \
      --use_migraphx --migraphx_home /mgx/install \
      --compile_no_warning_as_error \
      --cmake_extra_defines \
        CMAKE_C_COMPILER=/usr/bin/gcc \
        CMAKE_CXX_COMPILER=/usr/bin/g++ \
        'CMAKE_PREFIX_PATH=/mgx/install;/usr/local;/opt/rocm' \
        FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
        "GPU_TARGETS=${GPU_ARCH}" \
        "CMAKE_HIP_ARCHITECTURES=${GPU_ARCH}" \
        'CMAKE_INSTALL_RPATH=/mgx/install/lib;/mgx/install/lib/migraphx/lib;/opt/rocm/lib'

cp -f "${ORT_OUT}"/build/Release/dist/onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl \
    "${ORT_OUT}/wheels/"

DOCKER_BUILDKIT=1 docker build \
    --build-arg "BASE_IMAGE=${BUILDER_IMAGE}" \
    --build-context "migraphx=${MGX_OUT}" \
    --build-context "ort=${ORT_OUT}" \
    -f "${ROOT}/docker/rocm714/Dockerfile" \
    -t "${RUNTIME_IMAGE}" "${ROOT}"

if [[ -e /dev/kfd && -d /dev/dri ]]; then
    docker run --rm "${gpu_args[@]}" --ipc=host \
        "${RUNTIME_IMAGE}" python /opt/sam3-tools/smoke_test.py
else
    echo "Built ${RUNTIME_IMAGE}; skipping GPU smoke test because /dev/kfd is unavailable."
fi
