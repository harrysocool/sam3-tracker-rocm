# syntax=docker/dockerfile:1

ARG BASE_IMAGE=sam3-migraphx-builder:9f1a138-gfx1151
FROM ${BASE_IMAGE}

ARG TORCH_INDEX=https://repo.amd.com/rocm/whl/gfx1151/
ARG TORCH_VERSION=2.11.0+rocm7.13.0
ARG TORCHVISION_VERSION=0.26.0+rocm7.13.0
ARG TRITON_VERSION=3.6.0+rocm7.13.0

# The MIGraphX builder image contains the ROCm components needed by MIGraphX,
# but PyTorch links additional FFT, RNG and collective libraries. Install the
# ROCm 7.14 gfx1151 variants plus OpenCV's headless runtime dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        amdrocm-fft-host7.14 \
        amdrocm-fft7.14-gfx1151 \
        amdrocm-rand-host7.14 \
        amdrocm-rand7.14-gfx1151 \
        amdrocm-rccl-host7.14 \
        amdrocm-rccl7.14-gfx1151 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=migraphx /install/ /opt/migraphx-develop/
COPY --from=ort /wheels/onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl /tmp/onnxruntime_migraphx.whl
COPY requirements.txt /tmp/sam3-requirements.txt

RUN python3 -m venv --system-site-packages /opt/sam3-runtime \
    && /opt/sam3-runtime/bin/python -m pip install --no-cache-dir /tmp/onnxruntime_migraphx.whl \
    && /opt/sam3-runtime/bin/python -m pip install --no-cache-dir --no-deps \
        --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "triton==${TRITON_VERSION}" \
    && /opt/sam3-runtime/bin/python -m pip install --no-cache-dir \
        transformers==5.8.1 \
        safetensors==0.7.0 \
        opencv-python==4.11.0.86 \
    && /opt/sam3-runtime/bin/python -m pip install --no-cache-dir -r /tmp/sam3-requirements.txt \
    && rm /tmp/onnxruntime_migraphx.whl /tmp/sam3-requirements.txt

# The published gfx1151 Torch wheel currently targets ROCm 7.13. Its Python
# bootstrap would otherwise preload ROCm wheel libraries over the ROCm 7.14
# system installation. This compatibility module keeps the gfx1151 code
# objects while resolving the ABI-compatible runtime from /opt/rocm.
RUN mkdir -p /opt/sam3-runtime/lib/python3.12/site-packages/rocm_sdk
COPY docker/rocm714/rocm_sdk_system.py /opt/sam3-runtime/lib/python3.12/site-packages/rocm_sdk/__init__.py
COPY docker/rocm714/smoke_test.py /opt/sam3-tools/smoke_test.py

ENV PATH=/opt/sam3-runtime/bin:/opt/migraphx-develop/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/migraphx-develop/lib
ENV LD_LIBRARY_PATH=/opt/migraphx-develop/lib:/opt/migraphx-develop/lib/migraphx/lib:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/host-math/lib:/opt/rocm/core-7.14/lib/rocm_sysdeps/lib:/usr/local/lib
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV MIGRAPHX_GPU_HIP_FLAGS="-Wno-error -Wno-lifetime-safety-intra-tu-suggestions"

LABEL org.opencontainers.image.title="SAM3 ROCm 7.14 gfx1151 runtime" \
      org.opencontainers.image.description="PyTorch + ONNX Runtime MIGraphX + MIGraphX for SAM3 on gfx1151" \
      org.opencontainers.image.source="https://github.com/harrysocool/sam3-tracker-rocm"

CMD ["bash"]
