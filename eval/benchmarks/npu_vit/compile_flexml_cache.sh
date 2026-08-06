#!/usr/bin/env bash
set -euo pipefail

repo=${SAM3_TRACKER_ROOT:-/home/amd/project/sam3-tracker-rocm}
python=${SAM3_BENCH_PYTHON:-/home/amd/miniforge3/envs/rocm7p13-sam3/bin/python}
site=/home/amd/miniforge3/envs/rocm7p13-sam3/lib/python3.12/site-packages

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$site/flexmlrt/lib:$site/voe/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"

cd "$repo"
"$python" eval/benchmarks/npu_vit/compile_flexml_cache.py "$@"
