# SAM3 ROCm Workspace Rules

## Active project

- The only long-lived active source checkout is on the remote development box:
  `/home/amd/project/sam3-tracker-rocm`.
- Its active development branch is `perf/text-fullmodel-gpu`.
- The local directory `/home/lsun/project/sam3` is a control workspace only. Run
  project commands over SSH as `amd@10.170.19.127`.
- Do not create another long-lived source checkout unless concurrent branch
  comparison is explicitly required. Use a temporary Git worktree when needed.

## Branch policy

- GPU text full-model optimization: `perf/text-fullmodel-gpu`.
- GPU baseline commit: `3c6804f` plus the portable-script fix `d3a1d4c`.
- NPU development is frozen and retained remotely only:
  - branch: `origin/perf/iron-sub1s-backbone`
  - tag: `sam3-vit-xdna2-benchmark-v1`
  - frozen commit: `d77b12f`
- Do not merge NPU runtime or dependency changes into the GPU optimization
  branch. Do not rewrite or move the frozen NPU tag.
- Recreate NPU source only on demand with a temporary worktree, for example:
  `git worktree add /tmp/sam3-npu-frozen sam3-vit-xdna2-benchmark-v1`.

## Environments

The NPU and GPU routes require mutually incompatible ONNX Runtime providers.
Keep their runtime selection isolated.

For GPU development:

```bash
source /home/amd/miniforge3/etc/profile.d/conda.sh
conda activate rocm7p13-sam3
source /home/amd/venvs/sam3-gpu/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTHONPATH=/home/amd/project/sam3-tracker-rocm:/opt/rocm-7.2.0/lib:${PYTHONPATH:-}
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0
```

Before a GPU benchmark, verify:

```bash
python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

The expected result is ONNX Runtime `1.24.2` with
`MIGraphXExecutionProvider`. A VitisAI-only result is the NPU environment and
must not be used for GPU measurements.

## Artifact ownership

Generated artifacts live outside the Git checkout:

```text
/home/amd/project/sam3-artifacts/
  gpu/
    onnx_files_504/       # canonical GPU ONNX and existing tuned MXR files
    mxr_cache/
    baselines/
  npu/
    npu_artifacts/        # frozen VitisAI/IRON-related local artifacts
    workspace-extras/
  shared/
    inputs/
```

The active checkout exposes `onnx_files_504/` through ignored symlinks to the
GPU artifact directory. Treat the existing `tuned.mxr` files as immutable
baselines. New GPU-I/O builds must use a distinct name such as
`tuned_gpuio.mxr`; never overwrite a baseline artifact while experimenting.

Model weights remain centralized at
`/home/amd/project/3_model/sam3/model.safetensors` and are linked into
`model/sam3/`.

## Text full-model optimization baseline

The canonical pre-optimization profile is:

`/home/amd/project/sam3-artifacts/gpu/baselines/text_fullmodel_gpu_baseline_20260825.json`

Workload: `assets/blackswan.mp4`, prompt `swan`, 504 px, 30 propagation
frames. Baseline mean latency is 185.0 ms (5.41 FPS), including:

- detector vision encoder: 101.6 ms
- memory attention: 25.7 ms
- DETR encoder: 13.4 ms
- DETR decoder: 12.4 ms
- tracker neck: 4.0 ms

Record every optimization against this workload and run correctness checks
before accepting performance gains. For changes that affect masks, run the
PT-vs-MIG mask regression; for tracker changes, retain the DAVIS regression.

## Repository hygiene

- Keep the active checkout on `perf/text-fullmodel-gpu` unless the task
  explicitly requires another branch.
- Keep benchmark outputs under ignored `results/perf/` or the external
  artifact tree.
- Check `git status --short --branch` before and after each change.
- Preserve unrelated user files and never delete frozen NPU artifacts during
  GPU work.
