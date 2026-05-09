# Manual Setup Guide

Step-by-step installation instructions for the SAM3 video tracker on AMD ROCm.
Most users should use [`setup.sh`](../setup.sh) instead — it automates all steps below.

---

#### 0. Install system ROCm 7.2 APT packages (MIGraphX)

```bash
# Add AMD ROCm 7.2 APT repository
sudo apt-get update
sudo apt-get install -y wget gnupg
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/7.2 noble main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Install MIGraphX
sudo apt-get update
sudo apt-get install -y migraphx
```

> **Important**: PyTorch for gfx1151 (ROCm 7.13) and `onnxruntime-migraphx`
> are **not on standard PyPI**. Install them from AMD's nightly wheel index
> and the GitHub release linked below. A plain `conda create` + the steps
> below is sufficient — no TheRock pre-built environment is required.

#### 0b. (Optional) Install patched MIGraphX for full performance

The headline FPS numbers (9.46 / 2.39 at 504 / 1008 px) require two
unreleased MIGraphX fixes (`find_splits` multi-arg + NHWC `offload_copy`).
We refer to the resulting build as **`MIGraphX 2.15+patches`**.

| Path | Performance | What you need |
|---|---|---|
| Stay on stock APT 2.15.0 | 5.72 / 1.35 FPS (504 / 1008 px) | Check out tag `v0.1-migraphx-2.15` |
| **Install prebuilt tarball** | **9.46 / 2.39 FPS** | ~2 min — download release asset, run install script |
| Build patched from source | 9.46 / 2.39 FPS | ~30 min — for non-`gfx1151` GPUs or different ROCm/Python |

Both prebuilt and source paths are documented in [`docs/build_migraphx_patched.md`](docs/build_migraphx_patched.md).
Patched source lives in the fork: [`harrysocool/AMDMIGraphX` branch `fix/offload-copy-contiguous-output`](https://github.com/harrysocool/AMDMIGraphX/tree/fix/offload-copy-contiguous-output) (both patches stacked).

### 1. Install ROCm SDK + PyTorch for gfx1151

AMD provides official nightly wheels for gfx1151 at:
**`https://rocm.nightlies.amd.com/v2/gfx1151/`**

```bash
# Create a fresh conda environment
conda create -n sam3-tracker python=3.12 -y
conda activate sam3-tracker

# Step 1a: Install ROCm runtime Python packages (pin to 20260411 for onnxruntime-migraphx compatibility)
pip install rocm "rocm-sdk-core==7.13.0a20260411" rocm-sdk-libraries-gfx1151 rocm-sdk-devel \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# Step 1b: Install PyTorch matching the same ROCm build date
pip install "torch==2.12.0a0+rocm7.13.0a20260411" \
            "torchvision==0.27.0a0+rocm7.13.0a20260411" \
            triton \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

> **Why pin to `20260411`?** `onnxruntime-migraphx 1.24.2` was compiled against the
> ROCm SDK from that date. Mismatched ROCm versions (even a few days apart) can cause
> MIGraphX kernel compilation to crash at runtime.

Set the following environment variables (add to `~/.bashrc` or your run script):

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
export MIGRAPHX_GPU_HIP_FLAGS="-Wno-error -Wno-lifetime-safety-intra-tu-suggestions"
```

> **BIOS tip (128 GB systems)**: set *UMA Frame Buffer Size* to **64 GB** in BIOS.
> This maximises the GPU's fast non-coherent memory pool. Setting it to 128 GB
> starves the OS and paradoxically reduces GPU bandwidth. See
> [`docs/project_summary.md`](docs/project_summary.md) Finding #7 for details.

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.12.0a0+rocm7.13.0a20260411  True
```

### 2. Install onnxruntime-migraphx

The MIGraphX-enabled ONNX Runtime is provided by
[Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build).

```bash
pip install https://github.com/Looong01/onnxruntime-rocm-build/releases/download/v1.24.2/onnxruntime_migraphx-1.24.2-cp312-cp312-manylinux_2_34_x86_64.whl
```

### 3. Install remaining dependencies

```bash
# Install additional packages needed by this project
pip install -r requirements.txt
```

### 4. Download SAM3 model weights

```bash
huggingface-cli download facebook/sam3 --local-dir model/sam3
```

### 5. Export ONNX tracking modules (~5 minutes)

```bash
# 504px — recommended (7.10 FPS, DAVIS J=81.1%)
python export/export_tracker_modules.py --imgsz 504 --output-dir onnx_files

# 1008px — higher quality (2.39 FPS, DAVIS J=85.8%)
python export/export_tracker_modules.py --imgsz 1008 --output-dir onnx_files_1008
```

> `--fixed-slots 7` (default) also exports `memory_attention_fixed_N7.onnx` with static shapes.
> The tracker automatically picks this file and runs it on MIGraphX.

### 5b. Export and compile MIGraphX backbone (~10 minutes first time)

```bash
# Export backbone ONNX (single-session, simplified)
# Then compile to .mxr with kernel autotuning — saved once, loaded in ~3s afterwards

# 504px backbone
python export/export_backbone_single.py --imgsz 504 --output-dir onnx_files
# Creates: onnx_files/backbone_mxr_tuned.mxr  (~896 MB, one-time compile ~3 min)

# 1008px backbone
python export/export_backbone_single.py --imgsz 1008 --output-dir onnx_files_1008
# Creates: onnx_files_1008/backbone_mxr_tuned.mxr  (~920 MB, one-time compile ~9 min)
```

> The `.mxr` cache encodes kernel-autotuned GPU programs. After first compile the backbone
> loads in ~3s on subsequent runs. Pass `--backbone pytorch` to fall back to PyTorch.
