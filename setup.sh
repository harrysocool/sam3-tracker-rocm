#!/usr/bin/env bash
# setup.sh — One-command setup for SAM3 video tracker on AMD ROCm / gfx1151
#
# Usage:
#   ./setup.sh                        # full setup
#   ./setup.sh --skip-apt             # skip ROCm 7.2 APT install (already done)
#   ./setup.sh --skip-migraphx        # skip patched MIGraphX install (already done)
#   ./setup.sh --env my-env           # custom conda environment name
#   ./setup.sh --imgsz 1008           # 1008px instead of 504px
#
# Environment variables (alternative to flags):
#   SAM3_CONDA_ENV=sam3-tracker       conda environment name
#   SAM3_IMGSZ=504                    backbone resolution (504 or 1008)
#   SAM3_MODEL_DIR=model/sam3         where to place model weights
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONDA_ENV="${SAM3_CONDA_ENV:-sam3-tracker}"
IMGSZ="${SAM3_IMGSZ:-504}"
MODEL_DIR="${SAM3_MODEL_DIR:-model/sam3}"
SKIP_APT=false
SKIP_MIGRAPHX=false
AUTO_YES=false

# Pinned package versions — update together if you change the ROCm stack
ROCM_SDK_VER="7.13.0a20260411"
TORCH_VER="2.12.0a0+rocm7.13.0a20260411"
TORCHVISION_VER="0.27.0a0+rocm7.13.0a20260411"
NIGHTLY_INDEX="https://rocm.nightlies.amd.com/v2/gfx1151/"
ORT_WHL="https://github.com/Looong01/onnxruntime-rocm-build/releases/download/v1.24.2/onnxruntime_migraphx-1.24.2-cp312-cp312-manylinux_2_34_x86_64.whl"

MXR_TAG="v2.15+patches.20260511"
MXR_ASSET="migraphx-2.15+patches-linux-x86_64-rocm7.2-py312.tar.gz"
MXR_URL="https://github.com/harrysocool/AMDMIGraphX/releases/download/${MXR_TAG}/${MXR_ASSET}"
MXR_SHA256="a45cf7b8208b3807ea46ab9c57d449ed1c4033f387970c3e1ead5225692c1900"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-apt)       SKIP_APT=true ;;
        --skip-migraphx)  SKIP_MIGRAPHX=true ;;
        --yes)            AUTO_YES=true ;;
        --env)            CONDA_ENV="$2"; shift ;;
        --imgsz)          IMGSZ="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Colours ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1;34m'; NC='\033[0m'
step()  { echo -e "\n${B}══ $* ${NC}"; }
info()  { echo -e "  ${G}✓${NC} $*"; }
warn()  { echo -e "  ${Y}⚠${NC}  $*"; }
die()   { echo -e "  ${R}✗${NC}  $*"; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo -e "${G}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  SAM3 Video Tracker — ROCm Setup                 ║"
echo "  ║  Target: gfx1151 (Ryzen AI Max+ 395)             ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Conda env : $CONDA_ENV"
echo "  Resolution: ${IMGSZ}px"
echo "  Model dir : $MODEL_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
step "0. Prerequisites"
# ─────────────────────────────────────────────────────────────────────────────
if command -v rocminfo &>/dev/null; then
    GPU=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
    info "GPU: $GPU"
    [[ "$GPU" == "gfx1151" ]] || warn "GPU is $GPU — tested on gfx1151 only. Proceeding anyway."
else
    warn "rocminfo not found — cannot verify GPU. Proceeding."
fi

# Locate conda
if ! command -v conda &>/dev/null; then
    for p in ~/miniforge3/bin/conda ~/miniconda3/bin/conda /opt/conda/bin/conda; do
        [[ -f "$p" ]] && eval "$($p shell.bash hook 2>/dev/null)" && break
    done
fi
command -v conda &>/dev/null || die "conda not found. Install miniforge: https://github.com/conda-forge/miniforge"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
info "conda: $(conda --version)"

# ─────────────────────────────────────────────────────────────────────────────
step "0a. ROCm 7.2 APT (stock MIGraphX)"
# ─────────────────────────────────────────────────────────────────────────────
if $SKIP_APT; then
    info "Skipping APT install (--skip-apt)"
elif dpkg -s migraphx &>/dev/null 2>&1; then
    info "migraphx already installed: $(dpkg -s migraphx | grep Version | awk '{print $2}')"
else
    echo "  Installing ROCm 7.2 APT packages (requires sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y wget gnupg
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
        https://repo.radeon.com/rocm/apt/7.2 noble main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt-get update -qq
    sudo apt-get install -y migraphx migraphx-dev
    info "migraphx installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "0b. Patched MIGraphX 2.15+patches"
# ─────────────────────────────────────────────────────────────────────────────
if $SKIP_MIGRAPHX; then
    info "Skipping patched MIGraphX install (--skip-migraphx)"
elif [[ -f /opt/rocm-7.2.0/lib/libmigraphx_c.so.3.0.2016000 ]]; then
    info "Patched MIGraphX already installed"
else
    echo "  Downloading patched MIGraphX (~85 MB)..."
    TMP_MXR=$(mktemp -d)
    wget -q --show-progress -O "$TMP_MXR/$MXR_ASSET" "$MXR_URL"
    echo "  Verifying checksum..."
    echo "${MXR_SHA256}  $TMP_MXR/$MXR_ASSET" | sha256sum --check --quiet
    echo "  Installing (requires sudo)..."
    tar -xzf "$TMP_MXR/$MXR_ASSET" -C "$TMP_MXR"
    cd "$TMP_MXR/migraphx-2.15+patches"
    sudo BUILD=. bash install_migraphx_patched.sh
    cd "$REPO_DIR"
    rm -rf "$TMP_MXR"
    info "Patched MIGraphX installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "1. Conda environment: $CONDA_ENV"
# ─────────────────────────────────────────────────────────────────────────────
if conda env list | grep -q "^${CONDA_ENV} "; then
    info "Conda env '$CONDA_ENV' already exists — activating"
else
    echo "  Creating conda env '$CONDA_ENV' (Python 3.12)..."
    conda create -n "$CONDA_ENV" python=3.12 -y -q
    info "Conda env created"
fi
conda activate "$CONDA_ENV"

# ─────────────────────────────────────────────────────────────────────────────
step "2. ROCm SDK + PyTorch (pinned $ROCM_SDK_VER)"
# ─────────────────────────────────────────────────────────────────────────────
if python -c "import torch; assert torch.__version__ == '${TORCH_VER}'" 2>/dev/null; then
    info "PyTorch ${TORCH_VER} already installed"
else
    echo "  Installing ROCm SDK + PyTorch (~2-5 min)..."
    pip install -q \
        rocm \
        "rocm-sdk-core==${ROCM_SDK_VER}" \
        rocm-sdk-libraries-gfx1151 \
        rocm-sdk-devel \
        --index-url "$NIGHTLY_INDEX"
    pip install -q \
        "torch==${TORCH_VER}" \
        "torchvision==${TORCHVISION_VER}" \
        triton \
        --index-url "$NIGHTLY_INDEX"
    info "PyTorch ${TORCH_VER} installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "3. onnxruntime-migraphx"
# ─────────────────────────────────────────────────────────────────────────────
if python -c "import onnxruntime; assert '1.24.2' in onnxruntime.__version__" 2>/dev/null; then
    info "onnxruntime-migraphx 1.24.2 already installed"
else
    echo "  Installing onnxruntime-migraphx 1.24.2..."
    pip install -q "$ORT_WHL"
    info "onnxruntime-migraphx installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "4. Python dependencies"
# ─────────────────────────────────────────────────────────────────────────────
pip install -q -r requirements.txt
info "requirements.txt installed"

# ─────────────────────────────────────────────────────────────────────────────
step "5. Model weights"
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_FILE="$MODEL_DIR/model.safetensors"

if [[ -f "$WEIGHT_FILE" ]]; then
    SIZE=$(du -sh "$WEIGHT_FILE" | cut -f1)
    info "Weights already present ($WEIGHT_FILE, $SIZE)"
else
    echo ""
    echo "  SAM3 model weights (~3.3 GB) are required."
    echo "  Config/tokenizer files are already included in this repo."
    echo ""
    echo "  Option A — Official (requires HuggingFace account + accepted terms):"
    echo "    https://huggingface.co/facebook/sam3"
    echo "    huggingface-cli download facebook/sam3 model.safetensors --local-dir $MODEL_DIR"
    echo ""
    echo "  Option B — Community mirror (no account needed, same weights):"
    echo "    huggingface-cli download 1038lab/sam3 sam3.safetensors --local-dir $MODEL_DIR"
    echo "    mv $MODEL_DIR/sam3.safetensors $MODEL_DIR/model.safetensors"
    echo ""
    read -rp "  Download via Option B now? [Y/n] " yn
    yn="${yn:-Y}"
    if [[ "$yn" =~ ^[Yy] ]]; then
        mkdir -p "$MODEL_DIR"
        huggingface-cli download 1038lab/sam3 sam3.safetensors \
            --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
        mv "$MODEL_DIR/sam3.safetensors" "$MODEL_DIR/model.safetensors"
        info "Weights downloaded → $WEIGHT_FILE"
    else
        warn "Skipping weights — place model.safetensors in $MODEL_DIR/ then re-run"
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
step "6. Export ONNX tracking modules"
# ─────────────────────────────────────────────────────────────────────────────
ONNX_DIR="onnx_files"
[[ "$IMGSZ" == "1008" ]] && ONNX_DIR="onnx_files_1008"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export MIGRAPHX_GPU_HIP_FLAGS="-Wno-error -Wno-lifetime-safety-intra-tu-suggestions"
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
export PYTHONPATH=/opt/rocm-7.2.0/lib

if [[ -f "$ONNX_DIR/memory_attention_fixed_N7.onnx" ]]; then
    info "ONNX modules already exported ($ONNX_DIR/)"
else
    echo "  Exporting ONNX tracking modules (${IMGSZ}px, ~5 min)..."
    python export/export_tracker_modules.py \
        --checkpoint "$MODEL_DIR" \
        --imgsz "$IMGSZ" \
        --output-dir "$ONNX_DIR"
    info "ONNX modules exported → $ONNX_DIR/"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "7. Compile MIGraphX backbone (~3 min 504px / ~9 min 1008px)"
# ─────────────────────────────────────────────────────────────────────────────
MXR_CACHE="$ONNX_DIR/backbone_mxr_tuned.mxr"

if [[ -f "$MXR_CACHE" ]]; then
    info "Backbone cache already present ($MXR_CACHE)"
else
    echo "  Compiling + autotuning backbone (runs once, then loads in ~3s)..."
    python export/export_backbone_single.py \
        --checkpoint "$MODEL_DIR" \
        --imgsz "$IMGSZ" \
        --output-dir "$ONNX_DIR"
    info "Backbone compiled → $MXR_CACHE"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "8. Pre-warm ORT MIGraphX caches (~1 min)"
# ─────────────────────────────────────────────────────────────────────────────
PREWARM_SENTINEL="$ONNX_DIR/mxr_cache/memory_attention_fp16.mxr"

if [[ -f "$PREWARM_SENTINEL" ]]; then
    info "ORT caches already warmed"
else
    echo "  Pre-warming tracking module caches..."
    python export/prewarm_ort_cache.py --onnx-dir "$ONNX_DIR"
    info "ORT caches warmed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "9. Smoke test"
# ─────────────────────────────────────────────────────────────────────────────
echo "  Running demo on sample image..."
python demo.py \
    --checkpoint "$MODEL_DIR" \
    --onnx-dir "$ONNX_DIR" \
    --image assets/demo.jpg \
    --output /tmp/sam3_smoke_test.jpg \
    --box 85,281,1710,850 \

info "Smoke test passed → /tmp/sam3_smoke_test.jpg"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${G}══════════════════════════════════════════════════════${NC}"
echo -e "${G}  Setup complete!${NC}"
echo ""
echo "  Activate and run:"
echo "    conda activate $CONDA_ENV"
echo "    export HSA_OVERRIDE_GFX_VERSION=11.5.1"
echo "    export MIGRAPHX_GPU_HIP_FLAGS=\"-Wno-error -Wno-lifetime-safety-intra-tu-suggestions\""
echo "    export PYTHONPATH=/opt/rocm-7.2.0/lib"
echo ""
echo "    python demo.py --checkpoint $MODEL_DIR --onnx-dir $ONNX_DIR \\"
echo "        --image YOUR_IMAGE.jpg --box x1,y1,x2,y2"
echo -e "${G}══════════════════════════════════════════════════════${NC}"
