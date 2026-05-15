# Installing / building patched MIGraphX (`2.15+patches`) for SAM3 tracker

The SAM3 tracker's headline FPS numbers (9.46 / 2.39 at 504 / 1008 px) require
two MIGraphX fixes that are not in any released version yet. There are three
ways to obtain them:

| Path | Time | Best for |
|---|---|---|
| [**A. Install prebuilt tarball**](#a-install-prebuilt-tarball-fast-path) (this section) | ~2 min | Same hardware/distro as us (`gfx1151`, ROCm 7.2 APT, Ubuntu 24.04, Python 3.12) |
| [B. Build from source](#b-build-from-source) | ~30 min | Different GPU arch / ROCm version / glibc / Python; or to verify the patches |
| Stay on stock APT 2.15.0 | 0 min | Acceptable to lose ~40% perf — see tag [`v0.1-migraphx-2.15`](https://github.com/harrysocool/sam3-tracker-rocm/releases/tag/v0.1-migraphx-2.15) (5.72 / 1.35 FPS) |

---

## A. Install prebuilt tarball (fast path)

Prebuilt artifacts are published as a release on the fork:
**[harrysocool/AMDMIGraphX → Releases](https://github.com/harrysocool/AMDMIGraphX/releases)**

```bash
# 1. Download (~85 MB, expanded ~412 MB)
URL=https://github.com/harrysocool/AMDMIGraphX/releases/download/v2.15%2Bpatches.20260509/migraphx-2.15+patches-linux-x86_64-rocm7.2-py312.tar.gz
wget -O /tmp/migraphx-patched.tar.gz "$URL"

# 2. Verify checksum (compare against the SHA256 in the release notes)
sha256sum /tmp/migraphx-patched.tar.gz

# 3. Extract
mkdir -p /tmp/migraphx-patched && tar -xzf /tmp/migraphx-patched.tar.gz -C /tmp/migraphx-patched

# 4. Install over stock /opt/rocm-7.2.x/lib (script back-ups the stock libs as *.bak)
cd /tmp/migraphx-patched/migraphx-2.15+patches
sudo BUILD=. bash install_migraphx_patched.sh
```

Then add to your `~/.bashrc` or run-script env:

```bash
export PYTHONPATH=/opt/rocm-7.2.x/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.x/lib/migraphx/lib:/opt/rocm-7.2.x/lib:$LD_LIBRARY_PATH
```

Verify:
```bash
python3 -c "import migraphx; print('MIGraphX from:', migraphx.__file__)"
# Expect: /opt/rocm-7.2.x/lib/migraphx.cpython-312-x86_64-linux-gnu.so
```

> **Compatibility caveat**: the tarball was built on Ubuntu 24.04 with ROCm 7.2 APT
> for `gfx1151` and Python 3.12. Different glibc / libstdc++ / GPU arch will not
> work — fall back to [Path B](#b-build-from-source) below.

To roll back to stock 2.15, see [Rollback](#rollback) below.

---

## What the two patches do

| # | Patch | Why it's needed |
|---|---|---|
| 1 | `simplify_algebra: extend find_splits to handle N-arg ops with multiple constants` ([upstream issue AMDMIGraphX#4256](https://github.com/ROCm/AMDMIGraphX/issues/4256)) | Without it, MIGraphX cannot fuse the 90 `Split` ops in HF window attention. HF backbone takes ~916 ms instead of ~88 ms. |
| 2 | `offload_copy: normalise non-standard GPU outputs to C-contiguous NCHW` (3 files: `offload_copy.cpp`, `lowering.cpp`, `mlir.cpp`) | Without it, GPU outputs in NHWC layout require a 10 ms (504px) / 94 ms (1008px) CPU transpose every frame. |

Both patches sit on top of the upstream `develop` branch (which CMake labels
`2.16.0`). Since 2.16.0 has not been released yet, we refer to the result as
**`MIGraphX 2.15+patches`**.

The patches are stacked on the branch `fix/offload-copy-contiguous-output` —
checking that branch out gets you both fixes.

---

## B. Build from source

### Prerequisites

| Item | Version / Notes |
|---|---|
| Stock ROCm 7.2 APT (`migraphx`, `migraphx-dev`) installed at `/opt/rocm-7.2.x/` | See README step 0 |
| Clang from ROCm | `/opt/rocm/llvm/bin/clang++` (comes with ROCm 7.2) |
| GPU target | `gfx1151` (or whatever your GPU reports — adjust `GPU_TARGETS`) |
| Disk | ~10 GB build space |
| Build time | ~30 min on a fast 16-core box |
| Python | 3.12 (matches the conda env in README step 1) |

### 1. Clone and check out the patched branch

```bash
git clone git@github.com:harrysocool/AMDMIGraphX.git
cd AMDMIGraphX
git checkout fix/offload-copy-contiguous-output  # contains both patches stacked
```

Optional sanity check:
```bash
git log --oneline upstream/develop..HEAD
# Expected: 4 commits on top of upstream/develop, ending in
#   e58acef offload_copy: normalise non-standard GPU outputs to C-contiguous NCHW
#   c3d4d38 simplify_algebra: extend find_splits to handle N-arg ops with multiple constants
```

(`upstream` is just a remote pointing at `https://github.com/ROCm/AMDMIGraphX` — `git remote add upstream …` if you don't have it yet.)

### 2. Install build dependencies

```bash
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

### 3. Configure with CMake

The exact configuration this project's binaries were built with:

```bash
cmake -B build_docker \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DBUILD_SHARED_LIBS=ON \
    -DGPU_TARGETS=gfx1151
```

Notes:
- The directory name `build_docker` is just a convention — name it anything.
  If you use a different name, also pass it to the install script in step 5.
- For other GPUs change `GPU_TARGETS` (e.g. `gfx1100`, `gfx942`).

### 4. Build

```bash
cmake --build build_docker -j$(nproc)
```

After the build, you should see (sizes approximate):

```
build_docker/lib/libmigraphx.so.2016000.0           (~250 MB)
build_docker/lib/libmigraphx_gpu.so.2016000.0       (~270 MB)
build_docker/lib/libmigraphx_device.so.2016000.0    (~3 MB)
build_docker/lib/libmigraphx_onnx.so.2016000.0
build_docker/lib/libmigraphx_tf.so.2016000.0
build_docker/lib/libmigraphx_c.so.3.0               (the C API)
build_docker/lib/migraphx.cpython-312-x86_64-linux-gnu.so   (Python binding)
```

### 5. Install over stock 2.15 (`/opt/rocm-7.2.x/lib`)

The install script back-ups the stock 2.15 libs (`*.bak`) and drops the
patched 2.16-pre libs in their place under `/opt/rocm-7.2.x/lib/`. After
install, the C API symlink `libmigraphx_c.so.3.0.70200` is repointed to the
new patched lib, so the system loader (and ORT MIGraphX EP) picks up the
patched code.

The script lives in this repo at [`tools/install_migraphx_patched.sh`](../tools/install_migraphx_patched.sh).

```bash
# Default: BUILD=./build_docker, ROCM=<auto-detected or /opt/rocm-7.2.x>
sudo bash tools/install_migraphx_patched.sh

# Override paths if needed:
sudo BUILD=/path/to/AMDMIGraphX/build_docker ROCM=/opt/rocm-7.2.x \
    bash tools/install_migraphx_patched.sh
```

Then make sure your run scripts have:

```bash
export PYTHONPATH=/opt/rocm-7.2.x/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.x/lib/migraphx/lib:/opt/rocm-7.2.x/lib:$LD_LIBRARY_PATH
```

### 6. Verify

```bash
python3 -c "import migraphx; print('MIGraphX from:', migraphx.__file__)"
# Expect: /opt/rocm-7.2.x/lib/migraphx.cpython-312-x86_64-linux-gnu.so
```

End-to-end check — run the project's prewarm script (it compiles the patched
backbone via `migraphx.parse_onnx` and the tracking modules via ORT MIGraphX EP):

```bash
cd <sam3-tracker-rocm>
python export/prewarm_ort_cache.py --onnx-dir onnx_files
# Expect: a fresh mxr_cache/ populates without errors; the per-module timings
# should match analysis/backbone_optimization.md.
```

---

## Rollback to stock 2.15

The install script saves `*.bak` copies. To revert:

```bash
cd /opt/rocm-7.2.x/lib
sudo cp libmigraphx_c.so.3.0.70200.bak libmigraphx_c.so.3.0.70200
cd migraphx/lib
for lib in libmigraphx libmigraphx_gpu libmigraphx_device libmigraphx_onnx libmigraphx_tf; do
    sudo cp "${lib}.so.2015000.0.70200.bak" "${lib}.so.2015000.0.70200"
done
sudo ldconfig
```

---

## Limitations / known issues

- **ORT MIGraphX EP**: the EP wheel from
  [Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build)
  was compiled against the MIGraphX 2.15 ABI. The C-API symlink redirect
  (step 5) makes it load and run successfully for SAM3's tracking modules, but
  some non-tested code paths in the EP may surface ABI quirks. If you hit
  unexpected EP behaviour with a different model, fall back to the stay-on-2.15
  baseline tag.
- **No CI for the patches**: they are only validated against SAM3 backbone +
  tracking modules on `gfx1151`. PRs to upstream are tracked in
  [AMDMIGraphX#4256](https://github.com/ROCm/AMDMIGraphX/issues/4256).
- **Build inside Docker also works**: the directory name `build_docker` is a
  legacy from the original Docker-based build. The instructions above produce
  identical binaries when run on a host with ROCm 7.2 installed.
