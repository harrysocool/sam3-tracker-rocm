"""CPU-only guards for the download-binaries/build-models release flow."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def run(path, *args):
    return subprocess.run(
        [BASH, str(path), *args], cwd=ROOT, text=True,
        capture_output=True, timeout=10,
    )


def test_help_entrypoints_do_not_start_work():
    for path, args in (
        (ROOT / "setup.sh", ()),
        (ROOT / "docker/rocm101/build.sh", ("--help",)),
        (ROOT / "tools/docker_test_runner.sh", ("--help",)),
        (ROOT / "tools/release_gate.sh", ("--help",)),
    ):
        result = run(path, *args)
        assert result.returncode == 0, result.stderr
    assert "--runtime" in run(ROOT / "setup.sh").stdout
    assert "--models" in run(ROOT / "setup.sh").stdout
    assert "--resume" in run(ROOT / "tools/docker_test_runner.sh", "--help").stdout
    assert "--reuse-runtime" in run(
        ROOT / "tools/docker_test_runner.sh", "--help"
    ).stdout
    assert "--reuse-runtime" in run(ROOT / "tools/release_gate.sh", "--help").stdout


def test_setup_model_build_selects_build_root_without_reusing_inference_path(tmp_path):
    setup_root = tmp_path / "source"
    runner = setup_root / "docker/rocm101/run.sh"
    runner.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "setup.sh", setup_root / "setup.sh")
    (setup_root / "VERSION").write_text("test-version\n")
    runner.write_text(
        '#!/bin/bash\n'
        'printf "%s\\n" "$SAM3_MODEL_DIR" "$SAM3_ONNX_DIR" "$@"\n'
    )
    runner.chmod(0o755)
    build_root = tmp_path / "new-build"
    result = subprocess.run(
        [BASH, str(setup_root / "setup.sh"), "--models", "/existing/checkpoint"],
        env=os.environ | {"SAM3_MODEL_BUILD_ROOT": str(build_root),
                          "SAM3_ONNX_DIR": "/existing/inference-artifacts"},
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["/existing/checkpoint", str(build_root / "onnx_files_504")]
    assert (build_root / "onnx_files_504").is_dir()
    assert lines[lines.index("--onnx-root") + 1] == "/models"
    assert "--imgsz" not in lines


def test_runtime_assembly_contains_no_source_build_commands():
    script = (ROOT / "docker/rocm101/build.sh").read_text()
    dockerfile = (ROOT / "docker/rocm101/Dockerfile").read_text()
    combined = script + dockerfile
    for forbidden in (
        "git clone", "cmake", "ninja", "tools/ci_build", "rbuild",
        "MIGRAPHX_REPO", "ORT_REPO", "harrysocool", "fc1sink", "migraphx-develop",
    ):
        assert forbidden not in combined
    assert "--only-binary=:all:" in dockerfile
    assert "COPY --from=wheels" in dockerfile
    runner = (ROOT / "tools/docker_test_runner.sh").read_text()
    assert 'build.sh" --no-cache' in runner
    assert "Reusing existing runtime image (--reuse-runtime)" in runner
    assert 'docker image inspect "${IMAGE}"' in runner
    assert 'torch.__version__ == "2.14.0+rocm10.1.0"' in runner
    assert "stack_errors" in runner


def test_runtime_wheels_come_from_official_indexes_and_are_hash_pinned():
    script = (ROOT / "docker/rocm101/build.sh").read_text()
    for base in (
        "https://stable.repo.amd.com/rocm/pytorch/whl-next",
        "https://stable.repo.amd.com/rocm/migraphx/whl-next",
        "https://stable.repo.amd.com/rocm/onnxruntime/whl-next",
    ):
        assert base in script
    rows = re.findall(r'^"([0-9a-f]{64}) \S+ (\S+\.whl)"$', script, re.MULTILINE)
    names = [name for _, name in rows]
    for required in (
        "torch-2.14.0+rocm10.1.0", "amd_torch_device_gfx1151", "amd_torch_device_gfx115x",
        "migraphx-2.18.0+rocm10.1.0", "migraphx_libs-2.18.0+rocm10.1.0",
        "onnxruntime_ep_migraphx-1.0.0+rocm10.1.0", "onnxruntime-1.29.0",
    ):
        assert any(name.startswith(required) for name in names), required
    runtime_readme = (ROOT / "docker/rocm101/README.md").read_text()
    assert "sam3-gpu101-ort129-mgx218-gfx1151:0.4.0-rc1-local" in runtime_readme


def test_repository_license_scope_is_explicit():
    license_path = ROOT / "LICENSE"
    assert hashlib.sha256(license_path.read_bytes()).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )
    assert "Hugging Face Transformers" in (ROOT / "NOTICE").read_text()
    readme = (ROOT / "README.md").read_text()
    assert "project-authored source code and documentation" in readme
    assert "not licensed under Apache-2.0" in readme
    assert "SAM License" in (ROOT / "model/sam3/LICENSE").read_text()


def test_source_release_and_runtime_dependency_versions_are_consistent():
    source_version = "0.4.0-rc1"
    runtime_version = "0.4.0-rc1"
    assert (ROOT / "VERSION").read_text().strip() == source_version
    release_notes = ROOT / "docs/releases" / f"{source_version}.md"
    assert release_notes.is_file()
    assert f"SAM3 ROCm {source_version}" in release_notes.read_text()

    # The runtime image tag carries the source release version.
    for path in (
        ROOT / "docker/rocm101/build.sh",
        ROOT / "docker/rocm101/run.sh",
        ROOT / "docker/rocm101/README.md",
        ROOT / "tools/docker_test_runner.sh",
    ):
        assert runtime_version in path.read_text()


def test_model_build_enables_current_optimizations():
    source = (ROOT / "export/build_text_prompt_mig.py").read_text()
    assert "ROCMLIR_SINK_FINAL_ERF" not in source
    assert "export_fixed_detr_decoder.py" in source
    assert "compile_fixed_detr_decoder.py" in source
    assert "compile_memory_attention.py" in source
    assert "write_artifact_manifest.py" in source
    assert "smoke_live_release.py" in source
    assert source.index("smoke_live_release.py") < source.index(
        '"[11/11] Write artifact manifest'
    )
    assert 'env.pop("MIGRAPHX_SKIP_BENCHMARKING", None)' in source
    assert '"--onnx-dir", str(onnx_dir)' in source

    memory_compiler = (
        ROOT / "export/tracker_modules/compile_memory_attention.py"
    ).read_text()
    assert "GENERIC_AUTOTUNE_SLOTS = frozenset({8})" in memory_compiler
    assert 'env.pop("MIGRAPHX_SKIP_BENCHMARKING", None)' in memory_compiler
    assert '"--worker-slot"' in memory_compiler

    manifest = (ROOT / "export/write_artifact_manifest.py").read_text()
    for field in (
        "SAM3_SOURCE_COMMIT", "SAM3_DOCKER_IMAGE_ID", "checkpoint",
        "ec_power_mode", "specific_ops_by_slot", "SHA256SUMS",
    ):
        assert field in manifest

    benchmark = (
        ROOT / "eval/benchmarks/benchmark_latest_frame_canonical.py"
    ).read_text()
    assert "ARTIFACT_MANIFEST.json" in benchmark
    assert "BENCH_NUM_MASKMEM" in benchmark
    assert "pytorch_fallback_calls" in benchmark
    assert '"canonical-250"' in benchmark
    assert '"soak-1000"' in benchmark
    assert "CANONICAL_VIDEO_SHA256" in benchmark
    assert "SAM3_BUILD_HOST_ID" in benchmark
    assert "SAM3_SLOW_PPT_LIMIT_W" in benchmark
    assert "insufficient measured outputs" in benchmark
    assert 'parser.add_argument("--video"' not in benchmark
    assert 'parser.add_argument("--loops"' not in benchmark

    assert "--performance-build" in (ROOT / "setup.sh").read_text()
    assert "--performance-build" in (
        ROOT / "tools/docker_test_runner.sh"
    ).read_text()


def test_release_gate_pins_the_complete_acceptance_contract():
    source = (ROOT / "tools/release_gate.sh").read_text()
    for required in (
        "MAX_CANONICAL_MEAN_MS=100.0",
        "MAX_SOAK_MEAN_MS=100.0",
        "MIN_MASK_MEAN_IOU=0.99",
        "MIN_MASK_IOU=0.98",
        "MIN_REUSE_FREE_GIB=15",
        "MIN_NO_CACHE_FREE_GIB=30",
        "docker_test_runner.sh",
        "mask_diff_pt_vs_mig.py",
        "canonical-250",
        "soak-1000",
        "ARTIFACT_MANIFEST.sha256",
        "SHA256SUMS",
        "RELEASE_GATE_PASS.json",
    ):
        assert required in source

    assert "eval_davis.py" not in source
    assert "--davis-root" not in source
    assert "--davis-onnx-dir" not in source
    assert "SAM3_STAPM_LIMIT_W:120" in source
    assert "SAM3_FAST_PPT_LIMIT_W:140" in source
    assert "SAM3_SLOW_PPT_LIMIT_W:120" in source
    assert 'runner+=(--reuse-runtime)' in source
    assert 'runtime_build_mode="existing-image"' in source
    assert 'runtime image not found: ${IMAGE}; omit --reuse-runtime' in source


def test_mask_regression_disables_memory_attention_fallback():
    source = (ROOT / "eval/datasets/mask_diff_pt_vs_mig.py").read_text()
    assert "required_spatial_slots=range(1, 11)" in source
    assert "allow_pytorch_fallback=False" in source
    assert '"pytorch_fallback_calls"' in source


def test_locally_compiled_decoder_uses_its_checksum_sidecar():
    compiler = (ROOT / "export/detector/compile_fixed_detr_decoder.py").read_text()
    runtime = (ROOT / "tracker/mig_detr_decoder.py").read_text()
    assert 'with_suffix(args.output.suffix + ".sha256")' in compiler
    assert 'os.environ.pop("MIGRAPHX_SKIP_BENCHMARKING", None)' in compiler
    assert 'os.environ.pop("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", None)' in compiler
    assert 'with_suffix(self.mxr_path.suffix + ".sha256")' in runtime


def test_legacy_host_installers_are_removed():
    assert not (ROOT / "environment.yml").exists()
    assert not (ROOT / "tools/install_migraphx_patched.sh").exists()


def test_shell_syntax():
    for path in (
        ROOT / "setup.sh", ROOT / "docker/rocm101/build.sh",
        ROOT / "docker/rocm101/run.sh", ROOT / "tools/docker_test_runner.sh",
        ROOT / "tools/release_gate.sh",
    ):
        assert subprocess.run([BASH, "-n", str(path)], check=False).returncode == 0
