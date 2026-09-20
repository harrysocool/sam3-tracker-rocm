"""CPU-only guards for the download-binaries/build-models release flow."""

import hashlib
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
        (ROOT / "docker/rocm714/build.sh", ("--help",)),
        (ROOT / "tools/docker_test_runner.sh", ("--help",)),
    ):
        result = run(path, *args)
        assert result.returncode == 0, result.stderr
    assert "--runtime" in run(ROOT / "setup.sh").stdout
    assert "--models" in run(ROOT / "setup.sh").stdout
    assert "--resume" in run(ROOT / "tools/docker_test_runner.sh", "--help").stdout


def test_runtime_assembly_contains_no_source_build_commands():
    script = (ROOT / "docker/rocm714/build.sh").read_text()
    dockerfile = (ROOT / "docker/rocm714/Dockerfile").read_text()
    combined = script + dockerfile
    for forbidden in (
        "git clone", "cmake", "ninja", "tools/ci_build", "rbuild",
        "MIGRAPHX_REPO", "ORT_REPO",
    ):
        assert forbidden not in combined
    assert "--only-binary=:all:" in dockerfile
    assert "COPY --from=migraphx" in dockerfile
    assert "COPY --from=ort" in dockerfile
    assert 'docker/rocm714/build.sh" --no-cache' in (
        ROOT / "tools/docker_test_runner.sh"
    ).read_text()


def test_runtime_binary_locations_and_hashes_are_pinned():
    script = (ROOT / "docker/rocm714/build.sh").read_text()
    assert (
        "https://github.com/harrysocool/AMDMIGraphX/releases/download/"
        "v2.17.0%2Bsam3-fc1sink.20260908.1/${MGX_NAME}"
    ) in script
    mgx_sha = re.search(r"^MGX_SHA256=([0-9a-f]+)$", script, re.MULTILINE)
    assert mgx_sha is not None
    assert len(mgx_sha.group(1)) == 64
    assert mgx_sha.group(1) == (
        "ed1458c632eb2f0e2cab3c457aee93e39196cbb77d2180e47525e0009563dac1"
    )
    assert "ef10e3e808e8805c26cc27f47572a53e385463f29d578e1ea2fe13d00e6f5ee0" in script
    assert (
        "https://github.com/harrysocool/sam3-tracker-rocm/releases/download/"
        "v0.2.0-rc4/${ORT_NAME}"
    ) in script

    runtime_readme = (ROOT / "docker/rocm714/README.md").read_text()
    assert "v2.17.0%2Bsam3-fc1sink.20260908.1" in runtime_readme
    assert "sam3-gpu714-ort1242-mgx217-gfx1151:0.2.0-rc4-local" in runtime_readme
    assert "sam3-gpu714-ort1242-mgx217-gfx1151:torch211" not in runtime_readme


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
    source_version = "0.2.0-rc6"
    runtime_version = "0.2.0-rc4"
    assert (ROOT / "VERSION").read_text().strip() == source_version
    release_notes = ROOT / "docs/releases" / f"{source_version}.md"
    assert release_notes.is_file()
    assert f"SAM3 ROCm {source_version}" in release_notes.read_text()

    # rc5 reuses the checksum-pinned runtime bundle published for rc4. Keep
    # source-release metadata independent from the binary dependency version.
    for path in (
        ROOT / "docker/rocm714/build.sh",
        ROOT / "docker/rocm714/run.sh",
        ROOT / "docker/rocm714/README.md",
        ROOT / "tools/docker_test_runner.sh",
    ):
        assert runtime_version in path.read_text()


def test_model_build_enables_current_optimizations():
    source = (ROOT / "export/build_text_prompt_mig.py").read_text()
    assert 'sink_env["ROCMLIR_SINK_FINAL_ERF"] = "1"' in source
    assert "export_fixed_detr_decoder.py" in source
    assert "compile_fixed_detr_decoder.py" in source
    assert "compile_memory_attention.py" in source
    assert "write_artifact_manifest.py" in source
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

    assert "--performance-build" in (ROOT / "setup.sh").read_text()
    assert "--performance-build" in (
        ROOT / "tools/docker_test_runner.sh"
    ).read_text()


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
        ROOT / "setup.sh", ROOT / "docker/rocm714/build.sh",
        ROOT / "docker/rocm714/run.sh", ROOT / "tools/docker_test_runner.sh",
    ):
        assert subprocess.run([BASH, "-n", str(path)], check=False).returncode == 0
