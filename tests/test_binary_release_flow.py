"""CPU-only guards for the download-binaries/build-models release flow."""

from pathlib import Path
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


def test_runtime_binary_hashes_are_pinned():
    script = (ROOT / "docker/rocm714/build.sh").read_text()
    assert "00c1823e540c33f0ce658f87ed0e1d71dda75c830b8be380accd8531b82f1624" in script
    assert "ef10e3e808e8805c26cc27f47572a53e385463f29d578e1ea2fe13d00e6f5ee0" in script


def test_model_build_enables_current_optimizations():
    source = (ROOT / "export/build_text_prompt_mig.py").read_text()
    assert 'sink_env["ROCMLIR_SINK_FINAL_ERF"] = "1"' in source
    assert "export_fixed_detr_decoder.py" in source
    assert "compile_fixed_detr_decoder.py" in source
    assert '"--onnx-dir", str(onnx_dir)' in source


def test_locally_compiled_decoder_uses_its_checksum_sidecar():
    compiler = (ROOT / "export/detector/compile_fixed_detr_decoder.py").read_text()
    runtime = (ROOT / "tracker/mig_detr_decoder.py").read_text()
    assert 'with_suffix(args.output.suffix + ".sha256")' in compiler
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
