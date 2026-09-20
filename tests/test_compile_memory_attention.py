"""CPU-only guards for the memory-attention artifact compiler."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from export.tracker_modules import compile_memory_attention as compiler


def _write_graphs(root: Path, count: int = 10, ptr_tokens: int = 64) -> None:
    modules = root / "tracker_modules"
    modules.mkdir(parents=True)
    for slot in range(1, count + 1):
        (modules / f"memory_attention_fixed_S{slot}_P{ptr_tokens}.onnx").touch()


def test_slot_policy_forces_full_autotuning_and_generic_s8():
    for slot in range(1, 11):
        env = {
            "MIGRAPHX_SKIP_BENCHMARKING": "1",
            "MIGRAPHX_MLIR_USE_SPECIFIC_OPS": "stale",
        }
        policy = compiler.configure_worker_environment(slot, env)
        assert "MIGRAPHX_SKIP_BENCHMARKING" not in env
        if slot == 8:
            assert policy == ""
            assert "MIGRAPHX_MLIR_USE_SPECIFIC_OPS" not in env
        else:
            assert policy == "attention"
            assert env["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] == "attention"


def test_compile_all_uses_one_process_per_shape(monkeypatch, tmp_path):
    _write_graphs(tmp_path)
    calls = []

    def fake_run(command, *, check, env):
        assert check is True
        slot = int(command[command.index("--worker-slot") + 1])
        cache = compiler.cache_path(tmp_path)
        (cache / f"S{slot}.mxr").touch()
        calls.append((slot, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    compiler.compile_all(
        tmp_path, ptr_tokens=64, max_spatial_slots=10, force=True
    )

    assert [slot for slot, _env in calls] == list(range(1, 11))
    assert len(list(compiler.cache_path(tmp_path).glob("*.mxr"))) == 10
    assert (
        compiler.cache_path(tmp_path) / compiler.POLICY_FILENAME
    ).is_file()
    for slot, env in calls:
        assert "MIGRAPHX_SKIP_BENCHMARKING" not in env
        if slot == 8:
            assert "MIGRAPHX_MLIR_USE_SPECIFIC_OPS" not in env
        else:
            assert env["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] == "attention"

    # A resumable build trusts only a matching policy plus the recorded cache
    # file names/sizes and does not spawn workers again.
    compiler.compile_all(
        tmp_path, ptr_tokens=64, max_spatial_slots=10, force=False
    )
    assert len(calls) == 10


def test_compile_all_rejects_incomplete_onnx_coverage(tmp_path):
    _write_graphs(tmp_path, count=9)
    try:
        compiler.compile_all(
            tmp_path, ptr_tokens=64, max_spatial_slots=10, force=False
        )
    except FileNotFoundError as exc:
        assert "memory_attention_fixed_S10_P64.onnx" in str(exc)
    else:
        raise AssertionError("incomplete ONNX coverage was accepted")


def test_unverified_cache_is_preserved_without_force(tmp_path):
    _write_graphs(tmp_path)
    cache = compiler.cache_path(tmp_path)
    cache.mkdir(parents=True)
    existing = cache / "legacy.mxr"
    existing.write_bytes(b"do-not-delete")

    with pytest.raises(RuntimeError, match="refusing to delete"):
        compiler.compile_all(
            tmp_path, ptr_tokens=64, max_spatial_slots=10, force=False
        )

    assert existing.read_bytes() == b"do-not-delete"
