"""CPU-only guards for the ROCm 10.1 plugin-EP session factory and stack gates."""
from pathlib import Path
import os
import re
import subprocess
import sys
import types

import pytest

from tracker import ort_plugin_ep as ep

ROOT = Path(__file__).resolve().parents[1]
GOOD = {
    "ep": "plugin", "onnxruntime": "1.29.0",
    "migraphx": "2.18.0+rocm10.1.0", "ep_package": "1.0.0+rocm10.1.0",
}


def test_supported_stack_is_accepted():
    assert ep.stack_errors(GOOD) == []


@pytest.mark.parametrize("override", [
    {"migraphx": "2.17.0.dev+2e9924db6"},
    {"migraphx": None},
    {"onnxruntime": "1.24.2"},
    {"ep_package": None},
    {"ep_package": "0.9.0"},
])
def test_unsupported_stack_is_rejected(override):
    assert ep.stack_errors({**GOOD, **override})


class _Options:
    def __init__(self):
        self.entries, self.provider = {}, None

    def add_session_config_entry(self, key, value):
        self.entries[key] = value

    def add_provider_for_devices(self, devices, options):
        self.provider = (devices, options)


class _Assignment:
    def __init__(self, name, nodes):
        self.ep_name, self._nodes = name, nodes

    def get_nodes(self):
        return self._nodes


def _fake_session(monkeypatch, assignment):
    session = types.SimpleNamespace(get_provider_graph_assignment_info=lambda: assignment)
    monkeypatch.setattr(ep, "_devices", lambda: ["device0"])
    monkeypatch.setattr(ep.ort, "InferenceSession", lambda path, sess_options: session)
    return session


def test_plugin_session_uses_new_option_keys_and_is_tagged(monkeypatch, tmp_path):
    session = _fake_session(monkeypatch, [_Assignment("MIGraphXExecutionProvider", [1, 2])])
    options = _Options()
    result = ep.create_mig_session("m.onnx", options, tmp_path, fp16=True)
    assert result is session and ep.is_plugin_session(result)
    devices, provider_options = options.provider
    assert devices == ["device0"]
    assert provider_options == {"cache_dir": str(tmp_path), "migraphx_fp16_enable": "1"}
    assert options.entries["session.record_ep_graph_assignment_info"] == "1"


def test_fp32_session_omits_fp16_option(monkeypatch, tmp_path):
    _fake_session(monkeypatch, [_Assignment("MIGraphXExecutionProvider", [1])])
    options = _Options()
    ep.create_mig_session("m.onnx", options, tmp_path, fp16=False)
    assert options.provider[1] == {"cache_dir": str(tmp_path)}


@pytest.mark.parametrize("assignment", [
    [],
    [_Assignment("CPUExecutionProvider", [1])],
    [_Assignment("MIGraphXExecutionProvider", [1]), _Assignment("CPUExecutionProvider", [2])],
])
def test_any_cpu_node_raises_instead_of_falling_back(monkeypatch, tmp_path, assignment):
    _fake_session(monkeypatch, assignment)
    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        ep.create_mig_session("m.onnx", _Options(), tmp_path)


def test_no_device_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ep, "_devices", lambda: [])
    with pytest.raises(RuntimeError, match="no device"):
        ep.create_mig_session("m.onnx", _Options(), tmp_path)


def test_no_legacy_provider_options_remain():
    # The plugin EP rejects these keys and ORT then silently runs on CPU.
    for folder in ("tracker", "export", "tools", "eval", "examples"):
        for path in (ROOT / folder).rglob("*.py"):
            source = path.read_text()
            assert "migraphx_model_cache_dir" not in source, path
            assert 'providers=["MIGraphXExecutionProvider"' not in source, path
            assert not re.search(r'\("MIGraphXExecutionProvider",\s*\{', source), path


def test_release_gate_pins_match_supported_stack():
    gate = (ROOT / "tools/release_gate.sh").read_text()
    assert ep.SUPPORTED_STACK["migraphx"] in gate
    assert ep.SUPPORTED_STACK["ep_package"] in gate
    assert ep.SUPPORTED_STACK["onnxruntime_prefix"] in gate


def test_importing_tracker_does_not_change_benchmarking_env():
    # Build scripts pop MIGRAPHX_SKIP_BENCHMARKING, import tracker, then compile
    # caches; an import-time setdefault would silently disable autotuning.
    env = {k: v for k, v in os.environ.items() if k != "MIGRAPHX_SKIP_BENCHMARKING"}
    env["PYTHONPATH"] = str(ROOT)
    code = ("import os, tracker.migraphx_runtime; "
            "assert 'MIGRAPHX_SKIP_BENCHMARKING' not in os.environ")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
