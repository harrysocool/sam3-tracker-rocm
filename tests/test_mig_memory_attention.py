from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest
import torch

from tracker.mig_memory_attention import MIGMemoryAttention


class _FakeSession:
    def __init__(self, slots: int, *, mig: bool = True) -> None:
        self._slots = slots
        self._mig = mig

    def get_inputs(self):
        return [
            SimpleNamespace(name="current_vision_features", shape=(1296, 1, 256)),
            SimpleNamespace(name="memory", shape=(self._slots * 1296 + 64, 1, 64)),
        ]


def _fake_load_session(self, path):
    slots = int(re.search(r"_S(\d+)_", path.name).group(1))
    session = _FakeSession(slots)
    return session, "output", (1, 256, 36, 36), True


def _write_shapes(tmp_path, slots):
    for slot in slots:
        (tmp_path / f"memory_attention_fixed_S{slot}_P64.onnx").write_bytes(b"onnx")
    return tmp_path / "memory_attention_fixed_S7_P64.onnx"


def test_constructor_rejects_incomplete_required_shape_coverage(tmp_path, monkeypatch):
    base = _write_shapes(tmp_path, (7,))
    monkeypatch.setattr(MIGMemoryAttention, "_load_session", _fake_load_session)

    with pytest.raises(RuntimeError, match=r"missing S=\[1, 2, 3, 4, 5, 6, 8, 9, 10\]"):
        MIGMemoryAttention(
            base,
            lambda **kwargs: None,
            required_spatial_slots=range(1, 11),
            allow_pytorch_fallback=False,
        )


def test_constructor_accepts_complete_required_shape_coverage(tmp_path, monkeypatch):
    base = _write_shapes(tmp_path, range(1, 11))
    monkeypatch.setattr(MIGMemoryAttention, "_load_session", _fake_load_session)

    shim = MIGMemoryAttention(
        base,
        lambda **kwargs: None,
        required_spatial_slots=range(1, 11),
        allow_pytorch_fallback=False,
    )

    assert tuple(sorted(shim._sessions)) == tuple(range(1, 11))
    assert shim.required_spatial_slots == tuple(range(1, 11))


def _fallback_shim(*, allow: bool):
    shim = MIGMemoryAttention.__new__(MIGMemoryAttention)
    torch.nn.Module.__init__(shim)
    shim._original_forward = lambda **kwargs: "pytorch"
    shim.HW = 4
    shim.ptr_tokens = 64
    shim._sessions = {}
    shim._pt_fallback_calls = 0
    shim._warned_fallback_reasons = set()
    shim.allow_pytorch_fallback = allow
    return shim


def test_runtime_fallback_logs_once_per_shape(caplog):
    shim = _fallback_shim(allow=True)
    current = torch.empty((4, 1, 256))
    memory = torch.empty((4, 1, 64))

    with caplog.at_level(logging.WARNING):
        assert shim(current, memory) == "pytorch"
        assert shim(current, memory) == "pytorch"

    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "missing S1 specialization" in messages[0]
    assert "fallback count=1" in messages[0]
    assert shim._pt_fallback_calls == 2


def test_runtime_fallback_can_be_disabled():
    shim = _fallback_shim(allow=False)
    current = torch.empty((4, 1, 256))
    memory = torch.empty((4, 1, 64))

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        shim(current, memory)
