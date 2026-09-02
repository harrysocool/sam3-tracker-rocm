from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

import tracker.mig_detr_decoder as fixed_decoder


class _Shape:
    def __init__(self, lens, type_name="float_type"):
        self._lens = tuple(lens)
        self._type_name = type_name

    def lens(self):
        return self._lens

    def strides(self):
        stride = 1
        result = []
        for length in reversed(self._lens):
            result.append(stride)
            stride *= length
        return list(reversed(result))

    def type_string(self):
        return self._type_name


class _Program:
    def __init__(self, shapes):
        self._shapes = shapes

    def get_parameter_shapes(self):
        return self._shapes


class _OriginalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.box_head = nn.Identity()


def _valid_shapes():
    return {
        "vision_features": _Shape((1, 1296, 256)),
        "text_features": _Shape((1, 32, 256)),
        "vision_pos_encoding": _Shape((1, 1296, 256)),
        "text_cross_attn_mask": _Shape((1, 1, 1, 32)),
        "main:#output_0": _Shape((6, 1, 200, 256)),
        "main:#output_1": _Shape((6, 1, 200, 4)),
        "main:#output_2": _Shape((6, 1, 1)),
    }


def test_fixed_decoder_rejects_wrong_artifact_hash(tmp_path):
    artifact = tmp_path / "wrong.mxr"
    artifact.write_bytes(b"not the accepted artifact")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        fixed_decoder.MIGFixedDetrDecoder(artifact, _OriginalDecoder())


def test_fixed_decoder_validates_parameter_contract(monkeypatch, tmp_path):
    artifact = tmp_path / "fixed.mxr"
    artifact.write_bytes(b"mock")
    monkeypatch.setattr(
        fixed_decoder,
        "_sha256",
        lambda path: fixed_decoder.FIXED_DETR_DECODER_SHA256,
    )
    monkeypatch.setattr(
        fixed_decoder.migraphx,
        "load",
        lambda path: _Program(_valid_shapes()),
    )
    original = _OriginalDecoder()
    decoder = fixed_decoder.MIGFixedDetrDecoder(artifact, original)
    assert decoder.box_head is original.box_head
    assert decoder._original_decoder_keepalive is original
    assert decoder.output_names == (
        "main:#output_0",
        "main:#output_1",
        "main:#output_2",
    )


def test_fixed_decoder_rejects_wrong_shape(monkeypatch, tmp_path):
    artifact = tmp_path / "fixed.mxr"
    artifact.write_bytes(b"mock")
    shapes = _valid_shapes()
    shapes["text_features"] = _Shape((1, 31, 256))
    monkeypatch.setattr(
        fixed_decoder,
        "_sha256",
        lambda path: fixed_decoder.FIXED_DETR_DECODER_SHA256,
    )
    monkeypatch.setattr(
        fixed_decoder.migraphx,
        "load",
        lambda path: _Program(shapes),
    )
    with pytest.raises(RuntimeError, match="unexpected fixed decoder parameter"):
        fixed_decoder.MIGFixedDetrDecoder(artifact, _OriginalDecoder())
