"""Artifact selection across container entry points, without building models."""

import importlib
from pathlib import Path
import sys

import pytest


ENTRYPOINTS = [
    ("demo_box", ["--checkpoint", "model", "--box", "0,0,10,10"]),
    ("eval.benchmarks.bench_pipeline", []),
    ("eval.benchmarks.benchmark_parallel_tail",
     ["--video", "video.mp4", "--text", "swan", "--out", "report.json"]),
    ("eval.benchmarks.profile_full_mig",
     ["--video", "video.mp4", "--text", "swan", "--out", "report.json"]),
    ("eval.datasets.eval_davis", []),
    ("eval.datasets.eval_saco_sg", []),
    ("eval.datasets.mask_diff_pt_vs_mig",
     ["--video", "video.mp4", "--text", "swan", "--out", "report.json"]),
    ("export.backbone.export_backbone_single", []),
    ("export.backbone.simplify_backbone", []),
    ("export.backbone.compile_backbone_mxr", []),
    ("export.detector.export_detr_encoder", []),
    ("export.tracker_modules.export_tracker_modules", []),
    ("export.tracker_modules.export_memory_attention_padded", []),
    ("export.tracker_modules.compile_memory_attention", ["--ptr-tokens", "64"]),
    ("export.tracker_modules.prewarm_ort_cache", []),
    ("export.build", []),
    ("export.build_text_prompt_mig", []),
]


@pytest.mark.parametrize(("module_name", "flags"), ENTRYPOINTS)
def test_entrypoint_uses_selected_artifacts_and_explicit_override(
    monkeypatch, module_name, flags,
):
    entrypoint = importlib.import_module(module_name)
    monkeypatch.setenv("SAM3_ONNX_DIR", "/workspace/results/selected-artifacts")
    monkeypatch.setattr(sys, "argv", [module_name, *flags])
    assert entrypoint.parse_args().onnx_dir == Path("/workspace/results/selected-artifacts")
    monkeypatch.setattr(sys, "argv", [module_name, *flags, "--onnx-dir", "/models/override"])
    assert entrypoint.parse_args().onnx_dir == Path("/models/override")


@pytest.mark.parametrize(("module_name", "flags"), ENTRYPOINTS)
def test_entrypoint_requires_artifact_selection(monkeypatch, module_name, flags):
    entrypoint = importlib.import_module(module_name)
    monkeypatch.delenv("SAM3_ONNX_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", [module_name, *flags])
    with pytest.raises(SystemExit) as caught:
        entrypoint.parse_args()
    assert caught.value.code == 2


def test_text_builder_explicit_parent_preserves_setup_destination(monkeypatch):
    from export import build_text_prompt_mig as builder

    monkeypatch.setenv("SAM3_ONNX_DIR", "/models/unrelated")
    monkeypatch.setattr(sys, "argv", ["builder", "--onnx-root", "/models"])
    args = builder.parse_args()
    assert args.onnx_dir is None
    assert args.onnx_root / f"onnx_files_{args.imgsz[0]}" == Path("/models/onnx_files_504")


def test_text_builder_multi_resolution_requires_separate_directories(monkeypatch):
    from export import build_text_prompt_mig as builder

    monkeypatch.setenv("SAM3_ONNX_DIR", "/models/onnx_files_504")
    monkeypatch.setattr(sys, "argv", ["builder", "--imgsz", "504", "1008"])
    with pytest.raises(SystemExit):
        builder.parse_args()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--onnx-root", "/workspace/results/build"])
    args = builder.parse_args()
    assert args.onnx_dir is None
    assert args.imgsz == [504, 1008]


def test_box_builder_passes_same_directory_to_every_stage(monkeypatch, tmp_path):
    from export import build

    monkeypatch.setenv("SAM3_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["builder", "--pipeline", "box"])
    commands = []
    monkeypatch.setattr(build, "run", lambda cmd, label: commands.append(cmd) or True)
    assert build.build_box(504, build.parse_args())
    assert len(commands) == 5
    for cmd in commands:
        assert cmd[cmd.index("--onnx-dir") + 1] == str(tmp_path)
