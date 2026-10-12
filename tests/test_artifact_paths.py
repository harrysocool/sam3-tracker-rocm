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
    assert args.onnx_root / "onnx_files_504" == Path("/models/onnx_files_504")
    assert not hasattr(args, "imgsz")


@pytest.mark.parametrize("module_name", ["export.build", "export.build_text_prompt_mig"])
@pytest.mark.parametrize("resolution", ["504", "1008"])
def test_builders_have_no_resolution_option(monkeypatch, module_name, resolution):
    builder = importlib.import_module(module_name)
    monkeypatch.setenv("SAM3_ONNX_DIR", "/models/onnx_files_504")
    monkeypatch.setattr(sys, "argv", ["builder", "--imgsz", resolution])
    with pytest.raises(SystemExit):
        builder.parse_args()


@pytest.mark.parametrize(("module_name", "flags"), [
    item for item in ENTRYPOINTS
    if item[0] not in {"export.build", "export.build_text_prompt_mig",
                       "export.tracker_modules.compile_memory_attention",
                       "export.tracker_modules.prewarm_ort_cache"}
])
def test_lower_level_entrypoints_default_to_504_and_reject_1008(monkeypatch, module_name, flags):
    entrypoint = importlib.import_module(module_name)
    monkeypatch.setenv("SAM3_ONNX_DIR", "/models/onnx_files_504")
    monkeypatch.setattr(sys, "argv", [module_name, *flags])
    assert entrypoint.parse_args().imgsz == 504
    monkeypatch.setattr(sys, "argv", [module_name, *flags, "--imgsz", "1008"])
    with pytest.raises(SystemExit) as caught:
        entrypoint.parse_args()
    assert caught.value.code == 2


def test_box_builder_passes_same_directory_to_every_stage(monkeypatch, tmp_path):
    from export import build

    monkeypatch.setenv("SAM3_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["builder", "--pipeline", "box"])
    commands = []
    monkeypatch.setattr(build, "run", lambda cmd, label: commands.append(cmd) or True)
    assert build.build_box(build.parse_args())
    assert len(commands) == 5
    for cmd in commands:
        assert cmd[cmd.index("--onnx-dir") + 1] == str(tmp_path)
        if "--imgsz" in cmd:
            assert cmd[cmd.index("--imgsz") + 1] == "504"


def test_text_builder_passes_504_shapes_to_all_stages(monkeypatch, tmp_path):
    from export import build_text_prompt_mig as builder

    monkeypatch.setenv("SAM3_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["builder", "--checkpoint", "/models/sam3"])
    commands = []
    monkeypatch.setattr(builder, "prepare_build_root", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "invalidate_artifact_manifest", lambda *args: None)
    monkeypatch.setattr(builder, "run", lambda cmd, label, **kwargs: commands.append(cmd) or True)
    assert builder.build_artifacts(builder.parse_args())
    assert len(commands) == 20
    for cmd in commands:
        if "--imgsz" in cmd:
            assert cmd[cmd.index("--imgsz") + 1] == "504"
        if "--onnx-dir" in cmd:
            assert cmd[cmd.index("--onnx-dir") + 1] == str(tmp_path)
    manifest = commands[-1]
    assert manifest[1] == "export/write_artifact_manifest.py"
    assert manifest[manifest.index("--ptr-tokens") + 1] == "64"


def test_onnx_tracker_api_rejects_unsupported_resolution_before_loading():
    from tracker import SAM3OnnxTracker

    with pytest.raises(ValueError, match="only the validated 504px resolution"):
        SAM3OnnxTracker(checkpoint="unused", onnx_dir="unused", imgsz=1008)


def test_tracker_prewarm_rejects_1008_artifacts_before_creating_cache(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from export.tracker_modules import prewarm_ort_cache as prewarm

    modules = tmp_path / "tracker_modules"
    modules.mkdir()
    (modules / "memory_attention_fixed_N7.onnx").touch()
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(shape=[5184, 1, 256])])
    monkeypatch.setattr(prewarm.ort, "InferenceSession", lambda *args, **kwargs: session)
    monkeypatch.setenv("SAM3_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["prewarm"])
    with pytest.raises(ValueError, match="only 504px artifacts"):
        prewarm.main()
    assert not (modules / "mxr_cache").exists()
