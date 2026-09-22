"""CPU-only guards for the complete text-prompt artifact build."""

from pathlib import Path

import pytest

from export import build_text_prompt_mig as builder


def test_full_autotune_environment_removes_skip_flag():
    env = builder.full_autotune_env(
        {
            "MIGRAPHX_SKIP_BENCHMARKING": "1",
            "MIGRAPHX_MLIR_USE_SPECIFIC_OPS": "stale",
            "KEEP": "value",
        }
    )
    assert "MIGRAPHX_SKIP_BENCHMARKING" not in env
    assert "MIGRAPHX_MLIR_USE_SPECIFIC_OPS" not in env
    assert env["KEEP"] == "value"


def test_performance_build_accepts_ec_performance(tmp_path, capsys):
    mode = tmp_path / "power_mode"
    mode.write_text("performance\n", encoding="ascii")
    builder.verify_ec_power_mode(required=True, environ={}, power_mode_path=mode)
    assert "EC power mode: performance" in capsys.readouterr().out


def test_performance_build_rejects_balanced_ec(tmp_path):
    mode = tmp_path / "power_mode"
    mode.write_text("balanced\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="EC power mode is 'balanced'"):
        builder.verify_ec_power_mode(
            required=True, environ={}, power_mode_path=mode
        )


def test_nonperformance_build_warns_when_ec_is_unknown(tmp_path, capsys):
    builder.verify_ec_power_mode(
        required=False,
        environ={},
        power_mode_path=tmp_path / "missing",
    )
    warning = capsys.readouterr().err
    assert "WARNING: cannot automatically verify EC power mode" in warning
    assert "set the BIOS power mode to Performance" in warning
    assert "SAM3_EC_POWER_MODE=performance" in warning
    assert "Optional EC power-mode verification" in warning


def test_performance_build_missing_ec_has_actionable_error(tmp_path):
    with pytest.raises(RuntimeError) as caught:
        builder.verify_ec_power_mode(
            required=True,
            environ={},
            power_mode_path=tmp_path / "missing",
        )
    message = str(caught.value)
    assert "set the BIOS power mode to Performance" in message
    assert "autotuning can select different kernels" in message
    assert "SAM3_EC_POWER_MODE=performance" in message
    assert "Optional EC power-mode verification" in message


def test_verified_bios_override_supports_platforms_without_sysfs(tmp_path):
    builder.verify_ec_power_mode(
        required=True,
        environ={"SAM3_EC_POWER_MODE": "performance"},
        power_mode_path=tmp_path / "missing",
    )


def test_invalid_ec_override_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="SAM3_EC_POWER_MODE"):
        builder.verify_ec_power_mode(
            required=True,
            environ={"SAM3_EC_POWER_MODE": "turbo"},
            power_mode_path=tmp_path / "missing",
        )


def test_performance_build_rejects_dirty_or_unknown_source():
    for value in ("1", "unknown"):
        with pytest.raises(RuntimeError, match="verified clean source"):
            builder.verify_source_clean(
                required=True, environ={"SAM3_SOURCE_DIRTY": value}
            )


def test_performance_build_accepts_clean_source():
    builder.verify_source_clean(
        required=True, environ={"SAM3_SOURCE_DIRTY": "0"}
    )
