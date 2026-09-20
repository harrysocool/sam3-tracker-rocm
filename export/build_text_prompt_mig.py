#!/usr/bin/env python3
"""Build all MIG artefacts needed for `tools/text_baseline.py --mig` in one command.

Runs the complete local model-build pipeline for the text-prompt MIG path:
  1. export_backbone_single   — ViT backbone ONNX (FP32, 4 FPN + last_hidden_state)
  2. simplify_backbone        — onnxsim graph simplification
  3. compile_backbone_mxr     — host-I/O fallback → tuned.mxr
  4. compile_backbone_mxr     — Torch GPU-I/O path → tuned_gpuio.mxr
  5. export_detr_encoder      — DETR encoder ONNX + onnxsim
  6. export_memory_attention_padded — S1..S10 memory-attention ONNX
  7. compile_memory_attention — independent, fully autotuned S1..S10 MXRs
  8. export_fixed_detr_decoder — fixed 504px decoder ONNX
  9. compile_fixed_detr_decoder — direct-I/O decoder MXR + checksum
 10. write_artifact_manifest — source/model/runtime identity + file hashes

Each step skips if its output file already exists (use --force to rebuild).

Usage:
  # Build 504px artefacts (~15 min)
  ./docker/rocm714/run.sh python export/build_text_prompt_mig.py --imgsz 504

  # Build both resolutions
  python export/build_text_prompt_mig.py --imgsz 504 1008

  # Rebuild backbone only (force)
  python export/build_text_prompt_mig.py --imgsz 504 --force --steps backbone
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_EC_POWER_MODE_PATH = Path("/sys/class/ec_su_axb35/apu/power_mode")


def full_autotune_env(base=None) -> dict[str, str]:
    """Return a child environment that cannot silently skip benchmarking."""
    env = dict(os.environ if base is None else base)
    env.pop("MIGRAPHX_SKIP_BENCHMARKING", None)
    env.pop("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", None)
    return env


def read_ec_power_mode(*, environ=None, power_mode_path: Path | None = None):
    """Return ``(mode, source)`` without attempting to modify the EC."""
    env = os.environ if environ is None else environ
    override = env.get("SAM3_EC_POWER_MODE", "").strip().lower()
    if override:
        if override not in {"quiet", "balanced", "performance"}:
            raise ValueError(
                "SAM3_EC_POWER_MODE must be quiet, balanced, or performance"
            )
        return override, "SAM3_EC_POWER_MODE"

    path = power_mode_path or Path(
        env.get("SAM3_EC_POWER_MODE_PATH", str(DEFAULT_EC_POWER_MODE_PATH))
    )
    try:
        mode = path.read_text(encoding="ascii").strip().lower()
    except OSError:
        return None, str(path)
    return mode or None, str(path)


def verify_ec_power_mode(*, required: bool, environ=None,
                         power_mode_path: Path | None = None) -> None:
    """Check the build-time power policy used by hardware autotuning."""
    mode, source = read_ec_power_mode(
        environ=environ, power_mode_path=power_mode_path
    )
    if mode == "performance":
        print(f"  EC power mode: performance ({source})")
        return

    if mode is None:
        message = (
            "cannot verify EC power mode; install/expose the platform EC driver "
            "or set SAM3_EC_POWER_MODE=performance only after verifying the BIOS"
        )
    else:
        message = (
            f"EC power mode is {mode!r} via {source}; performance autotuning "
            "requires 'performance'"
        )
    if required:
        raise RuntimeError(message)
    print(f"WARNING: {message}", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--imgsz", type=int, nargs="+", default=[504],
                   choices=[504, 1008],
                   help="Resolution(s) to build (default: 504). Pass both to build all.")
    p.add_argument("--checkpoint", type=Path,
                   default=WORKSPACE / "model/sam3",
                   help="Path to model/sam3 (default: model/sam3)")
    p.add_argument("--onnx-root", type=Path, default=None,
                   help="Root for onnx_files_<imgsz>/ dirs (default: project root)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if output files already exist")
    p.add_argument("--steps", nargs="+",
                   choices=["backbone", "detr_encoder", "memory_attention",
                            "fixed_decoder", "manifest", "all"],
                   default=["all"],
                   help="Which steps to run (default: all)")
    p.add_argument("--ptr-tokens", type=int, default=None,
                   help="Pointer token slots for memory_attention. "
                        "Default: 64 at 504px, 48 at 1008px (highest safe K per kernel cliff). "
                        "Set explicitly to override.")
    p.add_argument("--max-spatial-slots", type=int, default=10,
                   help="Generate exact memory-attention shapes S1..N. Default 10 "
                        "covers 7 non-conditioning plus up to 4 conditioning frames.")
    p.add_argument(
        "--performance-build",
        action="store_true",
        help="Require a verified EC performance mode before hardware autotuning.",
    )
    return p.parse_args()


def run(cmd: list[str], label: str, *, env=None) -> bool:
    """Run a subprocess; return True on success."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=WORKSPACE, env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED after {elapsed:.0f}s (exit {result.returncode})")
        return False
    print(f"\n  ✓ done in {elapsed:.0f}s")
    return True


def exists(path: Path, label: str, force: bool) -> bool:
    """Return True (skip) if path exists and not forcing."""
    if not force and path.exists():
        try:
            display = path.relative_to(WORKSPACE)
        except ValueError:
            display = path
        print(f"  skip: {label} already exists at {display}")
        return True
    return False


def build_for_imgsz(imgsz: int, args) -> bool:
    onnx_root = args.onnx_root or WORKSPACE
    onnx_dir = onnx_root / f"onnx_files_{imgsz}"
    det_dir = onnx_dir / "backbone_detector"
    mod_dir = onnx_dir / "detector_modules"
    trk_dir = onnx_dir / "tracker_modules"
    fixed_dir = onnx_dir / "detr_decoder_fixed"
    ptr_tokens = (
        args.ptr_tokens if args.ptr_tokens is not None
        else {504: 64, 1008: 48}.get(imgsz, 32)
    )

    steps = set(args.steps)
    run_all = "all" in steps
    ok = True

    # ── Step 1: export backbone ONNX ─────────────────────────────────────
    if run_all or "backbone" in steps:
        sink_env = full_autotune_env()
        sink_env["ROCMLIR_SINK_FINAL_ERF"] = "1"
        out = det_dir / "single_fp32.onnx"
        if not exists(out, "backbone ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/export_backbone_single.py",
                "--imgsz", str(imgsz),
                "--backbone-source", "detector",
                "--checkpoint", str(args.checkpoint),
                "--onnx-dir", str(onnx_dir),
            ], f"[1/10] Export backbone ONNX @{imgsz}px")

        # ── Step 2: simplify backbone ─────────────────────────────────────
        out = det_dir / "single_simplified.onnx"
        if not exists(out, "simplified backbone ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/simplify_backbone.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "detector",
            ], f"[2/10] Simplify backbone @{imgsz}px")

        # ── Step 3: compile .mxr ─────────────────────────────────────────
        out = det_dir / "tuned.mxr"
        if not exists(out, "tuned.mxr", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/compile_backbone_mxr.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "detector",
                "--skip-verify",
            ], f"[3/10] Compile backbone .mxr @{imgsz}px  (~12 min at 1008px)",
                env=full_autotune_env())

        # Keep the established host-I/O artifact as a portable fallback and
        # compile a distinct GPU-resident artifact for the Torch text path.
        # Never overwrite tuned.mxr: box/debug consumers still use NumPy I/O.
        gpu_io_out = det_dir / "tuned_gpuio.mxr"
        if not exists(gpu_io_out, "tuned_gpuio.mxr", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/compile_backbone_mxr.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "detector",
                "--gpu-io",
                "--skip-verify",
            ], f"[4/10] Compile FC1-sink GPU-I/O backbone .mxr @{imgsz}px", env=sink_env)

    # ── Step 4: export DETR encoder ───────────────────────────────────────
    if run_all or "detr_encoder" in steps:
        out = mod_dir / "detr_encoder_simplified.onnx"
        if not exists(out, "detr_encoder ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/detector/export_detr_encoder.py",
                "--imgsz", str(imgsz),
                "--checkpoint", str(args.checkpoint),
                "--onnx-dir", str(onnx_dir),
            ], f"[5/10] Export DETR encoder @{imgsz}px", env=full_autotune_env())

    # ── Step 5: export memory_attention ──────────────────────────────────
    if run_all or "memory_attention" in steps:
        # Resolve None default per imgsz (504→64, 1008→48 — kernel cliff aware)
        for spatial_slots in range(1, args.max_spatial_slots + 1):
            name = f"memory_attention_fixed_S{spatial_slots}_P{ptr_tokens}.onnx"
            out = trk_dir / name
            if not exists(out, name, args.force):
                ok = ok and run([
                    sys.executable,
                    "export/tracker_modules/export_memory_attention_padded.py",
                    "--imgsz", str(imgsz),
                    "--spatial-slots", str(spatial_slots),
                    "--ptr-tokens", str(ptr_tokens),
                    "--checkpoint", str(args.checkpoint),
                    "--onnx-dir", str(onnx_dir),
                ], f"[6/10] Export memory_attention (S{spatial_slots}_P{ptr_tokens}) @{imgsz}px")

        command = [
            sys.executable,
            "export/tracker_modules/compile_memory_attention.py",
            "--onnx-dir", str(onnx_dir),
            "--ptr-tokens", str(ptr_tokens),
            "--max-spatial-slots", str(args.max_spatial_slots),
        ]
        if args.force:
            command.append("--force")
        ok = ok and run(
            command,
            f"[7/10] Compile independently autotuned memory_attention S1-S{args.max_spatial_slots} @{imgsz}px",
            env=full_autotune_env(),
        )

    if run_all or "fixed_decoder" in steps:
        if imgsz != 504:
            print("  fixed decoder currently supports only 504px")
            return False
        fixed_onnx = fixed_dir / "detr_decoder_fixed_simplified.onnx"
        if not exists(fixed_onnx, "fixed decoder ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/detector/export_fixed_detr_decoder.py",
                "--checkpoint", str(args.checkpoint),
                "--output-dir", str(fixed_dir),
            ], "[8/10] Export fixed DETR decoder")
        fixed_mxr = fixed_dir / "direct_gpuio.mxr"
        if not exists(fixed_mxr, "fixed decoder MXR", args.force):
            command = [
                sys.executable,
                "export/detector/compile_fixed_detr_decoder.py",
                "--onnx", str(fixed_onnx),
                "--output", str(fixed_mxr),
            ]
            if args.force:
                command.append("--force")
            ok = ok and run(
                command,
                "[9/10] Compile fixed DETR decoder MXR",
                env=full_autotune_env(),
            )

    if run_all or "manifest" in steps:
        ok = ok and run(
            [
                sys.executable,
                "export/write_artifact_manifest.py",
                "--root", str(onnx_dir),
                "--checkpoint", str(args.checkpoint / "model.safetensors"),
                "--imgsz", str(imgsz),
                "--ptr-tokens", str(ptr_tokens),
                "--max-spatial-slots", str(args.max_spatial_slots),
            ],
            f"[10/10] Write artifact manifest @{imgsz}px",
        )

    return ok


def main():
    args = parse_args()
    verify_ec_power_mode(required=args.performance_build)
    t_start = time.perf_counter()
    all_ok = True

    for imgsz in args.imgsz:
        print(f"\n{'='*60}")
        print(f"  Building text-prompt MIG artefacts @ {imgsz}px")
        print(f"  checkpoint : {args.checkpoint}")
        print(f"  onnx_files : onnx_files_{imgsz}/")
        print(f"{'='*60}")
        ok = build_for_imgsz(imgsz, args)
        all_ok = all_ok and ok
        status = "✓ complete" if ok else "✗ FAILED"
        print(f"\n  {imgsz}px: {status}")

    total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Total: {total/60:.1f} min  |  {'ALL OK' if all_ok else 'SOME STEPS FAILED'}")
    print(f"{'='*60}")

    if all_ok:
        print(f"""
Next: prewarm ORT caches, then run the live smoke through docker/rocm714/run.sh.
The clean-environment runner performs both steps automatically:
  ./tools/docker_test_runner.sh --help
""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
