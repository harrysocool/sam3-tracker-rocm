#!/usr/bin/env python3
"""Compare complete SAM3 detector masks for a canonical NPU route vs PyTorch."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_FILE.parent))

from benchmark_sam3_vit import (  # noqa: E402
    DEFAULT_FLEXML_CACHE_DIR,
    DEFAULT_FLEXML_CACHE_KEY,
    DEFAULT_FLEXML_ONNX,
    DEFAULT_IRON_BIN,
    create_flexml_session,
    preprocess_input,
    resolve_repo_path,
    sha256,
    warmup_flexml,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=("flexml", "iron"), required=True)
    p.add_argument("--checkpoint", default="model/sam3")
    p.add_argument("--image", default="assets/truck.jpg")
    p.add_argument("--prompt", default="truck")
    p.add_argument("--imgsz", type=int, default=504)
    p.add_argument("--output", required=True)
    p.add_argument("--visual-dir", required=True)
    p.add_argument("--min-mask-iou", type=float, default=0.95)
    p.add_argument("--flexml-onnx", default=DEFAULT_FLEXML_ONNX)
    p.add_argument("--flexml-cache-dir", default=DEFAULT_FLEXML_CACHE_DIR)
    p.add_argument("--flexml-cache-key", default=DEFAULT_FLEXML_CACHE_KEY)
    p.add_argument("--vaip-config", default="")
    p.add_argument("--npu-bin", default=DEFAULT_IRON_BIN)
    p.add_argument("--omp-threads", type=int, default=8)
    return p.parse_args()


def patch_flexml_encoder(model, session):
    import torch
    import torch.nn as nn
    from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

    original = model.detector_model.vision_encoder
    position_encoding = original.neck.position_encoding
    output_names = [o.name for o in session.get_outputs()]

    class FlexmlVisionEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.position_encoding = position_encoding

        def forward(self, pixel_values, **kwargs):
            device, dtype = pixel_values.device, pixel_values.dtype
            np_input = np.ascontiguousarray(
                pixel_values.detach().float().cpu().numpy()
            )
            raw = session.run(None, {session.get_inputs()[0].name: np_input})
            outputs = dict(zip(output_names, raw))
            fpn = tuple(
                torch.from_numpy(np.ascontiguousarray(outputs[f"fpn_{i}"]))
                .to(device=device, dtype=dtype)
                for i in range(4)
            )
            hidden = torch.from_numpy(
                np.ascontiguousarray(outputs["last_hidden_state"])
            ).to(device=device, dtype=dtype)
            positions = tuple(
                self.position_encoding(t.shape, t.device, t.dtype) for t in fpn
            )
            return Sam3VisionEncoderOutput(
                last_hidden_state=hidden,
                fpn_hidden_states=fpn,
                fpn_position_encoding=positions,
                hidden_states=None,
                attentions=None,
            )

    model.detector_model.vision_encoder = FlexmlVisionEncoder().to("cuda")


def prompt_ids(result: dict, prompt: str) -> list[int]:
    return [int(v) for v in result.get("prompt_to_obj_ids", {}).get(prompt, [])]


def as_mask(result: dict, object_id: int) -> np.ndarray:
    value = result["masks"][object_id]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).squeeze().astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def compare_masks(reference: dict, candidate: dict, prompt: str) -> dict:
    ref_ids = prompt_ids(reference, prompt)
    cand_ids = prompt_ids(candidate, prompt)
    pairs = []
    for ref_id in ref_ids:
        for cand_id in cand_ids:
            pairs.append(
                (mask_iou(as_mask(reference, ref_id), as_mask(candidate, cand_id)),
                 ref_id, cand_id)
            )
    pairs.sort(reverse=True)
    used_ref, used_cand, matches = set(), set(), []
    for iou, ref_id, cand_id in pairs:
        if ref_id in used_ref or cand_id in used_cand:
            continue
        used_ref.add(ref_id)
        used_cand.add(cand_id)
        ref_box = np.asarray(reference["boxes"][ref_id], dtype=np.float32)
        cand_box = np.asarray(candidate["boxes"][cand_id], dtype=np.float32)
        matches.append(
            {
                "reference_id": ref_id,
                "candidate_id": cand_id,
                "mask_iou": iou,
                "box_linf": float(np.max(np.abs(ref_box - cand_box))),
                "score_abs": abs(
                    float(reference["scores"][ref_id])
                    - float(candidate["scores"][cand_id])
                ),
            }
        )

    ious = [m["mask_iou"] for m in matches]
    box_errors = [m["box_linf"] for m in matches]
    score_errors = [m["score_abs"] for m in matches]
    return {
        "prompt": prompt,
        "reference_objects": len(ref_ids),
        "candidate_objects": len(cand_ids),
        "matched_objects": len(matches),
        "mask_iou_mean": float(np.mean(ious)) if ious else None,
        "mask_iou_min": min(ious) if ious else None,
        "box_linf_max": max(box_errors) if box_errors else None,
        "score_abs_max": max(score_errors) if score_errors else None,
        "matches": matches,
    }


def union_mask(result: dict, prompt: str, shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=bool)
    for object_id in prompt_ids(result, prompt):
        merged |= as_mask(result, object_id)
    return merged


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]):
    result = image.copy()
    color_arr = np.asarray(color, dtype=np.float32)
    result[mask] = (
        result[mask].astype(np.float32) * 0.45 + color_arr * 0.55
    ).astype(np.uint8)
    return result


def save_visuals(
    image: np.ndarray,
    reference: dict,
    candidate: dict,
    prompt: str,
    output_dir: Path,
) -> dict[str, str]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    shape = image.shape[:2]
    ref_mask = union_mask(reference, prompt, shape)
    cand_mask = union_mask(candidate, prompt, shape)
    reference_path = output_dir / "reference.png"
    candidate_path = output_dir / "candidate.png"
    difference_path = output_dir / "difference.png"
    cv2.imwrite(str(reference_path), overlay(image, ref_mask, (0, 220, 0)))
    cv2.imwrite(str(candidate_path), overlay(image, cand_mask, (220, 120, 0)))
    difference = (image.astype(np.float32) * 0.35).astype(np.uint8)
    difference[np.logical_and(ref_mask, cand_mask)] = (0, 180, 0)
    difference[np.logical_and(ref_mask, ~cand_mask)] = (0, 0, 255)
    difference[np.logical_and(~ref_mask, cand_mask)] = (255, 0, 0)
    cv2.imwrite(str(difference_path), difference)
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "difference": str(difference_path),
        "reference_sha256": sha256(reference_path),
        "candidate_sha256": sha256(candidate_path),
        "difference_sha256": sha256(difference_path),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_mask_iou <= 1.0:
        raise ValueError("--min-mask-iou must be between 0 and 1")

    flexml = None
    if args.route == "flexml":
        flexml = create_flexml_session(args)
        np_input = preprocess_input(args)
        warmup_flexml(flexml[0], np_input, 1)

    from tracker.rocm_env import apply as apply_rocm_env

    apply_rocm_env()
    import cv2
    from tracker.live_inference import SAM3Live

    image_path = resolve_repo_path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    live = SAM3Live(
        checkpoint=str(resolve_repo_path(args.checkpoint)),
        prompts=[args.prompt],
        imgsz=args.imgsz,
        mig=False,
        redetect_every=1,
    )

    reference = live.infer(image, full_detection=True)
    live.reset_tracking()

    cleanup = None
    if args.route == "flexml":
        patch_flexml_encoder(live.model, flexml[0])
        variant = "vitisai_ep_backbone_detector_504_v1"
    else:
        from tracker.npu_backbone_service import patch_sam3_with_npu_backbone

        cleanup = patch_sam3_with_npu_backbone(
            live.model,
            npu_bin=args.npu_bin,
            omp_threads=args.omp_threads,
        )
        variant = "p14_m1536_common_atb_affine"

    t0 = time.perf_counter()
    candidate = live.infer(image, full_detection=True)
    candidate_ms = (time.perf_counter() - t0) * 1000.0
    if cleanup is not None:
        cleanup.shutdown()

    comparison = compare_masks(reference, candidate, args.prompt)
    passed = (
        comparison["reference_objects"] > 0
        and comparison["reference_objects"] == comparison["candidate_objects"]
        and comparison["matched_objects"] == comparison["reference_objects"]
        and comparison["mask_iou_min"] is not None
        and comparison["mask_iou_min"] >= args.min_mask_iou
    )
    visual_dir = Path(args.visual_dir)
    if not visual_dir.is_absolute():
        visual_dir = REPO_ROOT / visual_dir
    visuals = save_visuals(
        image, reference, candidate, args.prompt, visual_dir
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "route": args.route,
        "variant": variant,
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "prompt": args.prompt,
        "candidate_detection_ms": candidate_ms,
        "thresholds": {"min_mask_iou": args.min_mask_iou},
        "comparison": comparison,
        "passed": passed,
        "visuals": visuals,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print("SAM3 complete mask validation")
    print(f"route={args.route} variant={variant} prompt={args.prompt}")
    print(
        f"objects reference/candidate/matched="
        f"{comparison['reference_objects']}/"
        f"{comparison['candidate_objects']}/"
        f"{comparison['matched_objects']}"
    )
    print(
        f"mask IoU mean/min={comparison['mask_iou_mean']}/"
        f"{comparison['mask_iou_min']}"
    )
    print(
        f"box_linf_max={comparison['box_linf_max']} "
        f"score_abs_max={comparison['score_abs_max']}"
    )
    print(f"result={output}")
    print(f"MASK_VALIDATION={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
