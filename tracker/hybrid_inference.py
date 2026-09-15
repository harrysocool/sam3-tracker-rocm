"""Unified scheduled-detection wrapper around :class:`SAM3Live`.

The wrapper keeps a single SAM3 model and one active inference session. Before
each scheduled keyframe it replaces the tracking session while preserving the
prompt cache, then runs fresh text detection. Between keyframes it uses the
same model's detector-skip tracker path. No second tracker backend or second
vision backbone is constructed.

The native tracker no-object gate remains active.  If a propagation result
loses an object that was present in the preceding output, the loss is reported
and full detection is forced on the next consumed frame.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .rocm_env import apply as _apply_rocm_env

_apply_rocm_env()

import numpy as np
import torch

from .live_inference import SAM3Live


class SAM3HybridLive:
    """Schedule full detection and propagation on one ``SAM3Live`` session.

    The public constructor remains compatible with the former dedicated hybrid
    implementation. Public object IDs are maintained across clean keyframe
    sessions by same-prompt mask-IoU association.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        prompts: Sequence[str],
        *,
        onnx_dir: str | Path,
        imgsz: int = 504,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        mig: bool = True,
        parallel_tail: bool | None = None,
        fixed_detr_decoder: bool | None = None,
        redetect_interval_ms: float = 1000.0,
        max_objects_per_prompt: int | dict[str, int] | None = 5,
        iou_assoc_threshold: float = 0.3,
        max_vision_features_cache_size: int = 1,
        bootstrap_frames: int = 0,
        bootstrap_min_score: float = 0.3,
        periodic_rebootstrap_seconds: float | None = None,
    ) -> None:
        self.imgsz = imgsz
        self.onnx_dir = Path(onnx_dir)
        self.keyframe_interval_s = max(
            0.001,
            float(redetect_interval_ms) / 1000.0,
        )
        self.iou_thresh = float(iou_assoc_threshold)
        self.max_per_prompt = max_objects_per_prompt

        started_at = time.perf_counter()
        self.live = SAM3Live(
            checkpoint=checkpoint,
            prompts=prompts,
            onnx_dir=onnx_dir,
            imgsz=imgsz,
            dtype=dtype,
            device=device,
            mig=mig,
            parallel_tail=parallel_tail,
            fixed_detr_decoder=fixed_detr_decoder,
            redetect_every=1,
            max_objects_per_prompt=max_objects_per_prompt,
            max_vision_features_cache_size=max_vision_features_cache_size,
            bootstrap_frames=bootstrap_frames,
            bootstrap_min_score=bootstrap_min_score,
            periodic_rebootstrap_seconds=periodic_rebootstrap_seconds,
        )
        self.device = self.live.device

        self._call_count = 0
        self._last_keyframe_time = 0.0
        self._last_was_keyframe = False
        self._force_keyframe_next = True
        self._pending_redetect_reason: str | None = "first_frame"
        self._previous_output_object_ids: set[int] = set()
        self._previous_public_masks: dict[int, np.ndarray] = {}
        self._previous_public_prompts: dict[int, str] = {}
        self._inner_to_public: dict[int, int] = {}
        self._next_public_object_id = 0
        self._inner_session_fresh = True
        print(
            f"[SAM3HybridLive] unified SAM3Live ready in "
            f"{time.perf_counter() - started_at:.1f}s "
            f"(redetect_interval_ms={redetect_interval_ms:.0f}, "
            "detector-skip propagation)",
        )

    def _ensure_inference_owner(self) -> None:
        for target in (self, getattr(self, "live", None)):
            if target is None:
                continue
            active_pipeline = getattr(
                target,
                "_latest_frame_pipeline_active",
                None,
            )
            if active_pipeline is not None and not getattr(
                active_pipeline,
                "_is_inference_owner_thread",
                lambda: False,
            )():
                raise RuntimeError(
                    "use LatestFramePipeline.infer_next() while the live "
                    "pipeline is active"
                )

    def _require_inactive_pipeline(self, operation: str) -> None:
        for target in (self, getattr(self, "live", None)):
            if target is None:
                continue
            if getattr(target, "_latest_frame_pipeline_active", None) is not None:
                raise RuntimeError(
                    f"close the active LatestFramePipeline before {operation}"
                )

    def _reset_scheduler(self, reason: str) -> None:
        self._force_keyframe_next = True
        self._pending_redetect_reason = reason
        self._last_keyframe_time = 0.0
        self._previous_output_object_ids.clear()
        self._previous_public_masks.clear()
        self._previous_public_prompts.clear()
        self._inner_to_public.clear()
        self._inner_session_fresh = True

    def reset_prompts(self, prompts: Sequence[str]) -> None:
        self._require_inactive_pipeline("reset_prompts()")
        self.live.reset_prompts(prompts)
        self._reset_scheduler("reset_prompts")

    def reset_tracking(self) -> None:
        self._require_inactive_pipeline("reset_tracking()")
        self.live.reset_tracking()
        self._reset_scheduler("reset_tracking")

    def close(self) -> None:
        self._require_inactive_pipeline("SAM3HybridLive.close()")
        self.live.close()

    @staticmethod
    def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape or not first.any() or not second.any():
            return 0.0
        intersection = int(np.logical_and(first, second).sum())
        if intersection == 0:
            return 0.0
        return intersection / int(np.logical_or(first, second).sum())

    @staticmethod
    def _object_prompts(result: dict) -> dict[int, str]:
        return {
            int(object_id): prompt
            for prompt, object_ids in result.get("prompt_to_obj_ids", {}).items()
            for object_id in object_ids
        }

    def _allocate_public_id(self) -> int:
        object_id = self._next_public_object_id
        self._next_public_object_id += 1
        return object_id

    def _fresh_keyframe_mapping(self, result: dict) -> dict[int, int]:
        """Associate fresh inner detections with the previous public masks."""
        inner_ids = [int(value) for value in result.get("object_ids", [])]
        inner_prompts = self._object_prompts(result)
        candidates = []
        for inner_id in inner_ids:
            prompt = inner_prompts.get(inner_id)
            mask = result.get("masks", {}).get(inner_id)
            if prompt is None or mask is None:
                continue
            for public_id, previous_mask in self._previous_public_masks.items():
                if self._previous_public_prompts.get(public_id) != prompt:
                    continue
                iou = self._mask_iou(np.asarray(mask), previous_mask)
                if iou >= self.iou_thresh:
                    candidates.append((iou, inner_id, public_id))

        mapping: dict[int, int] = {}
        used_public_ids: set[int] = set()
        for _iou, inner_id, public_id in sorted(
            candidates,
            key=lambda item: (-item[0], item[1], item[2]),
        ):
            if inner_id in mapping or public_id in used_public_ids:
                continue
            mapping[inner_id] = public_id
            used_public_ids.add(public_id)

        for inner_id in inner_ids:
            if inner_id not in mapping:
                mapping[inner_id] = self._allocate_public_id()
        return mapping

    def _continuing_mapping(self, result: dict) -> dict[int, int]:
        mapping = dict(self._inner_to_public)
        for inner_id in result.get("object_ids", []):
            inner_id = int(inner_id)
            if inner_id not in mapping:
                mapping[inner_id] = self._allocate_public_id()
        return mapping

    @staticmethod
    def _translate_result_ids(result: dict, mapping: dict[int, int]) -> dict:
        translated = dict(result)
        translated["object_ids"] = [
            mapping[int(object_id)]
            for object_id in result.get("object_ids", [])
            if int(object_id) in mapping
        ]
        for field in ("scores", "masks", "boxes"):
            translated[field] = {
                mapping[int(object_id)]: value
                for object_id, value in result.get(field, {}).items()
                if int(object_id) in mapping
            }
        translated["prompt_to_obj_ids"] = {
            prompt: [
                mapping[int(object_id)]
                for object_id in object_ids
                if int(object_id) in mapping
            ]
            for prompt, object_ids in result.get("prompt_to_obj_ids", {}).items()
        }
        return translated

    def _remember_public_output(self, result: dict) -> None:
        prompts = self._object_prompts(result)
        self._previous_public_masks = {
            int(object_id): np.asarray(result["masks"][object_id], dtype=bool).copy()
            for object_id in result.get("object_ids", [])
            if object_id in result.get("masks", {})
        }
        self._previous_public_prompts = {
            object_id: prompts[object_id]
            for object_id in self._previous_public_masks
            if object_id in prompts
        }

    def infer(
        self,
        frame_bgr: np.ndarray,
        *,
        full_detection: bool | None = None,
    ) -> dict:
        self._ensure_inference_owner()

        started_at = time.perf_counter()
        if self._force_keyframe_next:
            run_detection = True
            redetect_reason = self._pending_redetect_reason or "forced"
        elif full_detection is True:
            run_detection = True
            redetect_reason = "caller_override"
        elif full_detection is False:
            run_detection = False
            redetect_reason = None
        elif started_at - self._last_keyframe_time >= self.keyframe_interval_s:
            run_detection = True
            redetect_reason = "interval"
        else:
            run_detection = False
            redetect_reason = None

        clean_keyframe = False
        if run_detection:
            if not self._inner_session_fresh:
                infer_calls = self.live._infer_calls
                self.live._replace_tracking_session_preserving_prompts()
                # Session-local frame indices restart at zero, but this wrapper
                # must not reset SAM3Live's long-running bootstrap/drift cadence.
                self.live._infer_calls = infer_calls
                self._inner_session_fresh = True
            clean_keyframe = True

        try:
            inner_result = self.live.infer(frame_bgr, full_detection=run_detection)
        finally:
            self._inner_session_fresh = False

        actual_detected = bool(inner_result.get("detected", run_detection))
        if actual_detected and clean_keyframe:
            mapping = self._fresh_keyframe_mapping(inner_result)
        else:
            mapping = self._continuing_mapping(inner_result)
        self._inner_to_public = mapping
        result = self._translate_result_ids(inner_result, mapping)
        if actual_detected:
            self._last_keyframe_time = started_at
            self._force_keyframe_next = False
            self._pending_redetect_reason = None
            if not run_detection:
                redetect_reason = "inner_forced"
        elif run_detection:
            # Do not advance the cadence when the wrapped model reports that
            # the requested detector did not run. Keep retrying on the next
            # consumed frame.
            self._force_keyframe_next = True
            self._pending_redetect_reason = "detection_retry"
            redetect_reason = None

        current_ids = {int(object_id) for object_id in result.get("object_ids", [])}
        lost_object_ids = (
            sorted(self._previous_output_object_ids - current_ids)
            if not actual_detected
            else []
        )
        if lost_object_ids:
            self._force_keyframe_next = True
            self._pending_redetect_reason = "object_loss"

        self._previous_output_object_ids = current_ids
        self._last_was_keyframe = actual_detected
        result["detected"] = actual_detected
        result["keyframe"] = actual_detected
        result["frame_idx"] = self._call_count
        result["lost_object_ids"] = lost_object_ids
        result["redetect_reason"] = redetect_reason
        # Tracker-only output may add positive occupancy evidence, but absence
        # or zero masks must not clear free space until a fresh detector ran.
        result["negative_evidence_valid"] = actual_detected
        self._remember_public_output(result)
        self._call_count += 1
        return result


__all__ = ["SAM3HybridLive"]
