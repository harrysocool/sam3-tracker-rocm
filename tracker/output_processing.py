"""Shared object limits and output processing for live and offline SAM3."""

from __future__ import annotations

import numpy as np


DEFAULT_MAX_OBJECTS_PER_PROMPT = 5


def cap_for_prompt(max_objects_per_prompt, prompt_text: str) -> int | None:
    """Resolve a global or per-prompt limit; None leaves a prompt uncapped."""
    if isinstance(max_objects_per_prompt, int):
        return max_objects_per_prompt
    if isinstance(max_objects_per_prompt, dict):
        return max_objects_per_prompt.get(prompt_text)
    return None


def enforce_per_prompt_cap(session, tracker_scores: dict, max_objects_per_prompt) -> set[int]:
    """Evict excess session objects using current tracker scores, not frozen detection scores."""
    if max_objects_per_prompt is None:
        return set()

    by_prompt: dict[str, list[tuple[int, float]]] = {}
    for object_id in list(session.obj_ids):
        prompt_id = session.obj_id_to_prompt_id.get(object_id)
        if prompt_id is None:
            continue
        prompt = session.prompts.get(prompt_id, "?")
        score = float(tracker_scores.get(object_id, 0.0))
        by_prompt.setdefault(prompt, []).append((object_id, score))

    evicted = set()
    for prompt, items in by_prompt.items():
        cap = cap_for_prompt(max_objects_per_prompt, prompt)
        if cap is None or len(items) <= cap:
            continue
        # Stable sorting preserves session order for equal scores.
        items.sort(key=lambda item: item[1], reverse=True)
        for object_id, _ in items[cap:]:
            session.remove_object(object_id, strict=False)
            evicted.add(object_id)
    return evicted


def postprocess_frame_output(processor, session, raw_output, original_size, evicted_ids=None):
    """Apply the processor's mask, suppression, overlap and box rules, then return CPU outputs."""
    height, width = original_size

    def retained(values):
        if not evicted_ids or not isinstance(values, dict):
            return values
        return {key: value for key, value in values.items() if key not in evicted_ids}

    processed = processor.postprocess_outputs(
        inference_session=session,
        model_outputs={
            "obj_id_to_mask": retained(raw_output.obj_id_to_mask),
            "obj_id_to_score": retained(raw_output.obj_id_to_score),
            "obj_id_to_tracker_score": retained(raw_output.obj_id_to_tracker_score),
            "suppressed_obj_ids": raw_output.suppressed_obj_ids,
        },
        original_sizes=[[height, width]],
    )
    object_ids = processed["object_ids"].tolist()
    scores = processed["scores"].tolist()
    if object_ids:
        masks = processed["masks"].cpu().numpy()
        boxes = processed["boxes"].cpu().numpy()
    else:
        masks = np.zeros((0, height, width), dtype=bool)
        boxes = np.zeros((0, 4), dtype=np.float32)

    return {
        "object_ids": object_ids,
        "scores": {oid: float(score) for oid, score in zip(object_ids, scores)},
        "masks": {oid: masks[index] for index, oid in enumerate(object_ids)},
        "boxes": {
            oid: tuple(float(value) for value in boxes[index])
            for index, oid in enumerate(object_ids)
        },
        "prompt_to_obj_ids": processed["prompt_to_obj_ids"],
        "frame_idx": raw_output.frame_idx,
    }


def filter_result(result: dict, min_score: float) -> dict:
    """Return a threshold-filtered copy while preserving scheduling metadata."""
    keep = [oid for oid in result["object_ids"] if result["scores"].get(oid, 0.0) >= min_score]
    keep_set = set(keep)
    filtered = dict(result)
    filtered["object_ids"] = keep
    for field in ("scores", "masks", "boxes"):
        filtered[field] = {oid: value for oid, value in result[field].items() if oid in keep_set}
    filtered["prompt_to_obj_ids"] = {
        prompt: [oid for oid in object_ids if oid in keep_set]
        for prompt, object_ids in result["prompt_to_obj_ids"].items()
    }
    return filtered
