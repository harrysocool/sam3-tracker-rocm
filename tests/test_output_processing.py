"""Behavioral checks for common live/offline output and session semantics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Sam3VideoProcessor

from tools import text_baseline
from tracker.live_inference import SAM3Live
from tracker.output_processing import (
    enforce_per_prompt_cap,
    filter_result,
    postprocess_frame_output,
)


class _Session:
    def __init__(self, objects=None):
        self.obj_id_to_prompt_id = dict(objects or {})
        self.obj_ids = list(self.obj_id_to_prompt_id)
        self.prompts = {0: "people", 1: "dog"}
        self.video_height = None
        self.video_width = None
        self.hotstart_removed_obj_ids = set()
        self.removed = []

    def remove_object(self, object_id, strict=False):
        assert strict is False
        self.obj_ids.remove(object_id)
        self.obj_id_to_prompt_id.pop(object_id)
        self.removed.append(object_id)


def _raw(masks, scores=None, tracker_scores=None, frame_idx=0, suppressed=None):
    return SimpleNamespace(
        frame_idx=frame_idx,
        object_ids=list(masks),
        obj_id_to_mask=masks,
        obj_id_to_score=scores or {oid: 0.9 for oid in masks},
        obj_id_to_tracker_score=tracker_scores or {oid: 0.9 for oid in masks},
        suppressed_obj_ids=suppressed or set(),
    )


def _processor():
    # postprocess_outputs needs session state and tensor inputs, not weights/tokenizers.
    return Sam3VideoProcessor.__new__(Sam3VideoProcessor)


@pytest.mark.parametrize("use_live_method", [False, True])
def test_cap_evicts_real_session_objects_by_tracker_score(use_live_method):
    session = _Session({1: 0, 2: 0, 3: 1, 4: 1})
    session.obj_id_to_score = {1: 0.99, 2: 0.6, 3: 0.99, 4: 0.6}
    scores = {1: 0.1, 2: 0.9, 3: 0.1, 4: 0.9}
    if use_live_method:
        live = SAM3Live.__new__(SAM3Live)
        live.session = session
        live.max_objects_per_prompt = 1
        removed = live._enforce_per_prompt_cap(scores)
    else:
        removed = enforce_per_prompt_cap(session, scores, 1)
    assert removed == {1, 3}
    assert session.obj_ids == [2, 4]
    assert set(session.obj_id_to_prompt_id) == {2, 4}


def test_cap_is_reapplied_after_new_detections_and_keeps_stable_ties():
    session = _Session({1: 0, 2: 0})
    assert enforce_per_prompt_cap(session, {1: 0.9, 2: 0.9}, 1) == {2}
    session.obj_ids.append(3)
    session.obj_id_to_prompt_id[3] = 0
    assert enforce_per_prompt_cap(session, {1: 0.9, 3: 0.2}, 1) == {3}
    assert session.obj_ids == [1]


def test_per_prompt_api_limits_leave_unspecified_prompts_uncapped():
    session = _Session({1: 0, 2: 0, 3: 1, 4: 1})
    assert enforce_per_prompt_cap(session, {1: 0.9, 2: 0.8}, {"people": 1}) == {2}
    assert session.obj_ids == [1, 3, 4]


def test_unlimited_api_does_not_require_object_state():
    assert enforce_per_prompt_cap(object(), {}, None) == set()


def test_zero_api_cap_evicts_all_objects_unlike_cli_zero():
    session = _Session({1: 0, 2: 1})
    assert enforce_per_prompt_cap(session, {1: 0.9, 2: 0.8}, 0) == {1, 2}
    assert session.obj_ids == []


def test_masks_are_resized_as_logits_before_thresholding():
    session = _Session({1: 0})
    output = _raw({1: torch.tensor([[[-0.1, 1.0]]])})
    result = postprocess_frame_output(_processor(), session, output, (1, 3))
    np.testing.assert_array_equal(result["masks"][1], [[False, True, True]])
    assert result["boxes"][1] == (1.0, 0.0, 2.0, 0.0)
    assert result["masks"][1].dtype == np.bool_


def test_suppressed_empty_and_overlapping_masks_follow_processor_rules():
    session = _Session({1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1})
    session.hotstart_removed_obj_ids = {5}
    masks = {oid: torch.ones((1, 2, 2)) for oid in session.obj_ids}
    masks[6] *= -1
    output = _raw(masks, tracker_scores={1: 0.9, 2: 0.2, 3: 0.8, 4: 0.9, 5: 0.9, 6: 0.9},
                  suppressed={4})
    result = postprocess_frame_output(_processor(), session, output, (4, 4))
    assert result["object_ids"] == [1, 2, 3]
    assert result["masks"][1].all()
    assert not result["masks"][2].any()
    assert result["masks"][3].all()  # A different prompt keeps its overlapping region.
    assert result["prompt_to_obj_ids"] == {"people": [1, 2], "dog": [3]}


def test_evicted_ids_are_removed_from_output_before_prompt_lookup():
    session = _Session({1: 0, 2: 0})
    output = _raw({1: torch.ones((1, 2, 2)), 2: torch.ones((1, 2, 2))},
                  tracker_scores={1: 0.1, 2: 0.9})
    evicted = enforce_per_prompt_cap(session, output.obj_id_to_tracker_score, 1)
    result = postprocess_frame_output(_processor(), session, output, (4, 4), evicted)
    assert result["object_ids"] == [2]
    assert set(result["scores"]) == set(result["masks"]) == set(result["boxes"]) == {2}
    assert result["prompt_to_obj_ids"] == {"people": [2]}
    assert set(output.obj_id_to_mask) == {1, 2}  # Raw output is not mutated.


def test_score_filter_preserves_metadata_and_does_not_mutate_the_result():
    result = {
        "object_ids": [1, 2], "scores": {1: 0.5, 2: 0.4},
        "masks": {1: "one", 2: "two"}, "boxes": {1: (1,), 2: (2,)},
        "prompt_to_obj_ids": {"people": [1], "dog": [2]},
        "frame_idx": 7, "detected": False, "negative_evidence_valid": False,
        "lost_object_ids": [3], "redetect_reason": "object_loss", "generation": 4,
    }
    filtered = filter_result(result, 0.5)
    assert filtered["object_ids"] == [1]
    assert filtered["prompt_to_obj_ids"] == {"people": [1], "dog": []}
    assert filtered["generation"] == 4
    assert filtered["lost_object_ids"] == [3]
    assert filtered["negative_evidence_valid"] is False
    assert result["object_ids"] == [1, 2]
    assert set(result["scores"]) == {1, 2}


def test_empty_output_has_the_same_schema_and_original_size():
    result = postprocess_frame_output(_processor(), _Session(), _raw({}), (4, 5))
    assert result == {"object_ids": [], "scores": {}, "masks": {}, "boxes": {},
                      "prompt_to_obj_ids": {}, "frame_idx": 0}


def test_zero_cli_cap_is_unlimited_and_filtering_applies_every_frame():
    session = _Session({1: 0, 2: 1})
    args = SimpleNamespace(max_objects=0, min_score=0.5)
    first = _raw({1: torch.ones((1, 2, 2)), 2: torch.ones((1, 2, 2))})
    assert text_baseline._process_frame_output(args, _processor(), session, first, (4, 4))["object_ids"] == [1, 2]
    later = _raw(first.obj_id_to_mask, scores={1: 0.1, 2: 0.9}, frame_idx=1)
    result = text_baseline._process_frame_output(args, _processor(), session, later, (4, 4))
    assert result["object_ids"] == [2]
    assert session.obj_ids == [1, 2]
    assert session.removed == []
    assert result["detected"] and result["negative_evidence_valid"]


class _Capture:
    def __init__(self, count):
        self.frames = [np.full((2, 3, 3), i, dtype=np.uint8) for i in range(count)]
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, _property):
        return 24.0

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


@pytest.mark.parametrize(("limit", "expected"), [(0, 4), (2, 2), (8, 4)])
def test_offline_loader_zero_reads_to_eof(monkeypatch, limit, expected):
    capture = _Capture(4)
    monkeypatch.setattr(text_baseline.cv2, "VideoCapture", lambda _path: capture)
    pil, bgr, fps = text_baseline.load_video_frames(Path("unused"), limit)
    assert len(pil) == len(bgr) == expected
    assert fps == 24.0
    assert capture.released


def test_offline_keeps_processing_after_empty_first_frame(monkeypatch, tmp_path):
    session = _Session()
    processor_impl = _processor()

    class Processor:
        def init_video_session(self, **kwargs):
            session.processed_frames = dict(enumerate(kwargs["video"]))
            return session

        def add_text_prompt(self, target, prompts):
            target.prompts = dict(enumerate(prompts))

        def postprocess_outputs(self, **kwargs):
            return processor_impl.postprocess_outputs(**kwargs)

    class Model:
        def __init__(self):
            self.calls = []

        def __call__(self, *, inference_session, frame, frame_idx):
            assert len(session.processed_frames) == frame_idx
            session.processed_frames[frame_idx] = frame
            self.calls.append(frame_idx)
            if frame_idx == 0:
                return _raw({}, frame_idx=frame_idx)
            session.obj_ids = [1]
            session.obj_id_to_prompt_id[1] = 0
            return _raw({1: torch.ones((1, 2, 2))},
                        scores={1: 0.9 if frame_idx == 1 else 0.1}, frame_idx=frame_idx)

    class Writer:
        def __init__(self):
            self.frames = []
            self.released = False

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            self.released = True

    frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(text_baseline, "load_video_frames", lambda *_: (frames, frames, 24.0))
    writer = Writer()
    monkeypatch.setattr(text_baseline.cv2, "VideoWriter", lambda *_: writer)
    rendered_ids = []

    def render(frame, result, prompts, frame_idx=None):
        assert prompts == ["swan", "water"]
        rendered_ids.append(result["object_ids"])
        return frame

    monkeypatch.setattr(text_baseline, "overlay", render)
    args = SimpleNamespace(image=None, video=Path("unused"), max_frames=3,
                           text=["swan", "water"], max_objects=5, min_score=0.5,
                           output=tmp_path / "output.mp4", pipeline_backbone=False)
    model = Model()
    assert text_baseline._run_input(args, Processor(), model, torch.device("cpu"), torch.float32) == 0
    assert model.calls == [0, 1, 2]
    assert session.prompts == {0: "swan", 1: "water"}
    assert rendered_ids == [[], [1], []]
    assert len(writer.frames) == 3
    assert writer.released
