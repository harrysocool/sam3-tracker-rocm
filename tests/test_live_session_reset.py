from __future__ import annotations

from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import pytest
import torch

from tracker.live_inference import SAM3Live


class _Cache:
    def __init__(self) -> None:
        self._vision_features = {}


class _Session:
    def __init__(self) -> None:
        self.inference_device = torch.device("cpu")
        self.inference_state_device = torch.device("cpu")
        self.video_storage_device = torch.device("cpu")
        self.dtype = torch.float16
        self.max_vision_features_cache_size = 4
        self.processed_frames = None
        self.cache = _Cache()
        self._obj_id_to_idx = OrderedDict()
        self._obj_idx_to_id = OrderedDict()
        self.obj_ids = []
        self.mask_inputs_per_obj = {}
        self.point_inputs_per_obj = {}
        self.output_dict_per_obj = {}
        self.frames_tracked_per_obj = {}
        self.prompts = {}
        self.prompt_input_ids = {}
        self.prompt_embeddings = {}
        self.prompt_attention_masks = {}
        self.obj_id_to_prompt_id = {}
        self.obj_id_to_score = {}
        self.obj_id_to_tracker_score_frame_wise = defaultdict(dict)
        self.obj_id_to_last_occluded = {}
        self.max_obj_id = -1
        self.obj_first_frame_idx = {}
        self.unmatched_frame_inds = defaultdict(list)
        self.overlap_pair_to_frame_inds = defaultdict(list)
        self.trk_keep_alive = {}
        self.removed_obj_ids = set()
        self.suppressed_obj_ids = defaultdict(set)
        self.hotstart_removed_obj_ids = set()
        self.output_buffer = []
        self._parallel_tail_failed = False


class _Processor:
    def __init__(self, *, fail_init: bool = False, fail_prompt: bool = False) -> None:
        self.fail_init = fail_init
        self.fail_prompt = fail_prompt
        self.init_calls = []
        self.add_calls = []

    def init_video_session(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("synthetic session construction failure")
        session = _Session()
        session.inference_device = kwargs["inference_device"]
        session.inference_state_device = kwargs["inference_device"]
        session.video_storage_device = kwargs["inference_device"]
        session.dtype = kwargs["dtype"]
        session.max_vision_features_cache_size = kwargs[
            "max_vision_features_cache_size"
        ]
        return session

    def add_text_prompt(self, session, prompts):
        self.add_calls.append((session, list(prompts)))
        for prompt in prompts:
            prompt_id = len(session.prompts)
            session.prompts[prompt_id] = prompt
            session.prompt_input_ids[prompt_id] = torch.tensor([prompt_id])
            session.prompt_attention_masks[prompt_id] = torch.tensor([1])
            if self.fail_prompt:
                raise RuntimeError("synthetic prompt installation failure")


def _dirty_session() -> _Session:
    session = _Session()
    session.prompts = {0: "floor", 1: "wall"}
    session.prompt_input_ids = {
        0: torch.tensor([10, 11]),
        1: torch.tensor([20, 21]),
    }
    # Prompt 1 is intentionally still lazy and has no embedding.
    session.prompt_embeddings = {0: torch.tensor([[1.5]])}
    session.prompt_attention_masks = {
        0: torch.tensor([1, 1]),
        1: torch.tensor([1, 1]),
    }
    session.processed_frames = {8: torch.tensor([8])}
    session.cache._vision_features = {8: object()}
    session._obj_id_to_idx[42] = 0
    session._obj_idx_to_id[0] = 42
    session.obj_ids = [42]
    session.output_dict_per_obj = {0: {"cond_frame_outputs": {8: object()}}}
    session.frames_tracked_per_obj = {0: {8: {"reverse": False}}}
    session.obj_id_to_prompt_id = {42: 0}
    session.obj_id_to_score = {42: 0.9}
    session.max_obj_id = 42
    session.obj_first_frame_idx = {42: 8}
    session.unmatched_frame_inds[42].append(8)
    session.overlap_pair_to_frame_inds[(42, 43)].append(8)
    session.trk_keep_alive = {42: 3}
    session.removed_obj_ids = {41}
    session.suppressed_obj_ids[8].add(41)
    session.hotstart_removed_obj_ids = {40}
    session.output_buffer = [object()]
    session._parallel_tail_failed = True
    return session


def _live(processor: _Processor | None = None) -> SAM3Live:
    live = SAM3Live.__new__(SAM3Live)
    live.processor = processor or _Processor()
    live.device = torch.device("cpu")
    live.dtype = torch.float16
    live.max_vision_features_cache_size = 4
    live.parallel_tail = True
    live.session = _dirty_session()
    live.model = SimpleNamespace(_skip_detection=True)
    live._next_frame_idx = 17
    live._infer_calls = 19
    live._force_detect_next = False
    live._detector_call_counter = 5
    live.bootstrap_frames = 2
    live._bootstrap_remaining = {0: 1}
    live._exemplar_box_pool = {0: [object()]}
    live._exemplar_boxes = {0: torch.tensor([1.0])}
    live._drift_baseline_score = {0: 0.8}
    live._drift_recent_scores = {0: [0.7]}
    live._drift_frames_since_bootstrap = {0: 9}
    live._drift_pending_rebootstrap = True
    live._last_bootstrap_complete_time = 123.0
    return live


def _assert_empty_tracking_state(session: _Session) -> None:
    assert session.processed_frames is None
    assert session.cache._vision_features == {}
    assert session._obj_id_to_idx == {}
    assert session._obj_idx_to_id == {}
    assert session.obj_ids == []
    assert session.output_dict_per_obj == {}
    assert session.frames_tracked_per_obj == {}
    assert session.obj_id_to_prompt_id == {}
    assert session.obj_id_to_score == {}
    assert session.obj_id_to_tracker_score_frame_wise == {}
    assert session.obj_id_to_last_occluded == {}
    assert session.max_obj_id == -1
    assert session.obj_first_frame_idx == {}
    assert session.unmatched_frame_inds == {}
    assert session.overlap_pair_to_frame_inds == {}
    assert session.trk_keep_alive == {}
    assert session.removed_obj_ids == set()
    assert session.suppressed_obj_ids == {}
    assert session.hotstart_removed_obj_ids == set()
    assert session.output_buffer == []


def _assert_local_counters_reset(live: SAM3Live) -> None:
    assert live._next_frame_idx == 0
    assert live._infer_calls == 0
    assert live._force_detect_next is True
    assert live._detector_call_counter == 0
    assert live.model._skip_detection is False
    assert live.session._parallel_tail_failed is False


def test_reset_tracking_replaces_session_and_shares_lazy_prompt_tensors():
    live = _live()
    old = live.session
    old_dicts = {
        name: getattr(old, name)
        for name in (
            "prompts",
            "prompt_input_ids",
            "prompt_embeddings",
            "prompt_attention_masks",
        )
    }

    live.reset_tracking()

    assert live.session is not old
    _assert_empty_tracking_state(live.session)
    assert list(live.session.prompts.items()) == [(0, "floor"), (1, "wall")]
    for name, old_values in old_dicts.items():
        new_values = getattr(live.session, name)
        assert new_values is not old_values
        assert list(new_values) == list(old_values)
        for key, value in old_values.items():
            assert new_values[key] is value
    assert 1 not in live.session.prompt_embeddings
    assert live.processor.init_calls == [
        {
            "video": None,
            "inference_device": torch.device("cpu"),
            "dtype": torch.float16,
            "max_vision_features_cache_size": 4,
        }
    ]
    _assert_local_counters_reset(live)


def test_reset_prompts_installs_ordered_multi_prompt_state_before_swap():
    live = _live()
    old = live.session

    live.reset_prompts(["wall", "floor", "chair"])

    assert live.session is not old
    assert list(live.session.prompts.values()) == ["wall", "floor", "chair"]
    assert list(live.session.prompt_input_ids) == [0, 1, 2]
    assert list(live.session.prompt_attention_masks) == [0, 1, 2]
    assert live.session.prompt_embeddings == {}
    assert live.processor.add_calls == [
        (live.session, ["wall", "floor", "chair"])
    ]
    assert old.prompts == {0: "floor", 1: "wall"}
    _assert_empty_tracking_state(live.session)
    _assert_local_counters_reset(live)
    assert live._bootstrap_remaining == {0: 2, 1: 2, 2: 2}
    assert live._exemplar_box_pool == {0: [], 1: [], 2: []}
    assert live._exemplar_boxes == {}
    assert live._drift_baseline_score == {}
    assert live._drift_recent_scores == {}
    assert live._drift_frames_since_bootstrap == {}
    assert live._drift_pending_rebootstrap is False
    assert live._last_bootstrap_complete_time == 0.0


@pytest.mark.parametrize("operation", ["reset_tracking", "reset_prompts"])
def test_session_creation_failure_rolls_back_everything(operation):
    live = _live(_Processor(fail_init=True))
    old = live.session

    with pytest.raises(RuntimeError, match="session construction failure"):
        if operation == "reset_tracking":
            live.reset_tracking()
        else:
            live.reset_prompts(["new"])

    assert live.session is old
    assert live._next_frame_idx == 17
    assert live._infer_calls == 19
    assert live._force_detect_next is False
    assert live.model._skip_detection is True


def test_prompt_install_failure_rolls_back_old_session_and_local_state():
    live = _live(_Processor(fail_prompt=True))
    old = live.session

    with pytest.raises(RuntimeError, match="prompt installation failure"):
        live.reset_prompts(["new", "second"])

    assert live.session is old
    assert old.prompts == {0: "floor", 1: "wall"}
    assert live._bootstrap_remaining == {0: 1}
    assert live._next_frame_idx == 17
    assert live._infer_calls == 19
    assert live._force_detect_next is False
    assert live.model._skip_detection is True


def test_private_tracking_replacement_failure_rolls_back_without_public_guard():
    live = _live(_Processor(fail_init=True))
    live._latest_frame_pipeline_active = object()
    old = live.session

    with pytest.raises(RuntimeError, match="session construction failure"):
        live._replace_tracking_session_preserving_prompts()

    assert live.session is old
    assert live._next_frame_idx == 17
    assert live._infer_calls == 19
    assert live._force_detect_next is False
    assert live.model._skip_detection is True


def test_public_reset_guard_runs_before_replacement_construction():
    live = _live()
    live._latest_frame_pipeline_active = object()
    old = live.session

    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.reset_tracking()
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.reset_prompts(["new"])

    assert live.session is old
    assert live.processor.init_calls == []
