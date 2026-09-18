from __future__ import annotations

import pytest

from tracker.mig_memory_attention import required_memory_attention_slots


def test_required_memory_attention_slots_matches_tracker_horizon():
    assert required_memory_attention_slots(7, 4) == tuple(range(1, 11))
    assert required_memory_attention_slots(3, 1) == (1, 2, 3)
    assert required_memory_attention_slots(7, -1) is None


def test_required_memory_attention_slots_rejects_invalid_values():
    with pytest.raises(ValueError, match="num_maskmem"):
        required_memory_attention_slots(0, 1)
    with pytest.raises(ValueError, match="max_cond_frame_num"):
        required_memory_attention_slots(7, 0)
