from __future__ import annotations

import pytest
import torch

from tracker.mig_vision_encoder import MIGVisionEncoder


class _FakeBackbone:
    gpu_io = True


class _CountingPositionEncoding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        # Give the fake module meaningful state so the state-dict test would
        # catch accidentally registering cache entries as buffers.
        self.register_buffer("state_marker", torch.tensor(1.0))

    def forward(self, shape, device, dtype, mask=None):
        assert mask is None
        self.calls += 1
        return torch.full(
            (shape[0], 256, shape[2], shape[3]),
            self.calls,
            device=device,
            dtype=dtype,
        )


def _outputs(dtype=torch.float16, device="cpu"):
    fpn = [
        torch.empty((1, 256, side, side), dtype=dtype, device=device)
        for side in (16, 8, 4, 2)
    ]
    return [*fpn, torch.empty((1, 16, 1024), dtype=dtype, device=device)]


def test_position_encoding_cache_is_per_encoder_and_reuses_tensors():
    # Share the position-encoding module to make this a direct check that the
    # caches live on the two MIGVisionEncoder instances, rather than on the
    # wrapped module or its forward function.
    position_encoding = _CountingPositionEncoding()
    first_encoder = MIGVisionEncoder(_FakeBackbone(), position_encoding)
    second_encoder = MIGVisionEncoder(_FakeBackbone(), position_encoding)

    first = first_encoder._build_output(
        _outputs(), torch.device("cpu"), torch.float16
    )
    second = second_encoder._build_output(
        _outputs(), torch.device("cpu"), torch.float16
    )
    first_again = first_encoder._build_output(
        _outputs(), torch.device("cpu"), torch.float16
    )
    second_again = second_encoder._build_output(
        _outputs(), torch.device("cpu"), torch.float16
    )

    assert position_encoding.calls == 8
    assert (
        first_encoder._position_encoding_cache
        is not second_encoder._position_encoding_cache
    )
    assert len(first_encoder._position_encoding_cache) == 4
    assert len(second_encoder._position_encoding_cache) == 4
    assert all(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(
            first.fpn_position_encoding,
            first_again.fpn_position_encoding,
            strict=True,
        )
    )
    assert all(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(
            second.fpn_position_encoding,
            second_again.fpn_position_encoding,
            strict=True,
        )
    )
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(
            first.fpn_position_encoding,
            second.fpn_position_encoding,
            strict=True,
        )
    )


def test_position_encoding_cache_does_not_change_state_dict_keys():
    encoder = MIGVisionEncoder(_FakeBackbone(), _CountingPositionEncoding())
    keys_before = tuple(encoder.state_dict().keys())

    encoder._build_output(_outputs(), torch.device("cpu"), torch.float16)

    keys_after = tuple(encoder.state_dict().keys())
    assert keys_before == ("position_encoding.state_marker",)
    assert keys_after == keys_before
    assert all("position_encoding_cache" not in key for key in keys_after)


def test_position_encoding_cache_is_bounded_and_cleared_by_apply():
    position_encoding = _CountingPositionEncoding()
    encoder = MIGVisionEncoder(_FakeBackbone(), position_encoding)

    for side in (2, 4, 8, 16, 32):
        encoder._get_position_encoding(
            torch.empty((1, 256, side, side), dtype=torch.float16)
        )

    assert position_encoding.calls == 5
    assert len(encoder._position_encoding_cache) == 4

    cached_values = [value for value, _ in encoder._position_encoding_cache.values()]
    assert all(value.dtype == torch.float16 for value in cached_values)
    encoder.float()
    assert not encoder._position_encoding_cache

    rebuilt = encoder._build_output(
        _outputs(dtype=torch.float32),
        torch.device("cpu"),
        torch.float32,
    )
    assert position_encoding.calls == 9
    assert len(encoder._position_encoding_cache) == 4
    assert all(value.dtype == torch.float32 for value in rebuilt.fpn_position_encoding)
    old_ptrs = {value.data_ptr() for value in cached_values}
    new_ptrs = {value.data_ptr() for value in rebuilt.fpn_position_encoding}
    assert old_ptrs.isdisjoint(new_ptrs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_position_encoding_cache_cross_stream_eviction_and_clear_lifetime():
    device = torch.device("cuda", torch.cuda.current_device())
    position_encoding = _CountingPositionEncoding().to(device)
    encoder = MIGVisionEncoder(_FakeBackbone(), position_encoding)
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    source = torch.empty((1, 256, 64, 64), device=device, dtype=torch.float16)
    source_key = (tuple(source.shape), device.type, device.index, source.dtype)

    # The cache entry is published before its producer stream necessarily
    # finishes. Keep that stream busy long enough that a missing wait_event
    # cannot pass merely because the small position-encoding kernels happened
    # to finish before the consumer was submitted.
    with torch.cuda.stream(producer):
        torch.cuda._sleep(100_000_000)
        first = encoder._get_position_encoding(source)
    ready_event = encoder._position_encoding_cache[source_key][1]
    assert ready_event is not None
    assert not ready_event.query()

    with torch.cuda.stream(consumer):
        cross_stream = encoder._get_position_encoding(source)
        consumed = cross_stream.clone()
        first_consume_done = torch.cuda.Event()
        first_consume_done.record(consumer)
    assert cross_stream.data_ptr() == first.data_ptr()
    assert not first_consume_done.query()

    first_consume_done.synchronize()
    assert torch.count_nonzero(consumed != 1).item() == 0
    del consumed, cross_stream, first
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # The producer is complete now, so only record_stream protects the cached
    # allocation. Delay the consumer after the cache hit, then evict and clear
    # the cache while the clone is demonstrably still pending.
    with torch.cuda.stream(consumer):
        cross_stream = encoder._get_position_encoding(source)
        cached_ptr = cross_stream.data_ptr()
        torch.cuda._sleep(100_000_000)
        consumed_after_clear = cross_stream.clone()
        second_consume_done = torch.cuda.Event()
        second_consume_done.record(consumer)
    assert not second_consume_done.query()

    # Four new keys evict the source from the four-entry LRU. Then _apply clears
    # the remaining values. Dropping the final Python reference must still not
    # make cached_ptr available to its original producer-stream allocator.
    for side in (2, 4, 8, 16):
        encoder._get_position_encoding(
            torch.empty((1, 256, side, side), device=device, dtype=torch.float16)
        )
    assert source_key not in encoder._position_encoding_cache
    del cross_stream

    encoder.float()
    assert not encoder._position_encoding_cache
    with torch.cuda.stream(producer):
        trash = [
            torch.full(
                (1, 256, 64, 64),
                -123.0,
                device=device,
                dtype=torch.float16,
            )
            for _ in range(4)
        ]
    assert not second_consume_done.query()
    assert all(tensor.data_ptr() != cached_ptr for tensor in trash)

    second_consume_done.synchronize()
    assert torch.count_nonzero(consumed_after_clear != 1).item() == 0
    assert all(tensor[0, 0, 0, 0].item() == -123.0 for tensor in trash)
