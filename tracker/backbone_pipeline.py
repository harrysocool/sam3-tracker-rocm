"""Cross-frame vision-backbone prefetch for preloaded video sessions."""
from __future__ import annotations

import types
from collections.abc import Iterable, Iterator

import torch

from .mig_vision_encoder import MIGVisionEncoder, _PendingVisionOutput
from .ort_gpu_io import fence_ort_inputs


_MISSING = object()


class BackbonePrefetchPipeline:
    """Feed prefetched vision outputs into sequential ``Sam3VideoModel`` calls.

    The tracker state remains strictly frame ordered.  Only the state-free
    vision encoder for frame N+1 overlaps the detector/tracker tail for frame
    N.  The model must already have the parallel-tail patch, which supplies
    the ORT input fences needed when cached features cross stream boundaries.
    """

    def __init__(self, model, inference_session) -> None:
        if not hasattr(model, "_parallel_tail_runtime"):
            raise ValueError("backbone prefetch requires the parallel-tail patch")

        detector = model.detector_model
        encoder = detector.vision_encoder
        if not isinstance(encoder, MIGVisionEncoder) or not encoder.mxr.gpu_io:
            raise ValueError("backbone prefetch requires a MIG GPU-I/O vision encoder")

        self.model = model
        self.inference_session = inference_session
        self.detector = detector
        self.encoder = encoder
        self.device = next(model.parameters()).device
        self.stream = torch.cuda.Stream(device=self.device)
        self._current = None
        self._pending: _PendingVisionOutput | None = None
        self._consuming: _PendingVisionOutput | None = None
        self._expected_input_ptr: int | None = None
        self._current_consumed = False
        self._active = False
        self._running = False

    def __enter__(self):
        if getattr(self.model, "_backbone_prefetch_active", False):
            raise RuntimeError("backbone prefetch is already active on this model")
        if hasattr(self.model, "_failed_backbone_prefetch_keepalive"):
            raise RuntimeError(
                "a previous backbone prefetch could not be drained; rebuild the model"
            )
        self._original_instance_getter = self.detector.__dict__.get(
            "get_vision_features",
            _MISSING,
        )

        def use_current(_detector, pixel_values, **kwargs):
            if self._current is None:
                raise RuntimeError("no prefetched vision output is ready")
            if self._current_consumed:
                raise RuntimeError("prefetched vision output was requested more than once")
            if pixel_values.data_ptr() != self._expected_input_ptr:
                raise RuntimeError("prefetched vision output does not match the input frame")
            self._current_consumed = True
            return self._current

        self.detector.get_vision_features = types.MethodType(
            use_current,
            self.detector,
        )
        self.model._backbone_prefetch_active = True
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # A consumer can stop iteration while the next backbone is in flight.
        # Drain that stream before releasing its raw-pointer keepalive object.
        drained = False
        try:
            self.stream.synchronize()
            drained = True
        except BaseException:
            if exc_type is None:
                raise
        finally:
            if drained:
                self._pending = None
                self._consuming = None
            else:
                self.model._failed_backbone_prefetch_keepalive = (
                    self._pending,
                    self._consuming,
                )
            self._current = None
            self._expected_input_ptr = None
            self._current_consumed = False
            if self._original_instance_getter is _MISSING:
                delattr(self.detector, "get_vision_features")
            else:
                self.detector.get_vision_features = self._original_instance_getter
            delattr(self.model, "_backbone_prefetch_active")
            self._active = False
            self._running = False

    def _enqueue(self, frame_idx: int) -> _PendingVisionOutput:
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            pixel_values = self.inference_session.get_frame(frame_idx).unsqueeze(0)
            return self.encoder._prefetch(pixel_values, self.stream)

    def run(
        self,
        frame_indices: Iterable[int],
        *,
        reverse: bool = False,
    ) -> Iterator[tuple[int, object]]:
        """Yield ordered ``(frame_idx, output)`` pairs with one-frame lookahead."""
        if not self._active:
            raise RuntimeError("use BackbonePrefetchPipeline as a context manager")
        if self._running:
            raise RuntimeError("a backbone-prefetch iterator is already active")
        self._running = True

        indices = iter(frame_indices)
        try:
            current_index = next(indices)
        except StopIteration:
            self._running = False
            return

        # Input preprocessing ran on the caller's stream when the video
        # session was created. Establish that dependency before the first
        # background backbone launch.
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self._pending = self._enqueue(current_index)

        for next_index in indices:
            self._consuming = self._pending
            self._current = self._consuming.wait_on(
                torch.cuda.current_stream(self.device)
            )
            self._expected_input_ptr = self._consuming.source_data_ptr
            self._current_consumed = False
            self._pending = self._enqueue(next_index)
            with torch.inference_mode(), fence_ort_inputs():
                output = self.model(
                    inference_session=self.inference_session,
                    frame_idx=current_index,
                    reverse=reverse,
                )
            if not self._current_consumed:
                raise RuntimeError("model did not consume the prefetched vision output")
            self._consuming = None
            yield current_index, output
            if not self._active:
                raise RuntimeError("backbone-prefetch iterator resumed outside its context")
            current_index = next_index

        self._consuming = self._pending
        self._current = self._consuming.wait_on(torch.cuda.current_stream(self.device))
        self._expected_input_ptr = self._consuming.source_data_ptr
        self._current_consumed = False
        self._pending = None
        with torch.inference_mode(), fence_ort_inputs():
            output = self.model(
                inference_session=self.inference_session,
                frame_idx=current_index,
                reverse=reverse,
            )
        if not self._current_consumed:
            raise RuntimeError("model did not consume the prefetched vision output")
        self._consuming = None
        self._running = False
        yield current_index, output


__all__ = ["BackbonePrefetchPipeline"]
