"""Fixed-encoder/sliding-vocoder inference over inference-only TorchScript.

This is the public-runtime equivalent of
``lscodec/streaming/stream_recon_sliding.py``.  It intentionally knows only
the release model's tensor interfaces:

* encoder: ``(1, 1, samples) -> (tokens, groups)``;
* prompt pre-net: ``(1, prompt_frames, 1024) -> cached prompt``;
* vocoder: ``(1, repeated_tokens, dim), cached prompt -> waveform``.

No training model classes are imported or required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
import torch


# The released 25 Hz encoder's convolution stack. These values are model
# interface metadata, not learned parameters.
ENCODER_CONVOLUTIONS = (
    (512, 10, 5),
    (512, 8, 4),
    (512, 8, 4),
    (512, 4, 2),
    (512, 4, 2),
    (512, 4, 2),
)


class EncoderModule(Protocol):
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor: ...


class VocoderModule(Protocol):
    def __call__(
        self, vq: torch.Tensor, prompt: torch.Tensor
    ) -> torch.Tensor: ...


def convolution_geometry(
    layers: Sequence[tuple[int, int, int]] = ENCODER_CONVOLUTIONS,
) -> tuple[int, int]:
    """Return ``(total_stride, receptive_field)`` in input samples."""
    stride = 1
    receptive_field = 1
    for _, kernel, layer_stride in layers:
        receptive_field += (int(kernel) - 1) * stride
        stride *= int(layer_stride)
    return stride, receptive_field


def number_of_frames(
    samples: int,
    layers: Sequence[tuple[int, int, int]] = ENCODER_CONVOLUTIONS,
) -> int:
    """Number of encoder frames produced by an unpadded convolution stack."""
    length = int(samples)
    for _, kernel, stride in layers:
        length = (length - int(kernel)) // int(stride) + 1
        if length <= 0:
            return 0
    return length


@dataclass(frozen=True)
class StreamingConfig:
    """Public streaming controls, matching ``stream_recon_sliding.py``."""

    chunk_ms: float = 320.0
    left_context_ms: float = 320.0
    lookahead_ms: float = 160.0
    crossfade_ms: float = 20.0
    arrival_ms: float = 100.0
    prompt_anchor_ms: float = 320.0

    def frame_counts(self, token_rate: float) -> tuple[int, int, int]:
        center = max(1, int(round(self.chunk_ms / 1000.0 * token_rate)))
        left = max(
            0, int(round(self.left_context_ms / 1000.0 * token_rate))
        )
        right = max(0, int(round(self.lookahead_ms / 1000.0 * token_rate)))
        return center, left, right

    def crossfade_samples(self, output_sample_rate: int) -> int:
        return max(
            0,
            int(round(self.crossfade_ms / 1000.0 * output_sample_rate)),
        )

    def arrival_samples(self, input_sample_rate: int) -> int:
        return max(
            1, int(round(self.arrival_ms / 1000.0 * input_sample_rate))
        )


class FixedWindowEncoder:
    """Incremental ring buffer with fixed-window token transfer batches.

    The first transfer contains ``left + center`` tokens; later transfers
    contain one ``center`` hop. Right-context tokens are held until they have
    full lookahead, matching the reference implementation.
    """

    def __init__(
        self,
        model: EncoderModule,
        center_frames: int,
        left_frames: int,
        right_frames: int,
        *,
        normalization_anchor: Optional[torch.Tensor] = None,
        convolutions: Sequence[tuple[int, int, int]] = ENCODER_CONVOLUTIONS,
    ):
        if center_frames <= 0:
            raise ValueError("center_frames must be positive")
        if left_frames < 0 or right_frames < 0:
            raise ValueError("context frame counts cannot be negative")
        self.model = model
        self.center_frames = int(center_frames)
        self.left_frames = int(left_frames)
        self.right_frames = int(right_frames)
        self.bootstrap_tokens = self.left_frames + self.center_frames
        self.step_tokens = self.center_frames
        self.convolutions = tuple(convolutions)
        self.total_stride, self.receptive_field = convolution_geometry(
            self.convolutions
        )
        if normalization_anchor is not None:
            if (
                normalization_anchor.ndim != 3
                or tuple(normalization_anchor.shape[:2]) != (1, 1)
                or normalization_anchor.shape[-1] == 0
            ):
                raise ValueError(
                    "normalization_anchor must be a non-empty (1, 1, T) tensor"
                )
            normalization_anchor = normalization_anchor.detach()
        self.normalization_anchor = normalization_anchor
        self.reset()

    def reset(self) -> None:
        self._buffer: Optional[torch.Tensor] = None
        self._buffer_start = 0
        self._samples_seen = 0
        self._center_start = 0
        self._pending: Optional[torch.Tensor] = None
        self._bootstrapped = False

    def _append(self, chunk: torch.Tensor) -> None:
        if chunk.ndim != 3 or tuple(chunk.shape[:2]) != (1, 1):
            raise ValueError("encoder chunks must have shape (1, 1, samples)")
        self._buffer = (
            chunk
            if self._buffer is None
            else torch.cat((self._buffer, chunk), dim=-1)
        )
        self._samples_seen += int(chunk.shape[-1])

    def _trim(self) -> None:
        next_window_frame = max(
            0, self._center_start - self.left_frames
        )
        next_window_sample = next_window_frame * self.total_stride
        drop = next_window_sample - self._buffer_start
        if drop > 0 and self._buffer is not None:
            self._buffer = self._buffer[..., drop:].contiguous()
            self._buffer_start += drop

    def _emit_block(
        self, total_frames_available: int
    ) -> Optional[torch.Tensor]:
        center_start = self._center_start
        if center_start >= total_frames_available:
            return None

        center_end = min(
            center_start + self.center_frames, total_frames_available
        )
        window_start_frame = max(
            0, center_start - self.left_frames
        )
        window_start_sample = window_start_frame * self.total_stride
        window_end_frame = min(
            total_frames_available, center_end + self.right_frames
        )
        window_end_sample = min(
            self._samples_seen,
            (window_end_frame - 1) * self.total_stride + self.receptive_field,
        )
        window_end_sample = max(
            window_end_sample,
            window_start_sample + self.receptive_field,
        )
        window_end_sample = min(window_end_sample, self._samples_seen)

        assert self._buffer is not None
        low = window_start_sample - self._buffer_start
        high = window_end_sample - self._buffer_start
        window = self._buffer[..., low:high]
        window_frames = number_of_frames(
            int(window.shape[-1]), self.convolutions
        )
        encoder_input = window
        if self.normalization_anchor is not None:
            anchor = self.normalization_anchor.to(
                device=window.device, dtype=window.dtype
            )
            encoder_input = torch.cat((window, anchor), dim=-1)

        indices = self.model(encoder_input)
        if indices.ndim == 1:
            indices = indices.unsqueeze(-1)
        if indices.ndim != 2:
            raise RuntimeError(
                "released encoder must return (tokens, groups); "
                f"got {tuple(indices.shape)}"
            )
        indices = indices[:window_frames]
        local_start = center_start - window_start_frame
        local_end = min(
            local_start + center_end - center_start,
            int(indices.shape[0]),
        )
        output = (
            indices[local_start:local_end]
            if local_end > local_start
            else None
        )
        self._center_start = center_end
        self._trim()
        return output

    def _accumulate(self, indices: Optional[torch.Tensor]) -> None:
        if indices is None or indices.numel() == 0:
            return
        self._pending = (
            indices
            if self._pending is None
            else torch.cat((self._pending, indices), dim=0)
        )

    def _drain(self, final: bool) -> list[torch.Tensor]:
        batches: list[torch.Tensor] = []
        while self._pending is not None and self._pending.shape[0] > 0:
            needed = (
                self.step_tokens
                if self._bootstrapped
                else self.bootstrap_tokens
            )
            if self._pending.shape[0] >= needed:
                batches.append(self._pending[:needed])
                self._pending = self._pending[needed:]
                self._bootstrapped = True
            elif final:
                batches.append(self._pending)
                self._pending = None
            else:
                break
        return batches

    @torch.inference_mode()
    def push(self, waveform_chunk: torch.Tensor) -> list[torch.Tensor]:
        self._append(waveform_chunk)
        outputs: list[torch.Tensor] = []
        while True:
            needed_samples = (
                self._center_start
                + self.center_frames
                + self.right_frames
                - 1
            ) * self.total_stride + self.receptive_field
            if self._samples_seen < needed_samples:
                break
            block = self._emit_block(
                number_of_frames(
                    self._samples_seen, self.convolutions
                )
            )
            if block is not None and block.numel() > 0:
                outputs.append(block)
        if outputs:
            self._accumulate(torch.cat(outputs, dim=0))
        return self._drain(final=False)

    @torch.inference_mode()
    def flush(self) -> list[torch.Tensor]:
        total_frames = number_of_frames(
            self._samples_seen, self.convolutions
        )
        outputs: list[torch.Tensor] = []
        while self._center_start < total_frames:
            block = self._emit_block(total_frames)
            if block is not None and block.numel() > 0:
                outputs.append(block)
        if outputs:
            self._accumulate(torch.cat(outputs, dim=0))
        return self._drain(final=True)


class SlidingWindowVocoder:
    """Non-causal sliding-window decoder around the released TorchScript."""

    def __init__(
        self,
        model: VocoderModule,
        prompt: torch.Tensor,
        center_frames: int,
        left_frames: int,
        right_frames: int,
        *,
        crossfade_samples: int,
        repeat_input_tokens: bool = True,
        samples_per_repeated_token: int = 480,
    ):
        if center_frames <= 0:
            raise ValueError("center_frames must be positive")
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0)
        if prompt.ndim != 3:
            raise ValueError("prompt must have shape (1, frames, channels)")
        self.model = model
        self.prompt = prompt
        self.center_frames = int(center_frames)
        self.left_frames = int(left_frames)
        self.right_frames = int(right_frames)
        self.crossfade_samples = int(crossfade_samples)
        self.repeat = 2 if repeat_input_tokens else 1
        self.samples_per_token = (
            int(samples_per_repeated_token) * self.repeat
        )
        self.reset()

    def reset(self) -> None:
        self._tokens: Optional[torch.Tensor] = None
        self._token_start = 0
        self._token_count = 0
        self._emitted = 0
        self._held: Optional[torch.Tensor] = None
        self._bootstrapped = False

    def _append(self, vectors: torch.Tensor) -> None:
        if vectors.ndim != 3 or vectors.shape[0] != 1:
            raise ValueError("VQ chunks must have shape (1, tokens, dim)")
        self._tokens = (
            vectors
            if self._tokens is None
            else torch.cat((self._tokens, vectors), dim=1)
        )
        self._token_count += int(vectors.shape[1])

    def _trim(self) -> None:
        next_window_start = max(
            0, self._emitted - self.left_frames
        )
        drop = next_window_start - self._token_start
        if drop > 0 and self._tokens is not None:
            self._tokens = self._tokens[:, drop:, :].contiguous()
            self._token_start += drop

    def _decode_window(self, vectors: torch.Tensor) -> torch.Tensor:
        if self.repeat > 1:
            vectors = vectors.repeat_interleave(self.repeat, dim=1)
        waveform = self.model(vectors, self.prompt)
        if waveform.ndim != 3 or tuple(waveform.shape[:2]) != (1, 1):
            raise RuntimeError(
                "released vocoder must return (1, 1, samples); "
                f"got {tuple(waveform.shape)}"
            )
        return waveform

    def _commit(self, running: torch.Tensor) -> torch.Tensor:
        crossfade = self.crossfade_samples
        if crossfade <= 0:
            return running
        if running.shape[-1] <= crossfade:
            self._held = running
            return running.new_zeros((1, 1, 0))
        self._held = running[..., -crossfade:]
        return running[..., :-crossfade]

    @torch.inference_mode()
    def _decode_next(
        self, total_available: int
    ) -> Optional[torch.Tensor]:
        if not self._bootstrapped:
            window_start = 0
            center_start = 0
            center_end = min(
                self.left_frames + self.center_frames,
                total_available,
            )
        else:
            center_start = self._emitted
            center_end = min(
                center_start + self.center_frames, total_available
            )
            window_start = max(
                0, center_start - self.left_frames
            )
        window_end = min(
            total_available, center_end + self.right_frames
        )
        if center_end <= center_start:
            return None

        assert self._tokens is not None
        vectors = self._tokens[
            :,
            window_start - self._token_start:
            window_end - self._token_start,
            :,
        ]
        full_waveform = self._decode_window(vectors)
        local_start = center_start - window_start
        center_sample_start = local_start * self.samples_per_token
        center_sample_end = center_sample_start + (
            center_end - center_start
        ) * self.samples_per_token

        crossfade = self.crossfade_samples
        if (
            self._held is None
            or self._held.numel() == 0
            or crossfade <= 0
        ):
            running = full_waveform[
                ..., center_sample_start:center_sample_end
            ]
        else:
            extension_start = max(
                0, center_sample_start - crossfade
            )
            actual = center_sample_start - extension_start
            block = full_waveform[
                ..., extension_start:center_sample_end
            ]
            head = block[..., :actual]
            body = block[..., actual:]
            fade_in = torch.linspace(
                0.0,
                1.0,
                steps=actual,
                device=block.device,
                dtype=block.dtype,
            )
            fade_out = 1.0 - fade_in
            previous = self._held
            finalized = previous[
                ..., : previous.shape[-1] - actual
            ]
            blended = (
                previous[..., previous.shape[-1] - actual:] * fade_out
                + head * fade_in
            )
            running = torch.cat((finalized, blended, body), dim=-1)

        emitted = self._commit(running)
        self._emitted = center_end
        self._bootstrapped = True
        self._trim()
        return emitted

    @torch.inference_mode()
    def push(self, vectors: torch.Tensor) -> Optional[torch.Tensor]:
        self._append(vectors)
        outputs: list[torch.Tensor] = []
        while True:
            if not self._bootstrapped:
                ready = self._token_count >= (
                    self.left_frames
                    + self.center_frames
                    + self.right_frames
                )
            else:
                ready = self._token_count >= (
                    self._emitted
                    + self.center_frames
                    + self.right_frames
                )
            if not ready:
                break
            block = self._decode_next(self._token_count)
            if block is not None and block.numel() > 0:
                outputs.append(block)
        return torch.cat(outputs, dim=-1) if outputs else None

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        while self._emitted < self._token_count:
            block = self._decode_next(self._token_count)
            if block is None:
                break
            if block.numel() > 0:
                outputs.append(block)
        if self._held is not None and self._held.numel() > 0:
            outputs.append(self._held)
            self._held = self._held.new_zeros((1, 1, 0))
        if outputs:
            return torch.cat(outputs, dim=-1)
        if self._tokens is not None:
            return self._tokens.new_zeros((1, 1, 0))
        return torch.zeros((1, 1, 0), device=self.prompt.device)


class StreamingSession:
    """Stateful, genuinely incremental waveform-to-waveform session."""

    def __init__(
        self,
        encoder: FixedWindowEncoder,
        vocoder: SlidingWindowVocoder,
        codebook: torch.Tensor,
    ):
        if codebook.ndim == 2:
            codebook = codebook.unsqueeze(0)
        if codebook.ndim != 3:
            raise ValueError(
                "codebook must have shape (groups, entries, dimension)"
            )
        self.encoder = encoder
        self.vocoder = vocoder
        self.codebook = codebook
        self.num_groups = int(codebook.shape[0])
        self.transfer_sizes: list[int] = []
        self.finished = False

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            indices = indices.unsqueeze(-1)
        if indices.shape[1] != self.num_groups:
            raise RuntimeError(
                f"expected {self.num_groups} codebook groups, "
                f"got {indices.shape[1]}"
            )
        vectors = [
            self.codebook[group].index_select(
                0, indices[:, group].to(
                    device=self.codebook.device, dtype=torch.long
                )
            )
            for group in range(self.num_groups)
        ]
        return torch.cat(vectors, dim=-1).unsqueeze(0)

    def _decode_batches(
        self, batches: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for indices in batches:
            self.transfer_sizes.append(int(indices.shape[0]))
            waveform = self.vocoder.push(self._lookup(indices))
            if waveform is not None and waveform.numel() > 0:
                outputs.append(waveform)
        return outputs

    @torch.inference_mode()
    def push(self, samples: np.ndarray | torch.Tensor) -> np.ndarray:
        """Push mono 16 kHz float samples and return newly finalized 24 kHz audio."""
        if self.finished:
            raise RuntimeError("cannot push after flush")
        if isinstance(samples, np.ndarray):
            chunk = torch.from_numpy(
                np.ascontiguousarray(samples, dtype=np.float32)
            )
        else:
            chunk = samples
        if chunk.ndim != 1:
            raise ValueError("stream chunks must be one-dimensional mono audio")
        chunk = chunk.to(
            device=self.codebook.device, dtype=torch.float32
        ).view(1, 1, -1)
        outputs = self._decode_batches(self.encoder.push(chunk))
        if not outputs:
            return np.zeros(0, dtype=np.float32)
        return (
            torch.cat(outputs, dim=-1)
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def flush(self) -> np.ndarray:
        """Finalize the tail. A session may be flushed exactly once."""
        if self.finished:
            return np.zeros(0, dtype=np.float32)
        outputs = self._decode_batches(self.encoder.flush())
        tail = self.vocoder.flush()
        if tail.numel() > 0:
            outputs.append(tail)
        self.finished = True
        if not outputs:
            return np.zeros(0, dtype=np.float32)
        return (
            torch.cat(outputs, dim=-1)
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
        )
