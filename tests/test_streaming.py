from __future__ import annotations

import numpy as np
import torch

from lscodec_inference.streaming import (
    FixedWindowEncoder,
    SlidingWindowVocoder,
    StreamingSession,
    number_of_frames,
)


class FakeEncoder:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        frames = number_of_frames(int(waveform.shape[-1]))
        return (
            torch.arange(frames, device=waveform.device)
            .remainder(32)
            .view(-1, 1)
        )


class FakeVocoder:
    def __call__(
        self, vectors: torch.Tensor, prompt: torch.Tensor
    ) -> torch.Tensor:
        del prompt
        # The release vocoder produces 480 samples per already-repeated
        # (50 Hz) vector.
        count = int(vectors.shape[1]) * 480
        values = vectors[..., :1].transpose(1, 2)
        return values.repeat_interleave(480, dim=-1)[..., :count]


def make_session() -> StreamingSession:
    center, left, right = 8, 8, 4
    encoder = FixedWindowEncoder(
        FakeEncoder(), center, left, right
    )
    vocoder = SlidingWindowVocoder(
        FakeVocoder(),
        torch.zeros(1, 80, 1024),
        center,
        left,
        right,
        crossfade_samples=480,
        repeat_input_tokens=True,
    )
    codebook = torch.arange(32, dtype=torch.float32).view(1, 32, 1)
    return StreamingSession(encoder, vocoder, codebook)


def run(arrival_samples: int) -> tuple[np.ndarray, list[int]]:
    session = make_session()
    source = np.linspace(-0.5, 0.5, 32_000, dtype=np.float32)
    outputs = [
        session.push(source[start:start + arrival_samples])
        for start in range(0, source.size, arrival_samples)
    ]
    outputs.append(session.flush())
    return np.concatenate(outputs), session.transfer_sizes


def test_streaming_is_independent_of_arrival_chunking():
    first, first_transfers = run(1_600)
    second, second_transfers = run(731)
    np.testing.assert_array_equal(first, second)
    assert first_transfers == second_transfers


def test_default_transfer_protocol_and_output_length():
    waveform, transfers = run(1_600)
    frames = number_of_frames(32_000)
    assert transfers == [16, 8, 8, 8, 8]
    assert sum(transfers) == frames == 48
    assert waveform.shape == (frames * 960,)


def test_short_audio_flushes_without_a_bootstrap_window():
    session = make_session()
    assert session.push(np.zeros(3_000, dtype=np.float32)).size == 0
    tail = session.flush()
    assert session.transfer_sizes == [2]
    assert tail.shape == (2 * 960,)
