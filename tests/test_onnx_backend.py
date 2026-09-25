from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lscodec_inference.model import (
    DOWNLOAD_FILES,
    ONNX_FILES,
    TORCHSCRIPT_FILES,
    available_backends,
    choose_backend,
    download_patterns,
    resolve_model_directory,
)
from lscodec_inference.onnx_backend import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    MAX_DEFAULT_THREADS,
    OnnxModule,
    default_onnx_threads,
    select_onnx_providers,
)
from lscodec_inference.streaming import (
    FixedWindowEncoder,
    SlidingWindowVocoder,
    StreamingSession,
    number_of_frames,
)


ANCHOR_SAMPLES = 5_120


class FakeSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``."""

    def __init__(self, input_names, function):
        self._inputs = [SimpleNamespace(name=name) for name in input_names]
        self._function = function
        self.calls: list[dict[str, np.ndarray]] = []

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, feeds):
        assert output_names is None
        self.calls.append(feeds)
        return [self._function(*(feeds[item.name] for item in self._inputs))]


def numpy_encoder(waveform: np.ndarray) -> np.ndarray:
    frames = number_of_frames(int(waveform.shape[-1]))
    return (np.arange(frames, dtype=np.int64) % 32).reshape(-1, 1)


def numpy_vocoder(vectors: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    del prompt
    return np.repeat(vectors[..., :1].transpose(0, 2, 1), 480, axis=-1)


class TorchEncoder:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(numpy_encoder(waveform.numpy()))


class TorchVocoder:
    def __call__(self, vectors: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(numpy_vocoder(vectors.numpy(), prompt.numpy()))


def make_session(encoder, vocoder) -> StreamingSession:
    center, left, right = 8, 8, 4
    anchor = torch.linspace(-0.2, 0.2, ANCHOR_SAMPLES).view(1, 1, -1)
    return StreamingSession(
        FixedWindowEncoder(
            encoder, center, left, right, normalization_anchor=anchor
        ),
        SlidingWindowVocoder(
            vocoder,
            torch.zeros(1, 80, 184),
            center,
            left,
            right,
            crossfade_samples=480,
        ),
        torch.arange(32, dtype=torch.float32).view(1, 32, 1),
    )


def stream(session: StreamingSession, arrival: int) -> np.ndarray:
    source = np.linspace(-0.5, 0.5, 32_000, dtype=np.float32)
    outputs = [
        session.push(source[start:start + arrival])
        for start in range(0, source.size, arrival)
    ]
    outputs.append(session.flush())
    return np.concatenate(outputs)


def test_onnx_module_binds_inputs_in_declaration_order():
    session = FakeSession(["vq", "prompt_cache"], lambda vq, prompt: vq - prompt)
    module = OnnxModule(session, device="cpu")
    vq = torch.full((1, 3, 2), 5.0, dtype=torch.float64)
    prompt = torch.ones(1, 3, 2)
    output = module(vq, prompt)
    assert output.dtype == torch.float32
    assert torch.equal(output, torch.full((1, 3, 2), 4.0))
    fed = session.calls[0]
    assert list(fed) == ["vq", "prompt_cache"]
    assert all(array.dtype == np.float32 for array in fed.values())


def test_onnx_module_keeps_integer_outputs_and_checks_arity():
    module = OnnxModule(FakeSession(["waveform"], numpy_encoder), device="cpu")
    indices = module(torch.zeros(1, 1, 14_025))
    assert indices.dtype == torch.int64
    assert tuple(indices.shape) == (number_of_frames(14_025), 1)
    with pytest.raises(TypeError, match="expected 1 inputs"):
        module(torch.zeros(1, 1, 10), torch.zeros(1))


def test_anchored_streaming_is_identical_on_onnx_and_torch_callables():
    encoder_session = FakeSession(["waveform"], numpy_encoder)
    onnx_session = make_session(
        OnnxModule(encoder_session, device="cpu"),
        OnnxModule(FakeSession(["vq", "prompt_cache"], numpy_vocoder), device="cpu"),
    )
    torch_session = make_session(TorchEncoder(), TorchVocoder())

    onnx_output = stream(onnx_session, 1_600)
    np.testing.assert_array_equal(onnx_output, stream(torch_session, 1_600))
    assert onnx_session.transfer_sizes == torch_session.transfer_sizes
    assert onnx_output.shape == (number_of_frames(32_000) * 960,)

    # Every encoder call sees its window with the prompt anchor appended.
    lengths = [call["waveform"].shape[-1] for call in encoder_session.calls]
    assert lengths and all(length > ANCHOR_SAMPLES for length in lengths)
    tails = {
        tuple(call["waveform"][0, 0, -ANCHOR_SAMPLES:][[0, -1]])
        for call in encoder_session.calls
    }
    assert tails == {(np.float32(-0.2), np.float32(0.2))}


def test_provider_selection():
    assert select_onnx_providers("cpu", [CUDA_PROVIDER, CPU_PROVIDER]) == [
        CPU_PROVIDER
    ]
    assert select_onnx_providers("cuda", [CUDA_PROVIDER, CPU_PROVIDER]) == [
        CUDA_PROVIDER,
        CPU_PROVIDER,
    ]
    with pytest.warns(RuntimeWarning, match="CUDAExecutionProvider"):
        assert select_onnx_providers("cuda:0", [CPU_PROVIDER]) == [CPU_PROVIDER]


def test_default_thread_pool_is_capped(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 224)
    assert default_onnx_threads() == MAX_DEFAULT_THREADS
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert default_onnx_threads() == 4
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert default_onnx_threads() == 1


def _touch_release(root: Path, files) -> Path:
    for relative in ("codebook.npy", *files):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


def test_backend_selection_for_onnx_only_release(tmp_path):
    release = _touch_release(tmp_path, ONNX_FILES)
    assert available_backends(release) == ["onnx"]
    assert choose_backend("auto", ["onnx"]) == "onnx"
    assert choose_backend("auto", ["onnx", "torchscript"]) == "torchscript"
    assert resolve_model_directory(release, backend="onnx") == release.resolve()
    with pytest.raises(FileNotFoundError, match="torchscript"):
        resolve_model_directory(release, backend="torchscript")
    with pytest.raises(ValueError, match="unknown backend"):
        choose_backend("tensorrt", ["onnx"])


def test_download_patterns_fetch_only_the_requested_backend():
    assert download_patterns("auto") == list(DOWNLOAD_FILES)
    assert not any(name.endswith(".onnx") for name in download_patterns("auto"))
    assert download_patterns("onnx") == ["codebook.npy", *ONNX_FILES]
    assert download_patterns("torchscript") == ["codebook.npy", *TORCHSCRIPT_FILES]


@pytest.mark.skipif(
    not (
        os.environ.get("LSCODEC_TEST_MODEL_DIR")
        and os.environ.get("LSCODEC_WAVLM_PATH")
    ),
    reason="set LSCODEC_TEST_MODEL_DIR (with torchscript/ and onnx/) and "
    "LSCODEC_WAVLM_PATH to compare real weights",
)
def test_real_weights_onnx_matches_torchscript():
    pytest.importorskip("onnxruntime")
    from lscodec_inference import LSCodecStreaming

    model_dir = os.environ["LSCODEC_TEST_MODEL_DIR"]
    wavlm = os.environ["LSCODEC_WAVLM_PATH"]
    time = np.arange(48_000, dtype=np.float32) / 16_000
    audio = (
        0.3 * np.sin(2 * np.pi * (180 + 60 * time) * time)
        * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * time))
    ).astype(np.float32)

    outputs = {}
    for backend in ("torchscript", "onnx"):
        codec = LSCodecStreaming(
            model_dir, wavlm_path=wavlm, device="cpu", backend=backend
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs[backend] = codec.reconstruct(audio, 16_000)
        del codec

    reference, candidate = outputs["torchscript"], outputs["onnx"]
    assert reference.shape == candidate.shape
    correlation = np.corrcoef(reference, candidate)[0, 1]
    assert correlation > 0.99, correlation
