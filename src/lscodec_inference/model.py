"""High-level model loading and streaming reconstruction API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import resampy
import soundfile as sf
import torch
from torch.jit.mobile import _load_for_lite_interpreter

from .onnx_backend import (
    CUDA_PROVIDER,
    OnnxModule,
    available_onnx_providers,
    onnxruntime_installed,
    select_onnx_providers,
)
from .streaming import (
    ENCODER_CONVOLUTIONS,
    FixedWindowEncoder,
    SlidingWindowVocoder,
    StreamingConfig,
    StreamingSession,
    convolution_geometry,
)
from .wavlm_extractor import WavLMExtractor


DEFAULT_MODEL_ID = "Icerm/lscodec_25hz_v3"
TORCHSCRIPT_FILES = (
    "torchscript/lscodec_encoder.ts",
    "torchscript/lscodec_prompt.ts",
    "torchscript/lscodec_vocoder.ts",
)
ONNX_FILES = (
    "onnx/lscodec_encoder.onnx",
    "onnx/lscodec_prompt.onnx",
    "onnx/lscodec_vocoder.onnx",
)
LITE_FILES = (
    "ptl/lscodec_encoder.ptl",
    "ptl/lscodec_vocoder.ptl",
)
# Artifact sets in ``backend="auto"`` preference order.
ARTIFACT_SETS = {
    "torchscript": TORCHSCRIPT_FILES,
    "onnx": ONNX_FILES,
    "lite": LITE_FILES,
}
BACKENDS = ("auto",) + tuple(ARTIFACT_SETS)
DOWNLOAD_FILES = ("codebook.npy",) + TORCHSCRIPT_FILES + LITE_FILES
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
TOKEN_RATE = 25.0
VOCODER_SAMPLES_PER_REPEATED_TOKEN = 480


def _looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or os.path.sep * 2 in value


def _check_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}"
        )
    return backend


def download_patterns(backend: str = "auto") -> list[str]:
    """Files to fetch from Hugging Face for ``backend``.

    ``auto`` keeps the historical TorchScript + legacy ``.ptl`` download; the
    ONNX set is fetched only when requested explicitly.
    """
    _check_backend(backend)
    if backend == "auto":
        return list(DOWNLOAD_FILES)
    return ["codebook.npy", *ARTIFACT_SETS[backend]]


def available_backends(model_dir: str | os.PathLike[str]) -> list[str]:
    """Backends whose complete artifact set exists in ``model_dir``."""
    root = Path(model_dir)
    return [
        name
        for name, files in ARTIFACT_SETS.items()
        if all((root / relative).is_file() for relative in files)
    ]


def choose_backend(requested: str, available: Sequence[str]) -> str:
    """Pick the backend to load, preferring TorchScript, then ONNX, then Lite."""
    _check_backend(requested)
    if requested == "auto":
        for name in ARTIFACT_SETS:
            if name in available:
                return name
        raise FileNotFoundError(
            "no complete artifact set found; expected one of: "
            + "; ".join(", ".join(files) for files in ARTIFACT_SETS.values())
        )
    if requested not in available:
        raise FileNotFoundError(
            f"backend {requested!r} needs {', '.join(ARTIFACT_SETS[requested])}"
        )
    return requested


def resolve_model_directory(
    model_name_or_path: str | os.PathLike[str],
    *,
    cache_dir: Optional[str | os.PathLike[str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    backend: str = "auto",
) -> Path:
    """Resolve a local release directory or a Hugging Face model ID."""
    _check_backend(backend)
    value = os.fspath(model_name_or_path)
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        model_dir = candidate.resolve()
    elif candidate.exists():
        raise NotADirectoryError(
            f"model path is not a directory: {candidate}"
        )
    elif _looks_like_local_path(value):
        raise FileNotFoundError(
            f"model directory does not exist: {candidate}"
        )
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "Hugging Face model IDs require huggingface-hub"
            ) from error
        model_dir = Path(
            snapshot_download(
                repo_id=value,
                allow_patterns=download_patterns(backend),
                cache_dir=(
                    os.fspath(cache_dir)
                    if cache_dir is not None
                    else None
                ),
                revision=revision,
                local_files_only=local_files_only,
            )
        ).resolve()

    if not (model_dir / "codebook.npy").is_file():
        raise FileNotFoundError(
            f"incomplete LSCodec release at {model_dir}; missing: codebook.npy"
        )
    try:
        choose_backend(backend, available_backends(model_dir))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"incomplete LSCodec release at {model_dir}: {error}"
        ) from None
    return model_dir


def as_mono_float32(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    """Normalize common audio array layouts to contiguous mono float32."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio)
    if array.ndim == 0 or array.ndim > 2 or array.size == 0:
        raise ValueError(
            "audio must be non-empty with shape (samples,), "
            "(samples, channels), or (channels, samples)"
        )
    if np.issubdtype(array.dtype, np.integer):
        limits = np.iinfo(array.dtype)
        scale = float(max(abs(limits.min), limits.max))
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32, copy=False)
    if array.ndim == 2:
        channel_axis = 0 if array.shape[0] <= 8 < array.shape[1] else 1
        array = array.mean(axis=channel_axis)
    if not np.isfinite(array).all():
        raise ValueError("audio contains NaN or infinite samples")
    return np.ascontiguousarray(array, dtype=np.float32)


def resample(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    output = resampy.resample(
        audio,
        source_rate,
        target_rate,
        filter="kaiser_best",
    )
    # Match librosa.resample(..., fix=True), which is used by the private
    # streaming reference: Resampy floors the raw length while Librosa pads
    # or trims it to ceil(input_samples * target_rate / source_rate).
    expected = int(np.ceil(audio.shape[-1] * target_rate / source_rate))
    if output.shape[-1] < expected:
        output = np.pad(output, (0, expected - output.shape[-1]))
    else:
        output = output[:expected]
    return np.ascontiguousarray(output, dtype=np.float32)


def load_audio(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        os.fspath(path), dtype="float32", always_2d=False
    )
    return as_mono_float32(audio), int(sample_rate)


def write_audio(
    path: str | os.PathLike[str],
    audio: np.ndarray | torch.Tensor,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output,
        as_mono_float32(audio),
        int(sample_rate),
        subtype="PCM_16",
    )
    return output


def select_prompt_anchor(
    audio: np.ndarray, sample_rate: int, duration_ms: float
) -> Optional[np.ndarray]:
    """Select the highest-energy prompt segment used for GroupNorm anchoring."""
    if duration_ms <= 0:
        return None
    count = max(1, int(round(duration_ms / 1000.0 * sample_rate)))
    waveform = as_mono_float32(audio)
    if waveform.size < count:
        return np.pad(waveform, (0, count - waveform.size))
    starts = range(0, waveform.size - count + 1, count)
    best = max(
        starts,
        key=lambda start: float(
            np.mean(
                waveform[start:start + count].astype(np.float64) ** 2
            )
        ),
    )
    return waveform[best:best + count].copy()


class LSCodecStreaming:
    """Released LSCodec streaming runtime without training-code imports."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        *,
        wavlm_path: str | os.PathLike[str],
        device: str | torch.device = "auto",
        backend: str = "auto",
        onnx_providers: Optional[Sequence[str]] = None,
        onnx_threads: Optional[int] = None,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        available = available_backends(self.model_dir)
        if backend == "auto" and not onnxruntime_installed():
            available = [name for name in available if name != "onnx"]
        self.artifact_format = choose_backend(backend, available)
        if str(device) == "auto":
            if self.artifact_format == "torchscript":
                use_cuda = torch.cuda.is_available()
            elif self.artifact_format == "onnx":
                use_cuda = (
                    torch.cuda.is_available()
                    and CUDA_PROVIDER in available_onnx_providers()
                )
            else:
                use_cuda = False
            device = "cuda" if use_cuda else "cpu"
        if (
            self.artifact_format == "lite"
            and torch.device(device).type != "cpu"
        ):
            raise ValueError(
                "the legacy .ptl artifacts are CPU/mobile-only; use the "
                "torchscript/ or onnx/ release set for CUDA inference or "
                "pass device='cpu'"
            )
        self.device = torch.device(device)
        self.onnx_providers: Optional[list[str]] = None

        if self.artifact_format == "onnx":
            providers = (
                list(onnx_providers)
                if onnx_providers is not None
                else select_onnx_providers(
                    self.device, available_onnx_providers()
                )
            )
            self.onnx_providers = providers
            self.encoder, self.prompt_encoder, self.vocoder = (
                OnnxModule.from_path(
                    self.model_dir / relative,
                    providers=providers,
                    device=self.device,
                    threads=onnx_threads,
                )
                for relative in ONNX_FILES
            )
        elif self.artifact_format == "torchscript":
            self.encoder = torch.jit.load(
                str(self.model_dir / "torchscript/lscodec_encoder.ts"),
                map_location=self.device,
            ).eval()
            self.prompt_encoder = torch.jit.load(
                str(self.model_dir / "torchscript/lscodec_prompt.ts"),
                map_location=self.device,
            ).eval()
            self.vocoder = torch.jit.load(
                str(self.model_dir / "torchscript/lscodec_vocoder.ts"),
                map_location=self.device,
            ).eval()
        else:
            self.prompt_encoder = None
            self.encoder = _load_for_lite_interpreter(
                str(self.model_dir / "ptl/lscodec_encoder.ptl"),
                map_location=self.device,
            )
            self.vocoder = _load_for_lite_interpreter(
                str(self.model_dir / "ptl/lscodec_vocoder.ptl"),
                map_location=self.device,
            )
        codebook = np.load(
            self.model_dir / "codebook.npy", allow_pickle=False
        )
        self.codebook = torch.from_numpy(codebook).to(
            device=self.device, dtype=torch.float32
        )
        if self.codebook.ndim == 2:
            self.codebook = self.codebook.unsqueeze(0)
        if self.codebook.ndim != 3:
            raise ValueError(
                "codebook.npy must have shape (groups, entries, dimension)"
            )

        self.wavlm_path = Path(wavlm_path).expanduser().resolve()
        if not self.wavlm_path.is_file():
            raise FileNotFoundError(
                f"WavLM-Large checkpoint not found: {self.wavlm_path}"
            )
        self.wavlm = WavLMExtractor(
            self.wavlm_path, device=self.device, output_layer=6
        )
        self.input_sample_rate = INPUT_SAMPLE_RATE
        self.output_sample_rate = OUTPUT_SAMPLE_RATE
        self.token_rate = TOKEN_RATE

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike[str] = DEFAULT_MODEL_ID,
        *,
        wavlm_path: Optional[str | os.PathLike[str]] = None,
        device: str | torch.device = "auto",
        cache_dir: Optional[str | os.PathLike[str]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        backend: str = "auto",
        onnx_providers: Optional[Sequence[str]] = None,
        onnx_threads: Optional[int] = None,
    ) -> "LSCodecStreaming":
        """Load the release artifacts and the official WavLM checkpoint."""
        model_dir = resolve_model_directory(
            model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            backend=backend,
        )
        resolved_wavlm = (
            Path(wavlm_path).expanduser()
            if wavlm_path is not None
            else Path(
                os.environ.get(
                    "LSCODEC_WAVLM_PATH",
                    model_dir / "WavLM-Large.pt",
                )
            ).expanduser()
        )
        return cls(
            model_dir,
            wavlm_path=resolved_wavlm,
            device=device,
            backend=backend,
            onnx_providers=onnx_providers,
            onnx_threads=onnx_threads,
        )

    def prepare_audio(
        self, audio: np.ndarray | torch.Tensor, sample_rate: int
    ) -> np.ndarray:
        return resample(
            as_mono_float32(audio),
            int(sample_rate),
            self.input_sample_rate,
        )

    @torch.inference_mode()
    def extract_prompt(
        self, audio_16khz: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        prompt = as_mono_float32(audio_16khz)
        return (
            self.wavlm.extract(prompt)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

    @torch.inference_mode()
    def create_session(
        self,
        prompt_audio: np.ndarray | torch.Tensor,
        prompt_sample_rate: int,
        *,
        config: StreamingConfig = StreamingConfig(),
    ) -> StreamingSession:
        """Create one stateful session. Subsequent chunks must already be 16 kHz."""
        prompt_16k = self.prepare_audio(
            prompt_audio, prompt_sample_rate
        )
        prompt_features = self.extract_prompt(prompt_16k)
        prompt_for_vocoder = (
            self.prompt_encoder(prompt_features)
            if self.prompt_encoder is not None
            else prompt_features
        )
        anchor = select_prompt_anchor(
            prompt_16k,
            self.input_sample_rate,
            config.prompt_anchor_ms,
        )
        anchor_tensor = (
            torch.from_numpy(anchor)
            .to(device=self.device, dtype=torch.float32)
            .view(1, 1, -1)
            if anchor is not None
            else None
        )
        center, left, right = config.frame_counts(self.token_rate)
        encoder = FixedWindowEncoder(
            self.encoder,
            center,
            left,
            right,
            normalization_anchor=anchor_tensor,
            convolutions=ENCODER_CONVOLUTIONS,
        )
        vocoder = SlidingWindowVocoder(
            self.vocoder,
            prompt_for_vocoder,
            center,
            left,
            right,
            crossfade_samples=config.crossfade_samples(
                self.output_sample_rate
            ),
            repeat_input_tokens=True,
            samples_per_repeated_token=(
                VOCODER_SAMPLES_PER_REPEATED_TOKEN
            ),
        )
        return StreamingSession(encoder, vocoder, self.codebook)

    def reconstruct(
        self,
        audio: np.ndarray | torch.Tensor,
        sample_rate: int,
        *,
        prompt_audio: Optional[np.ndarray | torch.Tensor] = None,
        prompt_sample_rate: Optional[int] = None,
        config: StreamingConfig = StreamingConfig(),
    ) -> np.ndarray:
        """Run the same incremental protocol over an in-memory recording."""
        waveform = self.prepare_audio(audio, sample_rate)
        if prompt_audio is None:
            prompt_audio = waveform
            prompt_sample_rate = self.input_sample_rate
        elif prompt_sample_rate is None:
            raise ValueError(
                "prompt_sample_rate is required with prompt_audio"
            )
        session = self.create_session(
            prompt_audio,
            int(prompt_sample_rate),
            config=config,
        )
        arrival = config.arrival_samples(self.input_sample_rate)
        outputs = [
            session.push(waveform[start:start + arrival])
            for start in range(0, waveform.size, arrival)
        ]
        outputs.append(session.flush())
        nonempty = [part for part in outputs if part.size > 0]
        return (
            np.concatenate(nonempty)
            if nonempty
            else np.zeros(0, dtype=np.float32)
        )

    def reconstruct_file(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        prompt_path: Optional[str | os.PathLike[str]] = None,
        config: StreamingConfig = StreamingConfig(),
    ) -> Path:
        audio, sample_rate = load_audio(input_path)
        if prompt_path is None:
            prompt_audio, prompt_rate = None, None
        else:
            prompt_audio, prompt_rate = load_audio(prompt_path)
        output = self.reconstruct(
            audio,
            sample_rate,
            prompt_audio=prompt_audio,
            prompt_sample_rate=prompt_rate,
            config=config,
        )
        return write_audio(
            output_path, output, self.output_sample_rate
        )
