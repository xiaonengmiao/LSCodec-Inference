"""ONNX Runtime adapter for the released LSCodec graphs.

The adapter exposes each ONNX session as a callable with the same tensor
interface as the TorchScript modules, so ``FixedWindowEncoder``,
``SlidingWindowVocoder``, and the prompt anchor run unchanged on either backend.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import torch


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
# ONNX Runtime defaults to one intra-op thread per physical core. For the
# streaming window sizes, pools above ~16 threads get slower, and on
# multi-socket servers the default is slower than TorchScript.
MAX_DEFAULT_THREADS = 16


def default_onnx_threads() -> int:
    return max(1, min(MAX_DEFAULT_THREADS, os.cpu_count() or 1))


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ImportError as error:
        raise ImportError(
            "the ONNX backend requires onnxruntime; install it with "
            "`pip install lscodec-inference[onnx]` (CPU) or "
            "`pip install lscodec-inference[onnx-gpu]` (CUDA)"
        ) from error
    return onnxruntime


def onnxruntime_installed() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def available_onnx_providers() -> list[str]:
    return list(_import_onnxruntime().get_available_providers())


def select_onnx_providers(
    device: str | torch.device, available: Sequence[str]
) -> list[str]:
    """Choose execution providers for ``device`` from ``available``."""
    if torch.device(device).type == "cuda":
        if CUDA_PROVIDER in available:
            return [CUDA_PROVIDER, CPU_PROVIDER]
        warnings.warn(
            "onnxruntime has no CUDAExecutionProvider; running the ONNX "
            "graphs on CPU. Install onnxruntime-gpu for CUDA inference.",
            RuntimeWarning,
            stacklevel=2,
        )
    return [CPU_PROVIDER]


class OnnxModule:
    """Run an ONNX Runtime session on torch tensors and return a torch tensor.

    Inputs are bound to the graph inputs in declaration order and passed as
    float32. The first graph output is returned on ``device``.
    """

    def __init__(self, session: Any, *, device: str | torch.device):
        self.session = session
        self.input_names = [item.name for item in session.get_inputs()]
        self.device = torch.device(device)

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str],
        *,
        providers: Sequence[str],
        device: str | torch.device,
        threads: Optional[int] = None,
    ) -> "OnnxModule":
        onnxruntime = _import_onnxruntime()
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = (
            default_onnx_threads() if threads is None else int(threads)
        )
        session = onnxruntime.InferenceSession(
            os.fspath(path),
            sess_options=session_options,
            providers=list(providers),
        )
        return cls(session, device=device)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != len(self.input_names):
            raise TypeError(
                f"expected {len(self.input_names)} inputs "
                f"({', '.join(self.input_names)}), got {len(inputs)}"
            )
        feeds = {
            name: np.ascontiguousarray(
                tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            for name, tensor in zip(self.input_names, inputs)
        }
        output = self.session.run(None, feeds)[0]
        return torch.from_numpy(np.ascontiguousarray(output)).to(self.device)
