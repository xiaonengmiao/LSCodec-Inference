"""Minimal inference-only WavLM-Large layer-6 feature extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .wavlm import WavLM, WavLMConfig


class WavLMExtractor:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        output_layer: int = 6,
    ):
        self.device = torch.device(device)
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        self.config = WavLMConfig(payload["cfg"])
        self.model = WavLM(self.config)
        self.model.load_state_dict(payload["model"])
        self.model.eval().requires_grad_(False).to(self.device)
        self.output_layer = int(output_layer)

    @torch.inference_mode()
    def extract(self, waveform: np.ndarray) -> torch.Tensor:
        samples = torch.from_numpy(
            np.ascontiguousarray(waveform, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        if self.config.normalize:
            samples = torch.nn.functional.layer_norm(
                samples, samples.shape
            )
        representation = self.model.extract_features(
            samples, output_layer=self.output_layer
        )[0]
        return representation.squeeze(0).detach()
