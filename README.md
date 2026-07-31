# LSCodec inference

Inference-only, fixed/sliding-window streaming runtime for
[`Icerm/lscodec_25hz_v3`](https://huggingface.co/Icerm/lscodec_25hz_v3).

This directory is intended to become its own public repository. It contains:

- a stateful `push()` / `flush()` streaming API;
- an ordinary audio-file CLI;
- the inference-only WavLM model definition;
- tests for the streaming protocol and the public/private boundary.

It does **not** contain or import LSCodec training code, training configs,
datasets, experiment scripts, optimizer state, or raw training checkpoints.
The runtime downloads `codebook.npy` plus the preferred CUDA/CPU TorchScript set:

- `torchscript/lscodec_encoder.ts`;
- `torchscript/lscodec_prompt.ts`;
- `torchscript/lscodec_vocoder.ts`.

For compatibility it can fall back to the older CPU/mobile pair under `ptl/`.
All of these are inference-only exports; the larger research repository and its
Python model architecture are not needed.

## What “streaming” means

This follows the protocol in `lscodec/streaming/stream_recon_sliding.py`:

1. 16 kHz audio may arrive in arbitrary chunks.
2. The encoder uses bounded left context and right lookahead, then transfers
   `left + center` tokens for bootstrap and one `center` hop thereafter.
3. The non-causal vocoder repeatedly decodes a
   `left + center + lookahead` token window.
4. Finalized 24 kHz audio is emitted incrementally, with seam crossfading.

The default window is `320 ms left + 320 ms center + 160 ms lookahead`, with a
20 ms crossfade and 100 ms simulated input arrivals. This is real bounded-state
streaming, but it is **not a sample-causal codec**: the non-causal released
model intentionally waits for lookahead before finalizing each block.

## Install

Python 3.10 or newer is required. Install PyTorch appropriate for your CPU or
CUDA environment first, then install this project:

```bash
git clone https://github.com/xiaonengmiao/LSCodec-Inference.git
cd LSCodec-Inference
python -m pip install -e .
```

The vocoder uses WavLM-Large layer 6 for the speaker prompt. Download the
original `WavLM-Large.pt` from the
[official Microsoft WavLM release](https://github.com/microsoft/unilm/blob/master/wavlm/README.md#pre-trained-models)
and provide its path:

```bash
export LSCODEC_WAVLM_PATH=/absolute/path/to/WavLM-Large.pt
```

WavLM is a public base model and is deliberately not duplicated in the
LSCodec weight repository.

## File inference

Self reconstruction:

```bash
lscodec-stream input.wav reconstructed.wav
```

Cross-speaker reconstruction:

```bash
lscodec-stream \
  input.wav reconstructed.wav \
  --prompt reference_speaker.wav
```

The first command downloads the required inference artifacts from
`Icerm/lscodec_25hz_v3`. To use an already-downloaded model:

```bash
lscodec-stream input.wav reconstructed.wav \
  --model /path/to/lscodec_25hz_v3 \
  --wavlm /path/to/WavLM-Large.pt
```

The local model directory must contain `codebook.npy` and either all three
files under `torchscript/` (recommended, CPU and CUDA) or both files under
`ptl/` (legacy CPU/mobile fallback). The raw `.pt` files and YAML
training/research configs are not required.

Window controls mirror the private reference script:

```bash
lscodec-stream input.wav reconstructed.wav \
  --chunk-ms 320 \
  --left-ctx-ms 320 \
  --lookahead-ms 160 \
  --crossfade-ms 20 \
  --arrival-ms 100 \
  --prompt-anchor-ms 320
```

## Live Python API

`reconstruct()` runs the protocol over a completed recording. For a microphone,
socket, or other live source, keep one session and push mono float32 16 kHz
chunks as they arrive:

```python
import numpy as np

from lscodec_inference import LSCodecStreaming, StreamingConfig

codec = LSCodecStreaming.from_pretrained(
    "Icerm/lscodec_25hz_v3",
    wavlm_path="/absolute/path/to/WavLM-Large.pt",
    device="cuda",
)

# prompt_samples may come from any normal audio loader.
session = codec.create_session(
    prompt_samples,
    prompt_sample_rate=16000,
    config=StreamingConfig(),
)

for chunk_16khz in microphone_chunks:
    new_audio_24khz: np.ndarray = session.push(chunk_16khz)
    play_or_send(new_audio_24khz)

play_or_send(session.flush())
```

For an in-memory recording:

```python
output_24khz = codec.reconstruct(
    source_samples,
    source_sample_rate,
    prompt_audio=reference_samples,
    prompt_sample_rate=reference_sample_rate,
)
```

Each live session is single-utterance and single-use. Create a new session for
the next utterance.

## Publish boundary

Before publishing this directory, run:

```bash
python tools/audit_public_tree.py
pytest
```

The audit fails if a `.pt`, `.ptl`, `.ts`, `.pkl`, `.ckpt`, `.safetensors`, or `.npy`
file; a training/data directory; or an import of the private `lscodec` package
appears in the public tree. Weights remain versioned in the Hugging Face model
repository instead of this code repository.
