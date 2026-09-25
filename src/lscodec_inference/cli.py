"""Command-line interface for fixed/sliding-window reconstruction."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from .model import BACKENDS, DEFAULT_MODEL_ID, LSCodecStreaming
from .streaming import StreamingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lscodec-stream",
        description=(
            "Inference-only LSCodec fixed-encoder/sliding-vocoder "
            "streaming reconstruction"
        ),
    )
    parser.add_argument("input", help="input audio file")
    parser.add_argument("output", help="output 24 kHz WAV file")
    parser.add_argument(
        "--prompt",
        help="reference speaker audio; defaults to the input (self prompt)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local release directory",
    )
    parser.add_argument(
        "--wavlm",
        help=(
            "official WavLM-Large.pt path; alternatively set "
            "LSCODEC_WAVLM_PATH"
        ),
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help=(
            "inference artifacts: torchscript, onnx (ONNX Runtime), or the "
            "legacy CPU-only lite set; auto prefers torchscript"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="ONNX Runtime intra-op threads (default: min(16, CPU count))",
    )
    parser.add_argument("--revision", help="Hugging Face revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk-ms", type=float, default=320.0)
    parser.add_argument("--left-ctx-ms", type=float, default=320.0)
    parser.add_argument("--lookahead-ms", type=float, default=160.0)
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--arrival-ms", type=float, default=100.0)
    parser.add_argument("--prompt-anchor-ms", type=float, default=320.0)
    parser.add_argument(
        "--verbose", action="store_true", help="enable progress logging"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    config = StreamingConfig(
        chunk_ms=args.chunk_ms,
        left_context_ms=args.left_ctx_ms,
        lookahead_ms=args.lookahead_ms,
        crossfade_ms=args.crossfade_ms,
        arrival_ms=args.arrival_ms,
        prompt_anchor_ms=args.prompt_anchor_ms,
    )
    logging.info(
        "Loading %s (%s backend) on %s", args.model, args.backend, args.device
    )
    codec = LSCodecStreaming.from_pretrained(
        args.model,
        wavlm_path=args.wavlm,
        device=args.device,
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
        backend=args.backend,
        onnx_threads=args.threads,
    )
    codec.reconstruct_file(
        args.input,
        args.output,
        prompt_path=args.prompt,
        config=config,
    )
    logging.info("Wrote %s at %d Hz", args.output, codec.output_sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
