#!/usr/bin/env python3
"""Fail if private training artifacts leaked into the publishable tree."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".ptl",
    ".ts",
    ".pkl",
    ".ckpt",
    ".safetensors",
    ".npy",
    ".onnx",
}
FORBIDDEN_DIRECTORIES = {
    "conf",
    "data",
    "datasets",
    "exp",
    "local",
    "train",
}


def main() -> int:
    problems: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_dir() and path.name.lower() in FORBIDDEN_DIRECTORIES:
            problems.append(f"private directory: {relative}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"weight/checkpoint file: {relative}")
        if (
            path.is_file()
            and path.suffix == ".py"
            and (ROOT / "src/lscodec_inference") in path.parents
        ):
            text = path.read_text()
            if "from lscodec." in text or "import lscodec." in text:
                problems.append(f"private package import: {relative}")
    if problems:
        print("Public release audit failed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print("Public release audit passed: code only, no private imports or weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
