from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_tree_has_no_model_or_training_checkpoints():
    forbidden_suffixes = {
        ".pt",
        ".ptl",
        ".ts",
        ".pkl",
        ".ckpt",
        ".safetensors",
        ".npy",
        ".onnx",
    }
    leaked = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    assert leaked == []


def test_public_runtime_does_not_import_private_package():
    imported = []
    for path in (ROOT / "src/lscodec_inference").glob("*.py"):
        text = path.read_text()
        if "from lscodec." in text or "import lscodec." in text:
            imported.append(path.name)
    assert imported == []


def test_training_directories_are_not_part_of_public_tree():
    forbidden = {"conf", "data", "datasets", "exp", "local", "train"}
    leaked = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_dir() and path.name.lower() in forbidden
    ]
    assert leaked == []
