"""Validate the frozen data contract and materialize small COCO subset files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path

from .catalog import PACKAGE, read_json, sha256, write_json


def prepare_data(data_root, variant, work_dir, splits=("train", "val", "test")):
    root = Path(data_root).expanduser().resolve()
    contract = read_json(PACKAGE / "assets/data_contract.json")
    subsets = {}
    for split in splits:
        expected = contract[variant][split]
        source = root / expected["annotation"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing {source}. See docs/data.md for the required export.")
        if sha256(source) != expected["sha256"]:
            raise ValueError(
                f"Annotation hash mismatch: {source}. This is not the frozen paper export."
            )
        coco = read_json(source)
        indices = read_json(PACKAGE / "assets/splits.json")[split]
        images = [coco["images"][i] for i in indices]
        ids = {im["id"] for im in images}
        h, w = expected["shape"]
        for im in images:
            if (im["height"], im["width"]) != (h, w):
                raise ValueError(f"Wrong geometry for image {im['id']}")
            image_path = root / im["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
        for sample in expected["tensor_samples"]:
            import torch

            tensor = torch.load(root / sample["file"], map_location="cpu", weights_only=True)
            values_digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            if values_digest != sample["values_sha256"]:
                raise ValueError(f"Reference tensor differs: {sample['file']}")
        subset = {
            **coco,
            "images": images,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        }
        target = Path(work_dir).resolve() / "annotations" / f"{split}.json"
        write_json(target, subset)
        subsets[split] = str(target)
    return subsets


def environment():
    names = (
        "torch",
        "torchvision",
        "mmcv",
        "mmengine",
        "mmdet",
        "mmpretrain",
        "ultralytics",
        "numpy",
        "opencv-python",
        "pycocotools",
        "transformers",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def receipt(preset, data_root, work_dir):
    write_json(
        Path(work_dir) / "reproduction.json",
        {
            "preset": preset["name"],
            "source": preset["source"],
            "seed": preset["seed"],
            "data_root": str(Path(data_root).expanduser().resolve()),
            "environment": environment(),
            "note": "Original MMDetection runs used deterministic=False; identical retraining is not guaranteed.",
        },
    )
