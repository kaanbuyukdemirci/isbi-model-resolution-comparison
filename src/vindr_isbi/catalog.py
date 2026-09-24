"""Static validation-selected recipes; no search or remote tracking dependency."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DEFAULT_PRESET = "dino_convnext_s_fpn-h1600_w1280"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog():
    return read_json(PACKAGE / "assets/catalog.json")


def recipe(name=DEFAULT_PRESET):
    if name == "paper-best":
        name = DEFAULT_PRESET
    if name not in catalog():
        raise ValueError(f"Unknown preset {name!r}; run `vindr list`.")
    return read_json(PACKAGE / "presets" / f"{name}.json")


def runtime_config(preset):
    """Restore Python tuples required by MMCV without executable config files."""
    config = deepcopy(preset["config"])
    for path in sorted(preset.get("tuple_paths", []), key=len, reverse=True):
        parent = config
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = tuple(parent[path[-1]])
    return config
