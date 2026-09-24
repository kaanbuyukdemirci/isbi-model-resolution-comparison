import csv
import json
from collections import defaultdict

import numpy as np
import pytest

from vindr_isbi.catalog import PACKAGE, catalog, recipe, runtime_config
from vindr_isbi.postprocess import class_aware_nms


def test_presets_match_validation_selection_from_complete_audit():
    groups = defaultdict(list)
    rows = list(csv.DictReader((PACKAGE / "assets/sweep_runs.csv").open()))
    assert len(rows) == 558
    for row in rows:
        if row["state"] == "finished" and row["validation_map50"] and row["test_map50"]:
            groups[(row["model"], int(row["height"]), int(row["width"]))].append(row)
    assert sum(map(len, groups.values())) == 534
    assert len(catalog()) == len(groups) == 107
    assert len({r["model"] for r in catalog().values()}) == 18
    for name, item in catalog().items():
        p = recipe(name)
        candidates = groups[(p["model"], p["height"], p["width"])]
        selected = max(candidates, key=lambda r: (float(r["validation_map50"]), r["run_id"]))
        assert item["run_id"] == p["source"]["run_id"] == selected["run_id"]


def test_default_is_actual_retry_recipe():
    p = recipe("paper-best")
    c = runtime_config(p)
    assert p["source"]["run_id"] == "3gs6jzhb"
    assert p["source"]["test_map50"] == 0.709
    assert c["train_dataloader"]["batch_size"] == 1
    assert c["optim_wrapper"]["accumulative_counts"] == 2
    assert c["optim_wrapper"]["optimizer"]["lr"] == 0.000025
    assert c["optim_wrapper"]["type"] == "OptimWrapper"
    assert c["randomness"] == dict(seed=42, deterministic=False, diff_rank_seed=False)
    assert c["train_dataloader"]["dataset"]["pipeline"][2]["scale"] == (1280, 1600)


def test_every_recipe_is_portable_and_complete():
    for name in catalog():
        p = recipe(name)
        serialized = json.dumps(p["config"])
        assert "/home/" not in serialized and "/mnt/" not in serialized
        assert "vindr_mmdet" not in serialized and "Wandb" not in serialized
        assert p["seed"] == 42
        assert len(p["source"]["checkpoint"]["sha256"]) == 64
        assert p["height"] % 32 == p["width"] % 32 == 0
    with pytest.raises(ValueError, match="Unknown preset"):
        recipe("dino_hrnet_w32_hrfpn-h1600_w1280")


def test_frozen_split_sizes():
    splits = json.loads((PACKAGE / "assets/splits.json").read_text())
    assert {k: len(v) for k, v in splits.items()} == {"train": 929, "val": 151, "test": 219}
    assert all(indices == sorted(set(indices)) for indices in splits.values())
    studies = json.loads((PACKAGE / "assets/study_splits.json").read_text())
    assert len(studies) == 5000
    assert sum(v == "test" for v in studies.values()) == 1000
    assert sum(v == "val" for v in studies.values()) == 600


def test_nms_is_class_aware_and_stable_for_ties():
    boxes = np.array(
        [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32
    )
    scores = np.array([0.8, 0.8, 0.9, 0.5], dtype=np.float32)
    labels = np.array([0, 0, 1, 0])
    assert class_aware_nms(boxes, scores, labels, 0.5).tolist() == [2, 0, 3]
    assert class_aware_nms(boxes[:0], scores[:0], labels[:0], 0.5).tolist() == []


def test_tensor_loading_preserves_float_values_and_rejects_wrong_geometry(tmp_path):
    torch = pytest.importorskip("torch")
    from vindr_isbi.inference import load_input

    tensor = torch.linspace(0, 1, 512 * 416).reshape(1, 512, 416)
    path = tmp_path / "image.pt"
    torch.save(tensor, path)
    image, metadata = load_input(path, recipe("fcos-h512_w416"))
    np.testing.assert_array_equal(image[..., 0], tensor.numpy()[0] * np.float32(255))
    np.testing.assert_array_equal(image[..., 0], image[..., 2])
    assert metadata["coordinate_space"] == "exported_image"
    with pytest.raises(ValueError, match="Expected float32 tensor at"):
        load_input(path, recipe("paper-best"))


def test_bad_annotation_hash_stops_before_training(tmp_path):
    from vindr_isbi.data import prepare_data

    target = tmp_path / "annotations/h512_w416/instances_train.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"images": []}')
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_data(tmp_path, "h512_w416", tmp_path / "run", splits=("train",))


def test_cli_help_and_catalog_need_no_framework_imports():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vindr_isbi.cli import main; import sys; main(['list', '--model', 'dino_convnext']); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dino_convnext_s_fpn-h1600_w1280" in result.stdout
