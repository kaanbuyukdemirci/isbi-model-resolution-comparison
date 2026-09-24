"""YOLO11 training and evaluation with the recorded, fully resolved arguments."""

from contextlib import contextmanager
from pathlib import Path

import yaml

from .catalog import write_json
from .data import prepare_data, receipt


@contextmanager
def console_callbacks():
    """Keep the console repository independent of globally enabled trackers."""
    from ultralytics.utils import callbacks

    original = callbacks.add_integration_callbacks
    callbacks.add_integration_callbacks = lambda owner: None
    try:
        yield
    finally:
        callbacks.add_integration_callbacks = original


def data_yaml(preset, root, work_dir):
    subsets = prepare_data(root, preset["variant"], work_dir)
    data = {
        "path": str(Path(root).expanduser().resolve()),
        **subsets,
        "names": {0: "mass"},
        "global_pad_multires": {
            "enabled": True,
            "root": str(Path(root).expanduser().resolve()),
            "variant": preset["variant"],
            "height": preset["height"],
            "width": preset["width"],
            "positive_only_validation": True,
            "positive_only_test": True,
        },
    }
    target = Path(work_dir).resolve() / "data.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(target)


def trainer_class(preset):
    from .yolo_runtime import make_random_erasing_trainer

    return make_random_erasing_trainer(
        probability=0.0,
        checkpoint_metric=preset["checkpoint_metric"],
        gradient_clip_max_norm=preset["gradient_clip_max_norm"],
    )


def train(preset, data_root, work_dir, dry_run=False, resume=False):
    work = Path(work_dir).resolve()
    options = dict(preset["config"])
    options.update(
        data=data_yaml(preset, data_root, work),
        project=str(work.parent),
        name=str(work),
        exist_ok=True,
        device="0",
    )
    receipt(preset, data_root, work)
    write_json(work / "resolved_train_args.json", options)
    if dry_run:
        print(f"Configuration and data verified: {work}")
        return
    from .compat import ensure_mmdet_transformers_compat

    ensure_mmdet_transformers_compat()
    from ultralytics import YOLO

    model = options.pop("model")
    if resume:
        model = str(work / "weights/last.pt")
        if not Path(model).is_file():
            raise FileNotFoundError(model)
        options["resume"] = True
    with console_callbacks():
        YOLO(model).train(trainer=trainer_class(preset), **options)
    print(f"Best checkpoint: {work / 'weights/best.pt'}")


def evaluate(preset, data_root, work_dir, checkpoint, split):
    from .compat import ensure_mmdet_transformers_compat

    ensure_mmdet_transformers_compat()
    from ultralytics import YOLO

    trainer = trainer_class(preset)
    with console_callbacks():
        result = YOLO(str(checkpoint)).val(
            validator=trainer.mass_positive_validator_class,
            data=data_yaml(preset, data_root, work_dir),
            split=split,
            imgsz=preset["height"],
            batch=preset["config"]["batch"],
            device="0",
            project=str(Path(work_dir).resolve()),
            name="evaluation",
            exist_ok=True,
            plots=False,
            conf=0.001,
            iou=0.5 if split == "test" else preset["config"]["iou"],
        )
    names = {
        "metrics/precision(B)": "metrics/precision_mass_positive(B)",
        "metrics/recall(B)": "metrics/recall_mass_positive(B)",
        "metrics/mAP50(B)": "metrics/mAP50_mass_positive(B)",
        "metrics/mAP50-95(B)": "metrics/mAP50-95_mass_positive(B)",
    }
    metrics = {names.get(str(k), str(k)): float(v) for k, v in result.results_dict.items()}
    write_json(Path(work_dir) / "metrics.json", metrics)
    print(metrics)
