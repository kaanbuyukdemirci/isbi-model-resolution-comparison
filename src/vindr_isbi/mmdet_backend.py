"""Thin MMEngine runner around the fully resolved historical configurations."""

from copy import deepcopy
from pathlib import Path

from .catalog import runtime_config, write_json
from .data import prepare_data, receipt


def register():
    from .compat import ensure_mmdet_transformers_compat

    ensure_mmdet_transformers_compat()
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    from . import metric, transforms  # noqa: F401


def final_nms(cfg):
    test = cfg.model.get("test_cfg", {})
    final = test.get("rcnn", test)
    if "nms" in final:
        key = "iou_threshold" if "iou_threshold" in final.nms else "iou_thr"
        final.nms[key] = 0.5
        return False
    return True


def configuration(preset, data_root, work_dir, split=None):
    register()
    from mmengine.config import Config

    work_dir = Path(work_dir).resolve()
    cfg = Config(runtime_config(preset))
    cfg.work_dir = str(work_dir)
    cfg.launcher = "none"
    subsets = prepare_data(
        data_root, preset["variant"], work_dir, (split,) if split else ("train", "val")
    )
    for phase, annotation in subsets.items():
        dataset = cfg[f"{phase}_dataloader"]["dataset"]
        dataset["data_root"] = str(Path(data_root).expanduser().resolve())
        dataset["ann_file"] = annotation
        dataset.pop("indices", None)
        if phase != "train":
            cfg[f"{phase}_evaluator"]["ann_file"] = annotation
            cfg[f"{phase}_evaluator"]["outfile_prefix"] = str(work_dir / f"{phase}_predictions")
    if split:
        cfg.train_cfg = cfg.train_dataloader = cfg.optim_wrapper = cfg.param_scheduler = None
        cfg.custom_hooks = []
        for key in ("checkpoint", "sampler_seed", "param_scheduler"):
            cfg.default_hooks[key] = None
        if split == "test" and final_nms(cfg):
            cfg.test_evaluator.prediction_nms_iou_threshold = 0.5
        unused = "val" if split == "test" else "test"
        for suffix in ("cfg", "dataloader", "evaluator"):
            cfg[f"{unused}_{suffix}"] = None
    else:
        cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
    return cfg


def train(preset, data_root, work_dir, dry_run=False, resume=False):
    if resume:
        marker = Path(work_dir) / "last_checkpoint"
        if not marker.is_file() or not Path(marker.read_text().strip()).is_file():
            raise FileNotFoundError(
                "--resume requires an existing last_checkpoint and its checkpoint file."
            )
    cfg = configuration(preset, data_root, work_dir)
    from mmengine.runner import Runner, set_random_seed

    set_random_seed(preset["seed"], deterministic=False)
    cfg.resume = resume
    if resume:
        cfg.load_from = None
    receipt(preset, data_root, work_dir)
    cfg.dump(str(Path(work_dir) / "resolved_train_config.py"))
    if dry_run:
        print(f"Configuration and data verified: {work_dir}")
        return
    Runner.from_cfg(cfg).train()
    checkpoints = sorted(Path(work_dir).glob("best*.pth"))
    if not checkpoints:
        raise RuntimeError("Training ended without a validation-selected checkpoint.")
    print(f"Best checkpoint: {checkpoints[-1]}")


def evaluate(preset, data_root, work_dir, checkpoint, split):
    cfg = configuration(preset, data_root, work_dir, split=split)
    from mmengine.runner import Runner

    cfg.load_from = str(Path(checkpoint).expanduser().resolve())
    cfg.resume = False
    cfg.dump(str(Path(work_dir) / "resolved_evaluation_config.py"))
    runner = Runner.from_cfg(cfg)
    metrics = runner.test() if split == "test" else runner.val()
    write_json(Path(work_dir) / "metrics.json", metrics)
    print(metrics)


def load_detector(preset, checkpoint, device):
    register()
    from mmdet.apis import init_detector
    from mmengine.config import Config

    cfg = Config(runtime_config(preset))
    needs_posthoc_nms = final_nms(cfg)
    # Use the exact test pipeline but accept an already decoded float32 array.
    pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    pipeline[0] = {"type": "mmdet.LoadImageFromNDArray"}
    model = init_detector(cfg, str(checkpoint), device=device)
    from mmengine.dataset import Compose

    return model, Compose(pipeline), needs_posthoc_nms
