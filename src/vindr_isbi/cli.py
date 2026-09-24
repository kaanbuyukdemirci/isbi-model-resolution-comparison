"""One model per command. The catalog is static and works offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .catalog import DEFAULT_PRESET, catalog, recipe, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vindr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="List the frozen validation-selected presets")
    ls.add_argument("--model", help="Filter by model name substring")
    show = sub.add_parser("show", help="Print a complete recipe and its provenance")
    show.add_argument("--preset", default=DEFAULT_PRESET)
    prep = sub.add_parser(
        "prepare", help="Prepare just the paper cohort at one resolution from raw DICOM"
    )
    prep.add_argument("--preset", default=DEFAULT_PRESET)
    prep.add_argument("--raw-root", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--workers", type=int, default=4)
    for command in ("train", "evaluate", "infer", "verify-checkpoint"):
        p = sub.add_parser(command)
        p.add_argument(
            "--preset", default=DEFAULT_PRESET, help="Preset name; paper-best is an alias"
        )
        if command != "verify-checkpoint":
            p.add_argument(
                "--device",
                default="cuda:0",
                help="One CUDA device, e.g. cuda:0; inference also supports cpu",
            )
        if command in ("train", "evaluate"):
            p.add_argument("--data-root", required=True, type=Path)
            p.add_argument("--work-dir", required=True, type=Path)
        if command == "train":
            p.add_argument(
                "--dry-run",
                action="store_true",
                help="Verify data and save the resolved config without training",
            )
            p.add_argument(
                "--resume", action="store_true", help="Resume this preset's existing work directory"
            )
        if command in ("evaluate", "infer", "verify-checkpoint"):
            p.add_argument("--checkpoint", required=True, type=Path)
        if command == "evaluate":
            p.add_argument("--split", choices=("val", "test"), default="test")
        if command == "infer":
            p.add_argument(
                "--input", required=True, nargs="+", help="Float32 .pt, DICOM, or directory"
            )
            p.add_argument("--output", type=Path, default=Path("predictions.json"))
            p.add_argument("--score-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print(f"{'PRESET':52} {'VAL AP50':>9} {'TEST AP50':>10} {'LR':>10} {'BATCH×ACC':>10}")
            for name, item in catalog().items():
                if args.model and args.model not in name:
                    continue
                params = item["parameters"]
                print(
                    f"{name:52} {item['validation_map50']:9.3f} {item['test_map50']:10.3f} "
                    f"{params['learning_rate']:10g} {params['batch_size']:4}×{params['accumulation']:<4}"
                )
            return
        preset = recipe(args.preset)
        if args.command == "show":
            print(json.dumps(preset, indent=2))
            return
        if args.command == "prepare":
            if args.workers < 1:
                raise ValueError("workers must be positive")
            from .prepare import prepare

            prepare(preset, args.raw_root, args.output_root, args.workers)
            return
        if hasattr(args, "checkpoint"):
            args.checkpoint = args.checkpoint.expanduser().resolve()
            if not args.checkpoint.is_file():
                raise FileNotFoundError(args.checkpoint)
        if args.command == "verify-checkpoint":
            expected = preset["source"]["checkpoint"]["sha256"]
            actual = sha256(args.checkpoint)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint differs from the historical artifact. Expected {expected}; got {actual}. Retrained weights will have a different hash."
                )
            print("Checkpoint matches the historical paper artifact.")
            return
        if args.device != "cpu":
            device_index = args.device.removeprefix("cuda:")
            if not device_index.isdigit():
                raise ValueError("Use one CUDA device, e.g. cuda:0.")
            # Set before importing either ML backend. Logical GPU zero then maps to this device.
            os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            args.device = "cuda:0"
        elif args.command != "infer":
            raise ValueError(
                "Paper training/evaluation uses CUDA. CPU is supported for inference only."
            )
        if args.command == "infer":
            if not 0 <= args.score_threshold <= 1:
                raise ValueError("score-threshold must be between 0 and 1")
            from .inference import infer

            infer(
                preset, args.checkpoint, args.input, args.output, args.device, args.score_threshold
            )
            return
        args.work_dir = args.work_dir.expanduser().resolve()
        if args.command == "train" and args.work_dir.exists() and any(args.work_dir.iterdir()):
            if args.resume:
                from .catalog import read_json

                record = read_json(args.work_dir / "reproduction.json")
                if record["preset"] != preset["name"]:
                    raise ValueError("Cannot resume a different preset in this work directory.")
            else:
                raise ValueError(
                    "Use an empty work directory or --resume for an existing training run."
                )
        args.work_dir.mkdir(parents=True, exist_ok=True)
        if preset["backend"] == "mmdet":
            from . import mmdet_backend as backend
        else:
            from . import yolo_backend as backend
        if args.command == "train":
            backend.train(preset, args.data_root, args.work_dir, args.dry_run, args.resume)
        else:
            backend.evaluate(preset, args.data_root, args.work_dir, args.checkpoint, args.split)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"vindr: {exc}\n")
