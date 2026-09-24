"""Rebuild one resolution from licensed raw VinDr data using the frozen split."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .catalog import PACKAGE, read_json, sha256


def export_record(job):
    from .preprocessing import (
        _atomic_write_float32,
        _normalize_dicom_to_uint16,
        _render_variant,
        _variant_paths,
        pad_bottom_right,
    )

    record, variant, output = job
    path, _ = _variant_paths(output, variant, record)
    if path.is_file():
        return 0
    image = _normalize_dicom_to_uint16(record.dicom_path)
    if image.shape != (record.height, record.width):
        raise ValueError(f"Source geometry differs for {record.dicom_path}")
    canvas = pad_bottom_right(image, 3584, 2816)
    _atomic_write_float32(path, _render_variant(canvas, variant))
    return 1


def worker_init():
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def prepare(preset, raw_root, output_root, workers):
    from .preprocessing import OutputVariant, _write_coco_files, load_source_records

    raw_root = Path(raw_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output == raw_root:
        raise ValueError("Choose an output directory separate from the raw data.")
    records = load_source_records(raw_root)
    h, w = preset["height"], preset["width"]
    content_w = round(2816 * h / 3584)
    variant = OutputVariant(preset["variant"], h, w, h, content_w, content_w / 2816, h / 3584)
    _write_coco_files(output, records, [variant])
    contract = read_json(PACKAGE / "assets/data_contract.json")[variant.name]
    for split, expected in contract.items():
        if sha256(output / expected["annotation"]) != expected["sha256"]:
            raise ValueError(
                f"Rebuilt {split} annotations differ from the paper export. Check the raw VinDr release and CSV files."
            )
    indices = read_json(PACKAGE / "assets/splits.json")
    selected = [
        r
        for split in ("train", "val", "test")
        for i, r in enumerate(r for r in records if r.split == split)
        if i in set(indices[split])
    ]
    print(
        f"Preparing {len(selected)} tensors at {h}x{w}; approximately {len(selected) * h * w * 4 / 1e9:.1f} GB.",
        flush=True,
    )
    jobs = ((r, variant, output) for r in selected)
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for i, _ in enumerate(pool.map(export_record, jobs), 1):
            if i % 50 == 0 or i == len(selected):
                print(f"{i}/{len(selected)} images ready", flush=True)
    print(f"Ready: {output}")
