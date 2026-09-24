"""Essential data-export operations copied from the actual global-pad exporter."""

from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from pydicom.pixels import apply_modality_lut

from .catalog import PACKAGE, read_json


@dataclass(frozen=True)
class MassBox:
    annotation_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class SourceRecord:
    study_id: str
    image_id: str
    split: str
    dicom_path: str
    height: int
    width: int
    laterality: str
    view_position: str
    mass_boxes: tuple[MassBox, ...]


@dataclass(frozen=True)
class OutputVariant:
    name: str
    output_height: int
    output_width: int
    resized_content_height: int
    resized_content_width: int
    scale_x: float
    scale_y: float
    is_padded_original: bool = False


def pad_bottom_right(
    image: np.ndarray,
    output_height: int,
    output_width: int,
    *,
    value: int = 0,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    if height > output_height or width > output_width:
        raise ValueError(
            f"Image {height}x{width} does not fit output {output_height}x{output_width} (HxW)."
        )
    output = np.full((output_height, output_width), value, dtype=image.dtype)
    output[:height, :width] = image
    return output


def _parse_finding_categories(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value)
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return tuple(str(item).strip() for item in parsed)
    return (text,)


def load_source_records(
    data_root: str | Path,
    *,
    split_assignments: str | Path | None = None,
) -> list[SourceRecord]:
    data_root = Path(data_root).resolve()
    breast_path = data_root / "breast-level_annotations.csv"
    finding_path = data_root / "finding_annotations.csv"
    for required_path in (breast_path, finding_path, data_root / "images"):
        if not required_path.exists():
            raise FileNotFoundError(f"Required VinDr source path not found: {required_path}")

    breast = pd.read_csv(
        breast_path,
        dtype={"study_id": str, "image_id": str, "series_id": str},
    )
    required_breast = {
        "study_id",
        "image_id",
        "height",
        "width",
        "split",
        "laterality",
        "view_position",
    }
    missing = sorted(required_breast - set(breast.columns))
    if missing:
        raise ValueError(f"breast-level_annotations.csv is missing columns: {missing}")
    if breast.duplicated(["study_id", "image_id"]).any():
        raise ValueError("breast-level_annotations.csv contains duplicate study/image keys.")

    findings = pd.read_csv(
        finding_path,
        dtype={"study_id": str, "image_id": str, "series_id": str},
    ).reset_index(names="source_annotation_id")
    required_findings = {
        "study_id",
        "image_id",
        "finding_categories",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
    missing = sorted(required_findings - set(findings.columns))
    if missing:
        raise ValueError(f"finding_annotations.csv is missing columns: {missing}")

    mass_boxes_by_image: dict[tuple[str, str], list[MassBox]] = {}
    for row in findings.itertuples(index=False):
        if "Mass" not in _parse_finding_categories(row.finding_categories):
            continue
        coordinates = (float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"Non-finite Mass box for {row.study_id}/{row.image_id}: {coordinates}"
            )
        xmin, ymin, xmax, ymax = coordinates
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid Mass box for {row.study_id}/{row.image_id}: {coordinates}")
        key = (str(row.study_id), str(row.image_id))
        mass_boxes_by_image.setdefault(key, []).append(
            MassBox(
                annotation_id=int(row.source_annotation_id),
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    split_path = Path(split_assignments).resolve() if split_assignments else None
    study_splits = read_json(PACKAGE / "assets/study_splits.json")
    split_map = {
        (str(r.study_id), str(r.image_id)): study_splits[str(r.study_id)]
        for r in breast.itertuples(index=False)
    }
    records: list[SourceRecord] = []
    for row in breast.itertuples(index=False):
        study_id = str(row.study_id)
        image_id = str(row.image_id)
        key = (study_id, image_id)
        official_split = str(row.split).strip().casefold()
        fallback_split = {"training": "train", "test": "test"}.get(official_split)
        if fallback_split is None:
            raise ValueError(f"Unsupported official split {row.split!r} for {study_id}/{image_id}.")
        split = split_map.get(key, fallback_split)
        if split_map and key not in split_map:
            raise ValueError(f"Missing split assignment for {study_id}/{image_id} in {split_path}")
        records.append(
            SourceRecord(
                study_id=study_id,
                image_id=image_id,
                split=split,
                dicom_path=str(data_root / "images" / study_id / f"{image_id}.dicom"),
                height=int(row.height),
                width=int(row.width),
                laterality=str(row.laterality),
                view_position=str(row.view_position),
                mass_boxes=tuple(mass_boxes_by_image.get(key, ())),
            )
        )
    return records


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _normalize_dicom_to_uint16(path: str | Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path))
    values = dataset.pixel_array
    try:
        values = apply_modality_lut(values, dataset)
    except Exception:
        # The official VinDr files use identity rescale values. If a particular
        # file has malformed optional LUT metadata, retaining decoded pixels is
        # safer than dropping the image.
        pass
    values = np.asarray(values, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Expected one 2D frame, got {values.shape}: {path}")
    finite = np.isfinite(values)
    if not bool(finite.all()):
        if not bool(finite.any()):
            raise ValueError(f"DICOM contains no finite pixels: {path}")
        values = np.where(finite, values, float(np.nanmin(values[finite])))
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        values = float(values.max()) + float(values.min()) - values
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint16)
    normalized = (values - low) / (high - low)
    return np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)


def _atomic_write_float32(path: Path, image: np.ndarray) -> None:
    """Save one grayscale image as contiguous CHW [1,H,W] float32 in [0,1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.pt")
    values = image.astype(np.float32)
    values *= np.float32(1.0 / 65535.0)
    tensor = torch.from_numpy(values).unsqueeze(0).contiguous()
    torch.save(tensor, temporary)
    temporary.replace(path)


def _stem(record: SourceRecord) -> str:
    return f"{record.study_id}__{record.image_id}"


def _variant_paths(
    output_root: Path,
    variant: OutputVariant,
    record: SourceRecord,
) -> tuple[Path, Path]:
    filename = _stem(record)
    float32_path = (
        output_root / "images" / "float32" / variant.name / record.split / f"{filename}.pt"
    )
    label_path = output_root / "labels" / variant.name / record.split / f"{filename}.txt"
    return float32_path, label_path


def _transform_box(
    box: MassBox,
    variant: OutputVariant,
    *,
    source_height: int,
    source_width: int,
) -> tuple[float, float, float, float]:
    xmin = min(max(float(box.xmin), 0.0), float(source_width)) * variant.scale_x
    xmax = min(max(float(box.xmax), 0.0), float(source_width)) * variant.scale_x
    ymin = min(max(float(box.ymin), 0.0), float(source_height)) * variant.scale_y
    ymax = min(max(float(box.ymax), 0.0), float(source_height)) * variant.scale_y
    xmin = min(max(xmin, 0.0), float(variant.output_width))
    xmax = min(max(xmax, 0.0), float(variant.output_width))
    ymin = min(max(ymin, 0.0), float(variant.output_height))
    ymax = min(max(ymax, 0.0), float(variant.output_height))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Mass box became empty after clipping: {box}")
    return xmin, ymin, xmax, ymax


def _render_variant(canvas: np.ndarray, variant: OutputVariant) -> np.ndarray:
    if variant.is_padded_original:
        return canvas
    resized = cv2.resize(
        canvas,
        (variant.resized_content_width, variant.resized_content_height),
        interpolation=cv2.INTER_AREA,
    )
    return pad_bottom_right(resized, variant.output_height, variant.output_width)


def _write_coco_files(
    output_root: Path,
    records: Sequence[SourceRecord],
    variants: Sequence[OutputVariant],
) -> None:
    annotation_root = output_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for split in ("train", "val", "test"):
            split_records = [record for record in records if record.split == split]
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            annotation_number = 1
            for image_number, record in enumerate(split_records, start=1):
                float32_path, _ = _variant_paths(output_root, variant, record)
                images.append(
                    {
                        "id": image_number,
                        "file_name": float32_path.relative_to(output_root).as_posix(),
                        "float32_path": float32_path.relative_to(output_root).as_posix(),
                        "storage_format": "float32_only",
                        "height": variant.output_height,
                        "width": variant.output_width,
                        "study_id": record.study_id,
                        "image_id": record.image_id,
                    }
                )
                for box in record.mass_boxes:
                    xmin, ymin, xmax, ymax = _transform_box(
                        box,
                        variant,
                        source_height=record.height,
                        source_width=record.width,
                    )
                    width = xmax - xmin
                    height = ymax - ymin
                    annotations.append(
                        {
                            "id": annotation_number,
                            "image_id": image_number,
                            "category_id": 1,
                            "bbox": [xmin, ymin, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "source_annotation_id": box.annotation_id,
                        }
                    )
                    annotation_number += 1
            _atomic_write_json(
                annotation_root / variant.name / f"instances_{split}.json",
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": 1, "name": "Mass"}],
                },
            )
