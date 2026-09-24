"""Inference on the original float32 tensors or a single raw DICOM."""

from pathlib import Path

import numpy as np

from .catalog import write_json


def load_input(path, preset):
    import torch

    path = Path(path)
    h, w = preset["height"], preset["width"]
    metadata = {"input": str(path), "coordinate_space": "exported_image", "height": h, "width": w}
    if path.suffix.lower() == ".pt":
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or tensor.shape[0] not in (1, 3)
        ):
            raise ValueError("Expected a float32 [1,H,W] or [3,H,W] image tensor.")
        if tensor.dtype != torch.float32 or tuple(tensor.shape[-2:]) != (h, w):
            raise ValueError(
                f"Expected float32 tensor at {h}x{w}; got {tensor.dtype}, {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("Input tensor must be finite and in [0,1].")
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return np.ascontiguousarray(
            tensor.numpy().transpose(1, 2, 0)[..., ::-1] * np.float32(255)
        ), metadata
    if path.suffix.lower() not in (".dicom", ".dcm"):
        raise ValueError(
            "Use a .pt export or raw .dcm/.dicom; screenshots are not equivalent inputs."
        )
    import cv2
    import pydicom
    from pydicom.pixels import apply_modality_lut

    ds = pydicom.dcmread(path)
    values = ds.pixel_array
    try:
        values = apply_modality_lut(values, ds)
    except Exception:
        pass  # Preserve the original exporter's malformed optional LUT fallback.
    values = np.asarray(values, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError("Expected one 2D DICOM frame.")
    finite = np.isfinite(values)
    if not finite.all():
        if not finite.any():
            raise ValueError("DICOM contains no finite pixels.")
        values = np.where(finite, values, float(np.nanmin(values[finite])))
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        values = float(values.max()) + float(values.min()) - values
    low, high = float(values.min()), float(values.max())
    normalized = (
        np.zeros_like(values, dtype=np.uint16)
        if high <= low
        else np.rint(np.clip((values - low) / (high - low), 0, 1) * 65535).astype(np.uint16)
    )
    source_h, source_w = values.shape
    if source_h > 3584 or source_w > 2816:
        raise ValueError("DICOM exceeds the paper's 3584x2816 canvas.")
    canvas = np.zeros((3584, 2816), dtype=np.uint16)
    canvas[:source_h, :source_w] = normalized
    content_w = round(2816 * h / 3584)
    resized = cv2.resize(canvas, (content_w, h), interpolation=cv2.INTER_AREA)
    image = np.zeros((h, w), dtype=np.uint16)
    image[:, :content_w] = resized
    image = image.astype(np.float32) * np.float32(1 / 65535)
    metadata.update(
        coordinate_space="source_dicom",
        height=source_h,
        width=source_w,
        scale_x=content_w / 2816,
        scale_y=h / 3584,
    )
    return np.repeat(image[..., None], 3, axis=2) * np.float32(255), metadata


def infer(preset, checkpoint, inputs, output, device, score_threshold):
    if preset["backend"] == "mmdet":
        from .mmdet_backend import load_detector

        model, pipeline, use_nms = load_detector(preset, checkpoint, device)
    else:
        from .compat import ensure_mmdet_transformers_compat

        ensure_mmdet_transformers_compat()
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
    paths = []
    for value in inputs:
        path = Path(value).expanduser()
        paths.extend(
            sorted(p for p in path.rglob("*") if p.suffix.lower() in (".pt", ".dicom", ".dcm"))
            if path.is_dir()
            else [path]
        )
    if not paths:
        raise ValueError("No supported images found.")
    records = []
    for path in paths:
        image, metadata = load_input(path, preset)
        if preset["backend"] == "mmdet":
            from mmdet.apis import inference_detector

            from .postprocess import class_aware_nms

            pred = (
                inference_detector(model, image, test_pipeline=pipeline)
                .pred_instances.cpu()
                .numpy()
            )
            boxes, scores, labels = pred.bboxes, pred.scores, pred.labels
            if use_nms:
                keep = class_aware_nms(boxes, scores, labels, 0.5)
                boxes, scores = boxes[keep], scores[keep]
        else:
            import torch

            from .yolo_backend import console_callbacks

            # The historical validator uses rectangular pad=0.5: one extra
            # stride (32 pixels), centered, without upscaling the export.
            padded = np.full(
                (preset["height"] + 32, preset["width"] + 32, 3), 114, dtype=np.float32
            )
            padded[16:-16, 16:-16] = image
            tensor = (
                torch.from_numpy(np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))) / 255
            )
            with console_callbacks():
                result = model.predict(
                    tensor[None],
                    device=device,
                    imgsz=preset["height"],
                    conf=0.001,
                    iou=0.5,
                    max_det=300,
                    verbose=False,
                )[0]
            boxes, scores = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            boxes = boxes.copy() - 16
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, preset["width"])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, preset["height"])
        if metadata["coordinate_space"] == "source_dicom":
            boxes = boxes.copy()
            boxes[:, [0, 2]] /= metadata["scale_x"]
            boxes[:, [1, 3]] /= metadata["scale_y"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, metadata["width"])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, metadata["height"])
        metadata["detections"] = [
            {"label": "Mass", "category_id": 1, "score": float(score), "bbox_xyxy": box.tolist()}
            for box, score in zip(boxes, scores, strict=True)
            if score >= score_threshold and box[2] > box[0] and box[3] > box[1]
        ]
        records.append(metadata)
    write_json(
        output, {"preset": preset["name"], "score_threshold": score_threshold, "images": records}
    )
    print(f"Saved predictions for {len(records)} image(s) to {output}")
