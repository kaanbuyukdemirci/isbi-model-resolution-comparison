"""YOLO11 float-tensor adapter and original checkpoint selection semantics."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class LazyCropYOLODatasetMixin:
    def _lazy_settings(self):
        return {}

    def _float32_settings(self):
        return {}

    def _multires_settings(self) -> Mapping[str, Any]:
        settings = (getattr(self, "data", None) or {}).get("global_pad_multires") or {}
        return settings if settings.get("enabled", False) else {}

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        multires = self._multires_settings()
        if multires:
            if isinstance(img_path, list):
                if len(img_path) != 1:
                    raise ValueError("Multires datasets require exactly one COCO file per split")
                img_path = img_path[0]
            annotation = Path(img_path).expanduser().resolve()
            coco = json.loads(annotation.read_text(encoding="utf-8"))
            records = list(coco.get("images") or [])
            fraction = float(getattr(self, "fraction", 1.0))
            if fraction < 1.0:
                records = records[: round(len(records) * fraction)]
            selected_ids = {int(record["id"]) for record in records}
            annotations: dict[int, list[dict[str, Any]]] = {
                image_id: [] for image_id in selected_ids
            }
            for item in coco.get("annotations") or []:
                image_id = int(item["image_id"])
                if image_id in annotations:
                    annotations[image_id].append(item)
            self._multires_records = records
            self._multires_annotations = annotations
            virtual_root = annotation.parent / ".vindr_multires_virtual" / annotation.stem
            return [str(virtual_root / f"{index:09d}.png") for index in range(len(records))]
        raise ValueError("Only the paper global-pad tensor dataset is supported")

    def get_labels(self) -> list[dict[str, Any]]:
        if self._multires_settings():
            labels: list[dict[str, Any]] = []
            for image_file, record in zip(self.im_files, self._multires_records, strict=True):
                height = int(record["height"])
                width = int(record["width"])
                boxes: list[list[float]] = []
                for annotation in self._multires_annotations[int(record["id"])]:
                    if int(annotation.get("iscrowd", 0) or 0):
                        continue
                    x, y, box_width, box_height = [
                        float(value) for value in annotation.get("bbox", ())
                    ]
                    if box_width <= 0 or box_height <= 0:
                        continue
                    boxes.append(
                        [
                            (x + box_width / 2.0) / width,
                            (y + box_height / 2.0) / height,
                            box_width / width,
                            box_height / height,
                        ]
                    )
                labels.append(
                    {
                        "im_file": image_file,
                        "shape": (height, width),
                        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
                        "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                        "segments": [],
                        "keypoints": None,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
            return labels
        raise ValueError("Only the paper global-pad tensor dataset is supported")

    def load_image(
        self,
        i: int,
        rect_mode: bool = True,
        resize_short: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        if (
            not self._lazy_settings()
            and not self._float32_settings()
            and not self._multires_settings()
        ):
            return super().load_image(i, rect_mode=rect_mode, resize_short=resize_short)
        if self._multires_settings():
            try:
                import torch
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("PyTorch is required for multires YOLO input") from exc
            cached = self.ims[i]
            if cached is not None:
                return cached, self.im_hw0[i], self.im_hw[i]
            virtual_index = int(Path(self.im_files[i]).stem)
            record = self._multires_records[virtual_index]
            tensor_path = Path(str(self._multires_settings()["root"])) / str(record["file_name"])
            try:
                tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
            except TypeError:
                tensor = torch.load(tensor_path, map_location="cpu")
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 3
                or tensor.shape[0] not in {1, 3}
            ):
                raise ValueError(f"Expected CHW one- or three-channel tensor in {tensor_path}")
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            image = tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
            image = np.ascontiguousarray(np.transpose(image, (1, 2, 0))[..., ::-1])
            image *= np.float32(255.0)
            if not np.isfinite(image).all():
                raise ValueError(f"Non-finite values in {tensor_path}")
            h0, w0 = image.shape[:2]
            expected = (
                int(self._multires_settings()["height"]),
                int(self._multires_settings()["width"]),
            )
            if (h0, w0) != expected:
                raise ValueError(f"Tensor {tensor_path} has {(h0, w0)}, expected {expected}")
            # imgsz equals the variant height (the long side), so rectangular
            # loading is an identity transform for every exported variant.
            if not rect_mode or int(self.imgsz) != h0:
                raise ValueError(
                    "Multires YOLO input requires rect=True and imgsz equal to variant height"
                )
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = image, (h0, w0), (h0, w0)
                self.buffer.append(i)
            return image, (h0, w0), (h0, w0)
        raise ValueError("Only the paper global-pad tensor dataset is supported")


def _maximum_model_stride(value: Any, minimum: int = 32) -> int:
    """Normalize Ultralytics stride metadata across model generations.

    YOLO11 exposes a tensor-like stride vector, while the end-to-end YOLO26
    trainer can expose an already-reduced Python integer. The dataset only
    needs the maximum integer stride, so accept either representation without
    assuming a ``.max()`` method.
    """

    candidate = value
    max_method = getattr(candidate, "max", None)
    if callable(max_method):
        candidate = max_method()
    elif isinstance(candidate, (list, tuple)):
        if not candidate:
            raise ValueError("model stride sequence must not be empty")
        candidate = max(candidate)
    item_method = getattr(candidate, "item", None)
    if callable(item_method):
        candidate = item_method()
    try:
        stride = int(candidate)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"unsupported model stride metadata: {value!r}") from exc
    if stride <= 0:
        raise ValueError(f"model stride must be positive, got {stride}")
    return max(stride, int(minimum))


def _checkpoint_fitness_override(
    metrics: Any,
    fallback_fitness: Any,
    checkpoint_metric: str | None,
    previous_best_fitness: Any,
) -> tuple[float, float] | None:
    """Resolve a configured checkpoint metric without importing Ultralytics.

    Ultralytics updates ``best_fitness`` inside its base ``validate`` method
    using its composite fitness value.  A caller that selects a different
    metric must therefore restore the pre-validation best value and compare
    the selected metric against that value.  ``None`` means the configured
    metric was unavailable or invalid, so the caller should retain the base
    trainer's fallback fitness and state unchanged.
    """
    metric_name = str(checkpoint_metric or "").strip()
    if not metric_name or not isinstance(metrics, Mapping):
        return None

    selected = metrics.get(metric_name)
    try:
        selected_fitness = float(selected)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(selected_fitness):
        return None

    try:
        previous_best = float(previous_best_fitness)
    except (TypeError, ValueError):
        previous_best = None
    if previous_best is not None and not math.isfinite(previous_best):
        previous_best = None

    # Match Ultralytics' update rule, including its treatment of a zero-valued
    # initial best, while comparing values from the same selected metric.
    best_fitness = selected_fitness
    if previous_best is not None and previous_best and previous_best >= selected_fitness:
        best_fitness = previous_best
    return selected_fitness, best_fitness


def _apply_float32_medical_batch_augmentations(
    images: Any, augmentations: Mapping[str, Any] | None
) -> Any:
    """Apply the MMDetection-style monotonic intensity/noise profile to YOLO.

    This runs after Ultralytics normalizes the batch, preserves exact black
    background when requested, and never quantizes the float tensor.
    """

    settings = dict(augmentations or {})
    if not settings.get("enabled", False):
        return images
    intensity = dict(settings.get("intensity") or {})
    noise_settings = dict(settings.get("gaussian_noise") or {})
    if not intensity.get("enabled", False) and not noise_settings.get("enabled", False):
        return images

    import torch

    output = images.clone()
    for image in output:
        if intensity.get("enabled", False) and random.random() < float(intensity.get("prob", 0.5)):
            brightness_delta = float(intensity.get("brightness_delta", 0.05))
            brightness = random.uniform(-brightness_delta, brightness_delta)
            contrast = random.uniform(
                *[float(value) for value in intensity.get("contrast_range", [0.85, 1.15])]
            )
            gamma = random.uniform(
                *[float(value) for value in intensity.get("gamma_range", [0.85, 1.15])]
            )
            zero_tolerance = float(intensity.get("zero_tolerance", 0.0))
            zero_mask = image <= zero_tolerance
            mean = image.mean()
            adjusted = (image - mean) * contrast + mean + brightness
            adjusted = adjusted.clamp_(0.0, 1.0).pow_(gamma)
            if intensity.get("preserve_zero", True):
                adjusted = torch.where(zero_mask, image, adjusted)
            image.copy_(adjusted)

        if noise_settings.get("enabled", False) and random.random() < float(
            noise_settings.get("prob", 0.25)
        ):
            std = random.uniform(
                *[float(value) for value in noise_settings.get("std_range", [0.005, 0.03])]
            )
            noise_shape = (
                image.shape if noise_settings.get("per_channel", False) else (1, *image.shape[1:])
            )
            noise = torch.randn(noise_shape, dtype=image.dtype, device=image.device) * std
            candidate = (image + noise).clamp_(0.0, 1.0)
            if noise_settings.get("signal_only", True):
                threshold = float(noise_settings.get("signal_threshold", 0.0))
                candidate = torch.where(image > threshold, candidate, image)
            image.copy_(candidate)
    return output


def make_random_erasing_trainer(
    probability: float,
    scale: tuple[float, float] = (0.02, 0.33),
    ratio: tuple[float, float] = (0.3, 3.3),
    checkpoint_metric: str | None = None,
    gradient_clip_max_norm: float = 10.0,
    augmentations: Mapping[str, Any] | None = None,
) -> type:
    """Create the VinDr trainer with external-label mapping and random erasing.

    Image directories may live on filesystems that reject symlinks and hard
    links. ``VindrYOLODataset`` lets data.yaml point directly at those source
    images while labels remain in the run directory. Ultralytics also accepts
    the ``erasing`` argument for detection but does not consume it in its native
    detection transforms, so the trainer applies it to normalized batches.
    """
    import math
    import random
    from copy import copy

    import torch
    import torch.distributed as dist
    import ultralytics.data.dataset as ultralytics_dataset
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
    from ultralytics.utils import RANK, colorstr
    from ultralytics.utils.metrics import DetMetrics
    from ultralytics.utils.torch_utils import unwrap_model

    probability = float(probability)
    min_scale, max_scale = (float(v) for v in scale)
    min_ratio, max_ratio = (float(v) for v in ratio)
    checkpoint_metric = str(checkpoint_metric or "").strip()
    gradient_clip_max_norm = float(gradient_clip_max_norm)

    class VindrYOLODataset(LazyCropYOLODatasetMixin, YOLODataset):
        def get_labels(self) -> list[dict[str, Any]]:
            if self._lazy_settings():
                return super().get_labels()
            label_root = (self.data or {}).get("label_root")
            if not label_root:
                return super().get_labels()

            original_mapper = ultralytics_dataset.img2label_paths

            def map_external_labels(image_files: list[str]) -> list[str]:
                root = Path(str(label_root)).expanduser()
                return [
                    str(root / Path(image_file).parent.name / f"{Path(image_file).stem}.txt")
                    for image_file in image_files
                ]

            # YOLODataset.get_labels resolves the module-level mapper. Replace
            # it only for this synchronous initialization and restore it even
            # if label verification raises.
            ultralytics_dataset.img2label_paths = map_external_labels
            try:
                return super().get_labels()
            finally:
                ultralytics_dataset.img2label_paths = original_mapper

    def build_vindr_dataset(
        owner: Any, img_path: str, mode: str = "val", batch: int | None = None
    ) -> Any:
        configured_stride = getattr(owner, "stride", None)
        if configured_stride is None:
            configured_stride = unwrap_model(owner.model).stride
        stride = _maximum_model_stride(configured_stride)
        lazy_settings = owner.data.get("lazy_crop") or {}
        lazy_crop = bool(lazy_settings.get("enabled", False))
        lazy_augment = bool(lazy_settings.get("allow_augmentations", False))
        float32_input = bool((owner.data.get("float32_tensor") or {}).get("enabled", False))
        multires_input = bool((owner.data.get("global_pad_multires") or {}).get("enabled", False))
        return VindrYOLODataset(
            img_path=img_path,
            imgsz=owner.args.imgsz,
            batch_size=batch,
            augment=mode == "train" and (not lazy_crop or lazy_augment),
            hyp=owner.args,
            rect=owner.args.rect or mode == "val" or multires_input,
            cache=(
                None if lazy_crop or float32_input or multires_input else owner.args.cache or None
            ),
            single_cls=owner.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=owner.args.task,
            classes=owner.args.classes,
            data=owner.data,
            fraction=owner.args.fraction if mode == "train" else 1.0,
        )

    class MassPositiveDetectionValidator(DetectionValidator):
        """Compute full and Mass-positive AP from the same validation outputs."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mass_positive_metrics = DetMetrics()
            self.mass_positive_images = 0
            self.positive_only_dataset = False

        def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> Any:
            return build_vindr_dataset(self, img_path, mode, batch)

        def init_metrics(self, model: torch.nn.Module) -> None:
            super().init_metrics(model)
            split = str(getattr(self.args, "split", "val") or "val")
            whole = dict((self.data or {}).get("whole_resized") or {})
            if not whole:
                whole = dict((self.data or {}).get("global_pad_multires") or {})
            self.positive_only_dataset = bool(
                whole.get(
                    "positive_only_test" if split == "test" else "positive_only_validation",
                    False,
                )
            )
            self.mass_positive_metrics.names = model.names
            self.mass_positive_metrics.clear_stats()
            self.mass_positive_metrics.clear_image_metrics()
            self.mass_positive_images = 0

        def update_metrics(
            self,
            preds: list[dict[str, torch.Tensor]],
            batch: dict[str, Any],
        ) -> None:
            before = len(self.metrics.stats["tp"])
            super().update_metrics(preds, batch)
            if self.positive_only_dataset:
                self.mass_positive_images += len(preds)
                return
            # DetectionValidator appends one already-matched stat record per
            # image. Reuse those records for positive images instead of
            # postprocessing or matching predictions a second time.
            for sample_index in range(len(preds)):
                target_mask = batch["batch_idx"] == sample_index
                if not bool(target_mask.any()):
                    continue
                stat_index = before + sample_index
                stat = {
                    key: self.metrics.stats[key][stat_index]
                    for key in self.mass_positive_metrics.stats
                }
                stat["im_name"] = Path(batch["im_file"][sample_index]).name
                self.mass_positive_metrics.update_stats(stat)
                self.mass_positive_images += 1

        def gather_stats(self) -> None:
            super().gather_stats()
            if self.positive_only_dataset:
                self.mass_positive_images = len(self.metrics.stats["tp"])
                return
            if RANK == 0:
                gathered_stats = [None] * dist.get_world_size()
                dist.gather_object(
                    self.mass_positive_metrics.stats,
                    gathered_stats,
                    dst=0,
                )
                merged = {key: [] for key in self.mass_positive_metrics.stats}
                for stats in gathered_stats:
                    for key in merged:
                        merged[key].extend(stats[key])
                self.mass_positive_metrics.stats = merged
                gathered_counts = [None] * dist.get_world_size()
                dist.gather_object(
                    self.mass_positive_images,
                    gathered_counts,
                    dst=0,
                )
                self.mass_positive_images = sum(int(value) for value in gathered_counts)
            elif RANK > 0:
                dist.gather_object(self.mass_positive_metrics.stats, None, dst=0)
                dist.gather_object(self.mass_positive_images, None, dst=0)
                self.mass_positive_metrics.clear_stats()

        def get_stats(self) -> dict[str, Any]:
            full_results = super().get_stats()
            key_map = {
                "metrics/precision(B)": "metrics/precision_mass_positive(B)",
                "metrics/recall(B)": "metrics/recall_mass_positive(B)",
                "metrics/mAP50(B)": "metrics/mAP50_mass_positive(B)",
                "metrics/mAP50-95(B)": "metrics/mAP50-95_mass_positive(B)",
            }
            if self.positive_only_dataset:
                positive_only = {
                    output_key: float(full_results.pop(input_key, 0.0))
                    for input_key, output_key in key_map.items()
                }
                positive_only["metrics/mass_positive_images"] = float(self.mass_positive_images)
                return {**full_results, **positive_only}
            if self.mass_positive_images > 0:
                self.mass_positive_metrics.process(
                    save_dir=self.save_dir,
                    plot=False,
                    on_plot=None,
                )
                positive = self.mass_positive_metrics.results_dict
            else:
                positive = {
                    "metrics/precision(B)": 0.0,
                    "metrics/recall(B)": 0.0,
                    "metrics/mAP50(B)": 0.0,
                    "metrics/mAP50-95(B)": 0.0,
                }
            self.mass_positive_metrics.clear_stats()
            full_results.update(
                {
                    output_key: float(positive.get(input_key, 0.0))
                    for input_key, output_key in key_map.items()
                }
            )
            full_results["metrics/mass_positive_images"] = float(self.mass_positive_images)
            return full_results

    class Paper22DetectionTrainer(DetectionTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # This project owns one compact W&B run. If a user has enabled the
            # global Ultralytics W&B integration, suppress its callbacks while
            # BaseTrainer installs integrations. Otherwise it duplicates every
            # metric and finishes our run before optional post-training
            # evaluation. This also makes --no-wandb authoritative.
            from ultralytics.utils.callbacks import wb as ultralytics_wb

            native_wandb_callbacks = ultralytics_wb.callbacks
            ultralytics_wb.callbacks = {}
            try:
                super().__init__(*args, **kwargs)
            finally:
                ultralytics_wb.callbacks = native_wandb_callbacks

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            return build_vindr_dataset(self, img_path, mode, batch)

        def get_validator(self):
            """Use the shared-inference full/positive validation metric."""

            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return MassPositiveDetectionValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

        def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
            batch = super().preprocess_batch(batch)
            images = batch["img"]
            if bool((self.data.get("float32_tensor") or {}).get("enabled", False)):
                images = _apply_float32_medical_batch_augmentations(images, augmentations)
                batch["img"] = images
            height, width = images.shape[-2:]
            area = height * width
            for image in images:
                if random.random() >= probability:
                    continue
                for _ in range(10):
                    target_area = random.uniform(min_scale, max_scale) * area
                    aspect = math.exp(random.uniform(math.log(min_ratio), math.log(max_ratio)))
                    erase_h = int(round(math.sqrt(target_area * aspect)))
                    erase_w = int(round(math.sqrt(target_area / aspect)))
                    if 0 < erase_h < height and 0 < erase_w < width:
                        top = int(
                            torch.randint(0, height - erase_h + 1, (1,), device=images.device)
                        )
                        left = int(
                            torch.randint(0, width - erase_w + 1, (1,), device=images.device)
                        )
                        image[:, top : top + erase_h, left : left + erase_w] = 0.0
                        break
            return batch

        def optimizer_step(self) -> None:
            """Honor the experiment's explicit gradient-clipping value."""
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

        def validate(self):
            """Use the declared paper comparison metric for best/early-stop fitness."""
            previous_best_fitness = self.best_fitness
            metrics, fitness = super().validate()
            override = _checkpoint_fitness_override(
                metrics,
                fitness,
                checkpoint_metric,
                previous_best_fitness,
            )
            if override is not None:
                fitness, self.best_fitness = override
            return metrics, fitness

    Paper22DetectionTrainer.mass_positive_validator_class = MassPositiveDetectionValidator
    return Paper22DetectionTrainer
