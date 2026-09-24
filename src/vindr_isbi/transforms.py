"""Frozen runtime operations from the original sweep; see docs/reproducibility.md."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.models.dense_heads import TOODHead
from mmdet.registry import MODELS, TRANSFORMS


def _register_transform(cls):
    return TRANSFORMS.register_module()(cls)


def _register_model(cls):
    return MODELS.register_module()(cls)


def _infer_image_max_value(img: np.ndarray, dtype: np.dtype) -> float:
    """Infer whether an image is in [0, 1] or [0, 255]-style range.

    Some MMDetection/MMCV transforms, notably natural-image photometric
    transforms, may temporarily convert uint8 images into float32 images while
    keeping the numeric range as 0..255.  Earlier versions of this project
    assumed that every float image was 0..1, which clipped almost everything
    to 1 after a later custom transform and made combined previews look nearly
    black.
    """
    if np.issubdtype(dtype, np.integer):
        # MMDetection's image loader returns uint8 for the exported PNG crops.
        # Keep this generic in case another integer depth is used later.
        return float(np.iinfo(dtype).max)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 1.0
    max_seen = float(finite.max())
    min_seen = float(finite.min())
    # Float images from OpenMMLab photometric transforms are commonly still
    # 0..255. Treat any value outside normalized [0, 1]-ish range as 255-scale.
    if max_seen > 2.0 or min_seen < -0.01:
        return 255.0
    return 1.0


def _as_float_img(img: np.ndarray) -> tuple[np.ndarray, np.dtype, float]:
    dtype = img.dtype
    out = img.astype(np.float32, copy=False)
    max_value = _infer_image_max_value(out, dtype)
    return out, dtype, max_value


def _restore_img(img: np.ndarray, dtype: np.dtype, max_value: float) -> np.ndarray:
    img = np.clip(img, 0, max_value)
    if np.issubdtype(dtype, np.integer):
        img = np.rint(img)
    return img.astype(dtype, copy=False)


@_register_transform
class VindrRandomIntensity(BaseTransform):  # type: ignore[misc]
    """Mild monotonic intensity augmentation for mammography-derived channels.

    This is intentionally not HSV/color jitter. It applies a monotonic
    brightness/contrast/gamma perturbation to each pixel intensity. By default
    the same transform is applied to all channels so that three CLAHE/window/
    threshold-style channels keep their relative meaning. It can also preserve
    exact black background pixels, which is useful after breast masking or
    threshold preprocessing.
    """

    def __init__(
        self,
        prob: float = 0.25,
        brightness_delta: float = 0.015,
        contrast_range: tuple[float, float] | list[float] = (0.95, 1.05),
        gamma_range: tuple[float, float] | list[float] = (0.95, 1.05),
        per_channel: bool = False,
        preserve_zero: bool = True,
        zero_tolerance: float = 0.0,
    ) -> None:
        self.prob = float(prob)
        self.brightness_delta = float(brightness_delta)
        self.contrast_range = tuple(float(x) for x in contrast_range)
        self.gamma_range = tuple(float(x) for x in gamma_range)
        self.per_channel = bool(per_channel)
        self.preserve_zero = bool(preserve_zero)
        self.zero_tolerance = float(zero_tolerance)

    def transform(self, results: dict) -> dict:
        if random.random() >= self.prob:
            return results
        img, dtype, max_value = _as_float_img(results["img"])
        out = img.copy()
        zero_mask = img <= (self.zero_tolerance * max_value)

        def sample_params() -> tuple[float, float, float]:
            brightness = random.uniform(-self.brightness_delta, self.brightness_delta) * max_value
            contrast = random.uniform(*self.contrast_range)
            gamma = random.uniform(*self.gamma_range)
            return brightness, contrast, gamma

        if self.per_channel and out.ndim == 3:
            for c in range(out.shape[2]):
                brightness, contrast, gamma = sample_params()
                channel = out[..., c]
                mean = float(channel.mean())
                channel = (channel - mean) * contrast + mean + brightness
                channel = np.clip(channel / max_value, 0, 1) ** gamma * max_value
                out[..., c] = channel
        else:
            brightness, contrast, gamma = sample_params()
            mean = float(out.mean())
            out = (out - mean) * contrast + mean + brightness
            out = np.clip(out / max_value, 0, 1) ** gamma * max_value

        if self.preserve_zero:
            out[zero_mask] = img[zero_mask]

        results["img"] = _restore_img(out, dtype, max_value)
        return results


@_register_transform
class VindrRandomGaussianNoise(BaseTransform):  # type: ignore[misc]
    """Add mild Gaussian noise.

    For thresholded/CLAHE mammography channels the default adds the same noise
    field to all channels and only to nonzero signal pixels. That avoids turning
    the black background into speckle and avoids destroying channel semantics.
    """

    def __init__(
        self,
        prob: float = 0.10,
        std_range: tuple[float, float] | list[float] = (0.002, 0.008),
        per_channel: bool = False,
        signal_only: bool = True,
        signal_threshold: float = 0.0,
    ) -> None:
        self.prob = float(prob)
        self.std_range = tuple(float(x) for x in std_range)
        self.per_channel = bool(per_channel)
        self.signal_only = bool(signal_only)
        self.signal_threshold = float(signal_threshold)

    def transform(self, results: dict) -> dict:
        if random.random() >= self.prob:
            return results
        img, dtype, max_value = _as_float_img(results["img"])
        std = random.uniform(*self.std_range) * max_value
        if self.per_channel and img.ndim == 3:
            noise = np.random.normal(0.0, std, size=img.shape).astype(np.float32)
        else:
            noise_shape = img.shape[:2] + ((1,) if img.ndim == 3 else ())
            noise = np.random.normal(0.0, std, size=noise_shape).astype(np.float32)
        out = img + noise
        if self.signal_only:
            signal_mask = img > (self.signal_threshold * max_value)
            out = np.where(signal_mask, out, img)
        results["img"] = _restore_img(out, dtype, max_value)
        return results


@_register_model
class VindrAmpSafeTOODHead(TOODHead):  # type: ignore[misc]
    """Keep TOOD's probability-form focal loss outside AMP autocast.

    MMDetection 3.3's TOOD head passes sigmoid probabilities to
    ``binary_cross_entropy`` during its initial assignment epochs. PyTorch
    correctly rejects that operation inside CUDA autocast. Feature extraction
    and the TOOD forward pass still use AMP; only target construction and loss
    arithmetic run in FP32.
    """

    def loss_by_feat(
        self,
        cls_scores: list[Any],
        bbox_preds: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for TOOD training") from exc

        device_type = cls_scores[0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return super().loss_by_feat(
                [value.float() for value in cls_scores],
                [value.float() for value in bbox_preds],
                *args,
                **kwargs,
            )


@_register_transform
class VindrLoadFloat32TensorFromFile(BaseTransform):  # type: ignore[misc]
    """Load an exporter-owned CHW RGB float32 tensor as HWC BGR float32.

    The output is scaled to 0..255 so existing ImageNet mean/std preprocessors
    retain exactly the same units as their PNG path. Values remain floating
    point; no uint8 round-trip occurs.
    """

    def __init__(self, scale_to_255: bool = True, rgb_to_bgr: bool = True) -> None:
        self.scale_to_255 = bool(scale_to_255)
        self.rgb_to_bgr = bool(rgb_to_bgr)

    @staticmethod
    def tensor_path(image_path: str | Path) -> Path:
        path = Path(image_path)
        # The global-pad multires export stores the tensor path itself in
        # COCO ``file_name``.  The older research export instead stores a PNG
        # indexing path with the tensor mirrored below images/float32.
        if path.suffix.casefold() == ".pt":
            return path
        parts = list(path.parts)
        try:
            image_index = len(parts) - 1 - parts[::-1].index("images")
        except ValueError as exc:
            raise ValueError(f"Cannot derive float32 tensor path from {path}") from exc
        if image_index + 1 >= len(parts) or parts[image_index + 1] == "float32":
            raise ValueError(f"Unexpected research image path for float32 loading: {path}")
        parts.insert(image_index + 1, "float32")
        return Path(*parts).with_suffix(".pt")

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for float32 research tensors") from exc

        image_path = results.get("img_path")
        if not image_path:
            raise KeyError("VindrLoadFloat32TensorFromFile requires results['img_path']")
        tensor_path = self.tensor_path(str(image_path))
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        try:
            tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility.
            tensor = torch.load(tensor_path, map_location="cpu")
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or tensor.shape[0] not in {1, 3}
        ):
            raise ValueError(f"Expected CHW one- or three-channel tensor in {tensor_path}")
        if tensor.shape[0] == 1:
            # Replication is lossless and deterministic.  It preserves the
            # pretrained RGB backbone contract without inventing color.
            tensor = tensor.repeat(3, 1, 1)
        image = tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
        if not np.isfinite(image).all():
            raise ValueError(f"Float32 tensor contains non-finite values: {tensor_path}")
        image = np.transpose(image, (1, 2, 0))
        if self.rgb_to_bgr:
            image = image[..., ::-1]
        image = np.ascontiguousarray(image, dtype=np.float32)
        if self.scale_to_255:
            image *= np.float32(255.0)
        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        results["float32_tensor_path"] = str(tensor_path)
        return results
