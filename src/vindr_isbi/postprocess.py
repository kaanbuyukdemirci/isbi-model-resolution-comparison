"""Stable NumPy NMS used by the published DINO test results."""

import numpy as np


def class_aware_nms(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return score-ordered indices after deterministic class-aware NMS."""

    kept: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            left = np.maximum(bboxes[current, 0], bboxes[remaining, 0])
            top = np.maximum(bboxes[current, 1], bboxes[remaining, 1])
            right = np.minimum(bboxes[current, 2], bboxes[remaining, 2])
            bottom = np.minimum(bboxes[current, 3], bboxes[remaining, 3])
            intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
            current_area = max(
                0.0,
                float(bboxes[current, 2] - bboxes[current, 0]),
            ) * max(
                0.0,
                float(bboxes[current, 3] - bboxes[current, 1]),
            )
            remaining_area = np.maximum(
                0.0, bboxes[remaining, 2] - bboxes[remaining, 0]
            ) * np.maximum(0.0, bboxes[remaining, 3] - bboxes[remaining, 1])
            union = current_area + remaining_area - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection, dtype=np.float32),
                where=union > 0,
            )
            order = remaining[iou <= float(iou_threshold)]
    return np.asarray(
        sorted(kept, key=lambda index: -float(scores[index])),
        dtype=np.int64,
    )
