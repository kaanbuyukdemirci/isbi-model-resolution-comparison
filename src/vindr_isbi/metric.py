"""Original positive-only COCO AP endpoint without diagnostic metrics."""

from copy import deepcopy

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS

from .postprocess import class_aware_nms


@METRICS.register_module()
class PaperCocoMetric(CocoMetric):
    def __init__(self, prediction_nms_iou_threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.nms_iou = prediction_nms_iou_threshold

    def compute_metrics(self, results):
        positives = {
            a["image_id"]
            for a in self._coco_api.dataset["annotations"]
            if not a.get("iscrowd", 0) and a["bbox"][2] > 0 and a["bbox"][3] > 0
        }
        if set(self._coco_api.get_img_ids()) - positives:
            raise ValueError("The paper endpoint requires mass-positive images only.")
        if self.nms_iou is not None:
            results = deepcopy(results)
            for _, pred in results:
                keep = class_aware_nms(pred["bboxes"], pred["scores"], pred["labels"], self.nms_iou)
                for key in ("bboxes", "scores", "labels"):
                    pred[key] = pred[key][keep]
        metrics = super().compute_metrics(results)
        for key in ("mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"):
            metrics.setdefault(f"bbox_{key}", 0.0)
        return {
            key.replace("bbox_mAP", "bbox_mass_positive_mAP"): value
            for key, value in metrics.items()
        }
