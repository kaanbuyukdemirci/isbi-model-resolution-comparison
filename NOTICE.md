# Sources and third-party components

This repository distills the author's `vindr-mmdet` training workspace and the
global-pad exporter from `vindr-mammo-reader`. Their project metadata declares
MIT licensing. The retained operations are tensor loading, medical intensity and
noise augmentation, AMP-safe TOOD loss, stable NMS, the YOLO tensor adapter and
checkpoint-selection trainer, and DICOM/COCO export operations. Configuration
graphs derive from the saved MMDetection/Ultralytics runs.

MMDetection, MMCV, MMEngine, MMPReTrain, PyTorch, Ultralytics, upstream pretrained
weights, and VinDr-Mammo retain their own licenses and terms. In particular,
using the Ultralytics backend does not change its upstream licensing. No dataset
pixels or trained study weights are included here. W&B URLs are provenance links;
using this package does not require a W&B account or send metrics to W&B.
