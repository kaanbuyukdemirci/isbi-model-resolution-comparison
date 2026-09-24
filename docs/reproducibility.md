# Reproduction audit

Source: `isbi_update` paper and the `kaanbuyukdemirci/vindr-global-pad-multires-v2`
project, exported 2026-09-23. There are 558 project runs, including retry attempts;
549 have W&B state `finished`, 9 `crashed`, and 534 finished runs have both the
required validation and final test metrics. A finished state alone is not a
successful measured experiment.

Within each model/resolution group, maximize the recorded validation mAP50,
breaking ties by lexicographically maximum run ID, exactly as the paper figure
generator did. This gives 107 presets. The paper further removes 50 selected
results below test mAP50 0.60, leaving 57 for its figures. This repository keeps
all 107 presets so the paper's full model set remains usable. Test scores never
participate in recipe selection. `paper-best` names the paper's highest reported
test score among those validation-selected settings; it is not a new
validation-only claim that 1600×1280 is the optimal deployment resolution.

## What is frozen

Each JSON recipe contains the historical model graph, optimizer, learning-rate
schedule, physical batch, accumulation, precision, data-loader worker count,
train/validation pipelines, seed, initialization checkpoint URLs, and early
stopping/checkpoint settings. Python tuple paths preserve types required by
MMCV. Configuration source hashes, W&B run links, measured scores, package
versions, and selected checkpoint SHA-256 are included. The artifacts come from
the *actual resolved run*, including OOM retries, rather than the initial search
recipe. All 107 recipes share the same fixed training/validation/test membership.

Model/data settings are preserved while absolute paths, remote tracking,
diagnostic metrics, unused top-level variables, and old project imports are
removed. Positive subset COCO files contain the same images, annotations, and
order as the historical dataloaders. The retained custom transforms and YOLO
trainer operations were extracted from the original source, including RNG calls.

## Important implementation details

- DINO/ConvNeXt-S `3gs6jzhb`: FP32, physical batch 1, accumulation 2, LR
  0.000025, AdamW weight decay 0.05, L2 clipping 0.1, seed 42. The best checkpoint
  is epoch 7; training stopped at epoch 17 of a 25-epoch budget. Validation
  mAP50 is 0.752; final test mAP50 is 0.709.
- Final test NMS is IoU 0.5. Conventional detectors use their model's final
  NMS; proposal NMS is left intact. DINO uses the same stable NumPy class-aware
  NMS as the original evaluator, after its 300-query output cap. DINO's recorded
  result without this NMS was 0.707. Training validation retains the original
  model configuration and does not add final-test NMS.
- The MMDetection endpoint uses pycocotools through MMDetection's COCO metric
  and reports AP rounded to three decimals. YOLO retains Ultralytics' native AP
  implementation, as the sweep did; those implementations are not exchanged.
- The paper's broad augmentation/early-stopping description does not fully
  describe YOLO's executed settings. Its resolved arguments use `patience=100`
  and a 25-epoch cap, without the MMDetection 0.002 minimum-delta hook. Its
  global-pad data dictionary does not set `float32_tensor`, so the custom
  intensity/noise branch was inactive. Geometry augmentation remains enabled.
  These historical behaviors are deliberately preserved and disclosed.
- The selected YOLO11-M 1600×1280 run was resumed after interruption. Its preset
  records that fact and starts fresh by default; `--resume` always refers to the
  user's own new training directory, never the author's old checkpoint path.
- YOLO's historical validation adapter uses rectangular `pad=0.5`, adding one
  stride of centered padding (32 total pixels in each dimension) around the
  exported image without upscaling it. Inference retains this padding and maps
  detections back to exported-image coordinates. MMDetection uses its saved
  identity-resize and stride-padding pipeline.
- The original negative selector samples images without mass boxes. The actual
  frozen 186-image negative set was checked against the source breast IDs and
  contains no opposite-view image from a mass-positive breast. Freezing that set
  avoids relying on an implicit breast-level screen during regeneration.

## Limits on exactness

The source MMDetection runs explicitly set `deterministic=False` and
`cudnn_benchmark=False`. Floating-point reduction order and CUDA kernels can
change retraining results despite seed 42. Turning deterministic mode on,
altering the batch or accumulation, changing workers/framework versions, or
upgrading GPU hardware changes the reproduction conditions. The CLI preserves
the source conditions rather than silently enabling new determinism options.

The source receipts record a dirty workspace at git revision
`b556cd07928132332ec676a0342c750be0892d5b`; they do not contain complete historical
source diffs or every transitive dependency version. The extracted code is from
the available workspace. Numerical/package pins outside the recorded core ML
stack reflect the inspected local environment, not an invented historical lock.
The requirements are a pinned reproduction environment, not a byte-for-byte
container archive. Training also relies on the saved upstream initialization
URLs; cache those weights if long-term offline reproduction is needed.

Original checkpoint hashes permit exact artifact verification. Original weights
are kept separately to keep this repository small; no public study-weight
download URL was found in the local artifacts. A newly trained checkpoint is
expected to have a different file hash. Do not interpret that as a failed model
solely because the archive bytes differ.

See [validation](validation.md) for checks actually run on this extracted package.
