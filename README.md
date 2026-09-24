# VinDr-Mammo: input resolution and mass detection

Console training, evaluation, and inference for **Input Resolution in Mammographic
Mass Detection: A Controlled Multi-Architecture Study** (`isbi_update` manuscript).
Each command runs one model. This standalone repository contains frozen recipes,
the small custom tensor/augmentation adapters, and the fixed data partitions.
There is no GUI, experiment queue, hyperparameter search, or W&B requirement.

The default `paper-best` preset is **DINO / ConvNeXt-S, 1600×1280**, the paper's
highest reported test result: **0.709 mass-positive test mAP50**, from a checkpoint
selected by validation mAP50. Its validation mAP50 was 0.752. See
[reproducibility notes](docs/reproducibility.md) before interpreting “exact”.

## Install

Use Linux, Python **3.10.20**, and a CUDA-capable NVIDIA GPU. The source runs used
one RTX 3090 (24 GB), PyTorch 2.1.0 + CUDA 12.1, and MMDetection 3.3.0.

```bash
conda create -n vindr-isbi python=3.10.20 -y
conda activate vindr-isbi
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/mmdet.txt
python -m pip install -e .
vindr list
```

Install `requirements/yolo.txt` as well to run YOLO11-M. For a YOLO-only
installation, use that file in place of `requirements/mmdet.txt`. Install
`requirements/dicom.txt` for raw DICOM preparation/inference. Framework and
numerical packages are pinned; the complete original versions recorded for each
run are embedded in its preset. Installation follows the official
[PyTorch archived version instructions](https://pytorch.org/get-started/previous-versions/)
and uses the matching [MMCV CUDA wheel index](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html).

## Data

Already have `preprocessed-vindr-global-pad-multires-v1`? Pass its root directly.
Otherwise obtain licensed VinDr-Mammo v1.0.0 and prepare one resolution:

```bash
python -m pip install -r requirements/dicom.txt
vindr prepare --preset paper-best \
  --raw-root /path/to/vindr-mammo \
  --output-root /path/to/global-pad-data
```

Preparation recreates the frozen annotations and only the **1,299 images needed
by the paper cohort**, about 10.6 GB at 1600×1280. It preserves the fixed train,
validation, and test membership. See [data layout and preprocessing](docs/data.md).

## Train one model

```bash
vindr train --preset paper-best \
  --data-root /path/to/global-pad-data \
  --work-dir runs/dino-convnext-s-1600 --device cuda:0
```

Training saves a resolved configuration, an environment receipt, a rolling
checkpoint, and the best validation checkpoint. Add `--dry-run` with a separate
empty work directory to verify the data and configuration without training. Use
`--resume` with the original training directory to resume an interrupted run.

| Setting | Default paper-best recipe |
|---|---|
| Seed / partition seed | 42 / 42 |
| Physical batch / accumulation / effective batch | 1 / 2 / 2 |
| Optimizer / initial LR / weight decay | AdamW / 0.000025 / 0.05 |
| Precision | FP32 (AMP disabled) |
| Gradient clipping | L2 norm 0.1 |
| Layer decay | 0.7; 6 layers; layer-wise |
| LR schedule | 250-iteration warm-up; ×0.1 at epochs 16 and 23 |
| Maximum epochs | 25 |
| Early stopping | Validation mAP50; patience 10; min delta 0.002 |
| Historical best checkpoint | Epoch 7 |
| Historical stopped epoch | 17 |
| Training images | 743 mass-positive + 186 fixed negatives |
| Validation / test images | 151 / 219 mass-positive images |

## Evaluate and infer

Use the best checkpoint printed by training. Pretrained detector initialization
weights download from the URLs in the recipes on first use. **Trained study
checkpoints are not bundled or hosted by this repository**; use your training
output or a historical checkpoint supplied separately.

```bash
vindr evaluate --preset paper-best \
  --data-root /path/to/global-pad-data \
  --checkpoint runs/dino-convnext-s-1600/best_coco_bbox_mass_positive_mAP_50_epoch_7.pth \
  --work-dir runs/dino-convnext-s-1600-test

vindr infer --preset paper-best \
  --checkpoint /path/to/best.pth \
  --input /path/to/image.pt \
  --output predictions.json
```

The checkpoint filename above is the historical run's name; a retraining run may
select a different epoch. `evaluate` writes `metrics.json` and MMDetection COCO
predictions. `--split val` evaluates the checkpoint-selection endpoint. The test
endpoint applies final NMS at IoU 0.5; this is part of the reported result.

`infer` accepts one or more `.pt` tensors, `.dcm`/`.dicom` files, or directories.
It writes `Mass` scores and `[x1,y1,x2,y2]` boxes. For tensors, coordinates refer
to the exported image; for DICOMs, they are mapped back to the source image.
`--score-threshold 0.25` controls inference output filtering and does not alter
the evaluation protocol. CPU inference is available with `--device cpu`.

## Other paper models

```bash
vindr list --model cascade_rcnn
vindr show --preset fcos-h1024_w832
vindr train --preset ultralytics_yolo11-h640_w512 \
  --data-root /path/to/global-pad-data --work-dir runs/yolo11-640
```

There are **107 validation-selected presets** covering all 18 systems: Cascade
R-CNN, TOOD, FCOS, and DINO crossed with ResNet-50, Swin-T, HRNet-W32, and
ConvNeXt-S; plus RTMDet-M and YOLO11-M. Resolutions are 512×416, 640×512, 800×640,
1024×832, 1280×1024, and 1600×1280. DINO/HRNet-W32 at 1600×1280 has no completed
test result and no preset. The [parameter table](docs/selected_models.csv) lists
each selected run, LR, batch, accumulation, precision, and measured scores.

This catalog was extracted from **558 runs / 534 completed test results** on
2026-09-23. Selection uses validation mAP50 and the original run-ID tie-break,
without the manuscript figure's test-score cutoff. The run audit is included as
[CSV](src/vindr_isbi/assets/sweep_runs.csv) and
[JSON](src/vindr_isbi/assets/sweep_audit.json).

## Reproducibility and scope

The saved settings, actual retry batch sizes, initialization URLs, augmentation
code, splits, checkpoint hashes, and numerical package versions are preserved.
MMDetection's original `deterministic=False` is preserved too: a fixed seed does
**not** guarantee bit-identical retraining. For the strongest match to the
reported result, evaluate the original checkpoint with the verified dataset and
runtime. `vindr verify-checkpoint --checkpoint /path/to/best.pth` checks its
historical SHA-256. See [the detailed audit](docs/reproducibility.md), including
differences between the prose and YOLO's actual recorded behavior.

The endpoint measures localization on mass-positive mammograms; it does not
measure screening performance on mass-negative test images. Dataset access and
framework/weight licenses remain with their respective providers. See
[source acknowledgments](NOTICE.md).
