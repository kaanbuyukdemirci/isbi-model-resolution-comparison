# Validation performed on 2026-09-23

The extracted package was tested in the available Python 3.10.20 / PyTorch
2.1.0+cu121 / MMDetection 3.3.0 / Ultralytics 8.4.91 environment on RTX 3090
hardware. An isolated virtual environment reused these installed framework
packages; this was not a fresh download and installation of the entire ML stack.

| Check | Result |
|---|---|
| Contract tests | 8 passed |
| Ruff checks | Passed |
| Frozen configuration comparison | All 101 MMDetection presets matched their original model, optimizer, hooks, schedules, randomness, pipelines and loader settings |
| Model construction | All 17 MMDetection architecture/backbone combinations built successfully |
| DINO/ConvNeXt-S, 1600×1280, original checkpoint | All 219 test images: mAP50 **0.709**, mAP50:95 **0.354**, mAP75 **0.312**, matching the historical result |
| YOLO11-M, 640×512, original checkpoint | All 219 test images: native mAP50 **0.6365722146717495**, exactly matching the saved run |
| DINO training smoke check | Two iterations with gradient accumulation and one optimizer update completed with finite losses at 512×416 |
| YOLO training smoke check | One epoch on two positive training and two validation images completed |
| Tensor inference | Both backends produced JSON detections using the installed console command outside this source directory |
| Annotation regeneration | All 18 COCO files (three splits × six resolutions) matched the historical SHA-256 hashes |
| Pixel regeneration | A real source DICOM reproduced its saved tensor values exactly at all six resolutions; the raw-DICOM inference loader produced the same model input |
| Packaging | Wheel built with all 107 recipes and data-contract assets; console entry point installed successfully |
| Independence | Architecture validation did not import the original `vindr_mmdet` package |

These checks establish configuration fidelity, working training/inference code,
and measured evaluation parity for two original checkpoints. Full multi-epoch
retraining of all recipes was **not** performed. The smoke runs intentionally
used a tiny dataset and shorter schedule and are not accuracy experiments.

Run the small portable checks with:

```bash
python -m pip install pytest ruff
python -m pytest -q
ruff check src tests
```

The full-data evaluation commands are in the README; they require the licensed
dataset and a separately available trained checkpoint.
