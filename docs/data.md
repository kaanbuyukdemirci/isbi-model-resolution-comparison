# Data contract

Use VinDr-Mammo v1.0.0 with its source CSV files and DICOMs:

```text
raw-root/
  breast-level_annotations.csv
  finding_annotations.csv
  images/<study_id>/<image_id>.dicom
```

`vindr prepare --preset paper-best --raw-root RAW --output-root DATA` reads these
files, uses the bundled 5,000-exam split assignments, and recreates one selected
resolution. It writes all three COCO annotation files in their original order,
then decodes only the 929 training, 151 validation, and 219 test images. Repeating
the command resumes missing tensors. Other resolutions can be added to the same
output root by choosing a preset at that resolution.

```text
DATA/
  annotations/h1600_w1280/instances_train.json
  annotations/h1600_w1280/instances_val.json
  annotations/h1600_w1280/instances_test.json
  images/float32/h1600_w1280/train/<study_id>__<image_id>.pt
  images/float32/h1600_w1280/val/<study_id>__<image_id>.pt
  images/float32/h1600_w1280/test/<study_id>__<image_id>.pt
```

The original full annotation files contain 13,600/2,400/4,000 images. They need
not all have tensor files: the CLI reads the frozen indices to form the paper's
929/151/219-image cohort. Category ID is 1, with the case-sensitive name `Mass`.
Training includes 743 positives and 186 negatives; evaluation includes only
positives. There are 237 test boxes. The included ordered indices reproduce
seed-42 negative sampling without relying on a newly sampled dataset.

The exporter applies the DICOM modality LUT, corrects MONOCHROME1 polarity,
min-max normalizes and rounds to uint16, pads bottom/right to 3584×2816, resizes
using OpenCV `INTER_AREA`, and adds minimal right padding to a multiple of 32.
It then converts to contiguous float32 `[1,H,W]` tensors in `[0,1]`. The uint16
intermediate and rounding are intentional. Training repeats the grayscale
channel three times and scales to `[0,255]` before the recorded model
normalization; there is no uint8 conversion, CLAHE, breast crop, or mirroring.

| Export H×W | Content width before final right padding |
|---|---|
| 512×416 | 402 |
| 640×512 | 503 |
| 800×640 | 629 |
| 1024×832 | 805 |
| 1280×1024 | 1006 |
| 1600×1280 | 1257 |

Annotation SHA-256 checks cover the full export ordering and coordinates.
Reference tensor checks hash the decoded numerical values, so equivalent
`torch.save` containers are accepted. The three reference tensors per resolution
are spot checks, not a hash of every pixel file. Missing cohort images fail
before training. Modified datasets should be documented as a new experiment
rather than presented as the frozen study result.

Raw images and source annotation CSVs are not included. Obtain access through
the [official VinDr-Mammo release](https://physionet.org/content/vindr-mammo/1.0.0/).
For compressed DICOM transfer syntaxes, install a decoder supported by your
pydicom installation; the local validation used the existing source decoder.
