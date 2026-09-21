# UVIT³: Cross-Scale Feature Memory with Vision Test-Time Training for Medical Image Segmentation

This repository releases the **UVIT3** model and code for training and evaluating it on BUSI, CVC-ClinicDB, and GlaS. `uvit3.py` contains the complete UVIT3 network; `ttt_block.py` is the unchanged original ViT³ `TTT` implementation that it imports. `TRAIN.py`, `test.py`, and `utils.py` provide training, validation, and single-image inference. Run every command below from the repository root on Ubuntu Bash.

The results in this repository are **validation-split results**. The same validation splits selected the best checkpoints; they are not independent test-set results.

## Quick start: download and install

```bash
git clone https://github.com/WFLiu0327/UVIT3.git
cd UVIT3
conda env create -f environment.yml
conda activate uvit3a
```

Download these archives into the repository root (beside `TRAIN.py`). The data, checkpoint binaries, and prediction images are distributed separately from Git.

| Archive | Google Drive link | Purpose |
| --- | --- | --- |
| `data.zip` | [Preprocessed BUSI, CVC-ClinicDB, and GlaS data](https://drive.google.com/file/d/1TdJhJN99DFySzddhRfIkGCo3gzcaXPH7/view?usp=sharing) | Training and validation |
| `weight.zip` | [Nine pretrained checkpoints](https://drive.google.com/file/d/1RPM6v6eCOphA5S1wmRSbRRvWh7AL0aMT/view?usp=sharing) | Pretrained inference and validation |
| `output.zip` | [Reference inference outputs](https://drive.google.com/file/d/1z5OX-hEwxqHJZ3t4qhK7d2hTS58TzveP/view?usp=sharing) | Optional: released metrics, case CSVs, and masks |

In a fresh clone, extract the downloaded archives from the repository root:

```bash
unzip -nq data.zip -d .
unzip -nq weight.zip -d .
# Optional: unzip -nq output.zip -d .
```

The `-n` option preserves files already tracked in Git. After extraction, check that `data/BUSI/images/`, `data/CVC-ClinicDB/images/`, `data/GlaS/images/`, and `weights/BUSI/seed2981/best.pth` exist, along with the matching mask and weight directories for every dataset and seed. If an archive has an extra enclosing directory, move its `data/` or `weights/` directory to the repository root. The nine `split_seed<seed>.json` files and `weights/manifest.json` are already tracked in Git; do not regenerate the splits.

## Layout

```text
UVIT3/
├── uvit3.py             # UVIT3 network in one file
├── ttt_block.py         # original ViT³ TTT block
├── TRAIN.py             # single training task
├── test.py              # validation or single-image prediction
├── utils.py             # shared data, loss, metrics, and split checks
├── data/
│   ├── BUSI/
│   ├── CVC-ClinicDB/
│   └── GlaS/
├── environment.yml
├── weights/             # nine pretrained best checkpoints, plus manifest.json
├── LICENSE
├── THIRD_PARTY_NOTICES.md
└── requirements.txt
```

Each dataset directory contains `images/`, `masks/0/`, and three frozen `split_seed<seed>.json` files. The split JSON files are in Git; `data.zip` supplies the preprocessed images and masks. The image and mask files are excluded from Git.

## Environment

Requires Ubuntu, Python 3.11, a compatible NVIDIA driver, and a CUDA GPU for full training. The pinned baseline is PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128. The quick start above installs the environment from `environment.yml`.

Before full training, check the driver and PyTorch CUDA access:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## Data and fixed protocol

The preprocessed images must have these exact names and paired mask conventions:

| CLI dataset | Directory | Images | Image | Mask in `masks/0/` | Validation cases per seed |
| --- | --- | ---: | --- | --- | ---: |
| `BUSI` | `BUSI` | 647 lesions | `<id>.png` | `<id>_mask.png` | 130 |
| `CVC` | `CVC-ClinicDB` | 612 RGB | `<id>.png` | `<id>.png` | 123 |
| `GLAS` | `GlaS` | 165 | `<id>.png` | `<id>.png` | 33 |

BUSI excludes 133 normal images. CVC stays three-channel RGB. Images are read as RGB, resized to 256×256 with linear interpolation, and ImageNet-normalized. Masks are resized with nearest-neighbor interpolation and binarized. Training augmentation is random 90-degree rotations and independent horizontal and vertical flips.

All datasets use seeds **2981, 6142, 1187** and the supplied frozen split files. `get_split` verifies that train and validation IDs are disjoint and cover every image; training records a `split_sha256`, and validation checks it against the checkpoint. Run a full data and weight audit after extraction:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from utils import audit_dataset

manifest = json.loads(Path('weights/manifest.json').read_text())
expected_images = {'BUSI': 647, 'CVC': 612, 'GLAS': 165}
for dataset in ('BUSI', 'CVC', 'GLAS'):
    for seed in (2981, 6142, 1187):
        entry = manifest['datasets'][dataset][str(seed)]
        path = Path(entry['path'])
        assert path.is_file(), f'Missing checkpoint: {path}'
        with path.open('rb') as checkpoint:
            digest = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
        assert digest == entry['sha256'], f'Checkpoint hash mismatch: {path}'
        report = audit_dataset(dataset, seed)
        assert report['valid'], f'Data audit failed: {dataset} seed{seed}: {report}'
        assert report['images'] == expected_images[dataset]
        assert report['split_sha256'] == entry['split_sha256']
        print(dataset, seed, report['images'], 'OK')
PY
```

The audit rejects missing or empty masks and images that are not three-channel. It decodes every image and mask, so it can take a few minutes. The expected image counts are in the table above. You may set `UVIT3_DATA_ROOT` to a different directory containing those three named dataset directories and identical splits.

## Training

The fixed maximum is 200 epochs, AdamW with learning rate 1e-4 and weight decay 1e-4, 5-epoch linear warm-up followed by cosine decay to 1e-5, BCE + Dice loss, gradient clipping at 1.0, and early stopping on validation IoU with patience 20. CUDA training uses FP16 AMP, as in the original `uvit3` experiment. The prediction threshold is 0.5. Change only batch size and workers for GPU capacity.

Train one task:

```bash
CUDA_VISIBLE_DEVICES=0 python TRAIN.py --model uvit3 --dataset BUSI --seed 2981 --batch-size 8 --workers 4
```

Repeat the same command for the remaining seeds and datasets, changing only `--dataset` and `--seed`. For example:

```bash
for dataset in BUSI CVC GLAS; do
  for seed in 2981 6142 1187; do
    CUDA_VISIBLE_DEVICES=0 python TRAIN.py --model uvit3 --dataset "$dataset" --seed "$seed" --batch-size 8 --workers 4
  done
done
```

Outputs are written under `runs/uvit3/<dataset>/seed<seed>/`: `config.json`, `log.csv`, `best.pth`, `best.json`, and `best_cases.csv`. Only a new best validation IoU atomically updates `best.pth`. It contains model weights and config, without optimizer state. An existing nonempty run directory is never overwritten or automatically resumed; use `--run-dir` with a new empty directory to restart. `UVIT3_RUN_ROOT` can move the output root to another disk. Keep 200-epoch and 400-epoch runs separate.

## Pretrained weights

Download `weight.zip` from the link above to obtain nine `weights/<dataset>/seed<seed>/best.pth` checkpoints for BUSI, CVC, and GLAS. `weights/manifest.json` records each checkpoint's SHA-256, frozen split hash, selected epoch, and validation IoU. The released checkpoint metadata uses the public model name `uvit3`; its learned parameter tensors are unchanged. Checkpoint binaries are excluded from Git. The audit above verifies every released checkpoint against the manifest.

## Validation and inference

Evaluate the fixed validation split of a saved checkpoint. This verifies the saved `split_sha256` and writes per-case masks and all standard metrics (Dice, IoU, Precision, Recall, Specificity, Accuracy, HD95, ASSD):

```bash
python test.py --dataset BUSI --seed 2981 --batch-size 8 --workers 4
```

Results are written to `output/BUSI/seed2981/metrics.json`, `cases.csv`, and `predictions/`.

The evaluated pretrained checkpoints have per-seed CSV files and three-seed mean ± standard deviation tables in [`results/`](results/README.md). The reference evaluation used **CPU FP32**, batch size 1, and threshold 0.5. To reproduce all nine validation outputs in the same setting:

```bash
for dataset in BUSI CVC GLAS; do
  for seed in 2981 6142 1187; do
    CUDA_VISIBLE_DEVICES='' python test.py --dataset "$dataset" --seed "$seed" --batch-size 1 --workers 0
  done
done
```

Each seed metric is the mean over its validation cases. HD95 and ASSD are measured in pixels on the 256×256 evaluation masks. Minor numerical differences are possible on a different software or hardware setup. The three-seed mean ± sample standard deviation of the released results is:

| Dataset | Dice | IoU |
| --- | ---: | ---: |
| BUSI | 0.8103 ± 0.0283 | 0.7295 ± 0.0342 |
| CVC-ClinicDB | 0.9281 ± 0.0074 | 0.8753 ± 0.0096 |
| GlaS | 0.9222 ± 0.0018 | 0.8624 ± 0.0025 |

See [`results/README.md`](results/README.md) for all eight metrics and per-seed values. The optional `output.zip` contains the corresponding `output/<dataset>/seed<seed>/metrics.json`, `cases.csv`, and `predictions/` files. Newly generated `metrics.json` files may record different absolute checkpoint paths on another machine.

Predict one RGB image and restore the mask to the input image size:

```bash
python test.py --dataset BUSI --seed 2981 \
  --image 'data/BUSI/images/benign (16).png' --output output/example_mask.png
```

Replace the image path with any readable image. The output is a binary PNG (0 or 255). If `--dataset` and `--seed` are omitted, `test.py` loads `weights/BUSI/seed2981/best.pth`; use `--checkpoint` to load another checkpoint explicitly.

<!-- ## Provenance

The architecture in `uvit3.py` is consolidated from the parent project's model module and its required shared modules in `lib/UVIT3.py`; parameter names and computation are unchanged. `ttt_block.py` is copied byte-for-byte from `lib/ttt_block.py`, which records the upstream ViT³ authorship in its header. The data, loss, metrics, training, and checkpoint rules are adapted from the parent project's unified `lib/utils.py` and `TRAIN.py`.

The code is released under MIT. The copied ViT³ component and its original copyright notice are identified in `THIRD_PARTY_NOTICES.md`. -->

## Citation

If you use this repository, please cite **UVIT³: Cross-Scale Feature Memory with Vision Test-Time Training for Medical Image Segmentation**. Bibliographic details will be added when available.
