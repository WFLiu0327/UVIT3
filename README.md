# UVIT3: reproducible binary medical image segmentation

This directory is a standalone release of the **UVIT3** model. `uvit3.py` contains the complete UVIT3 network; `ttt_block.py` is the unchanged official ViT³ `TTT` implementation that it imports. `TRAIN.py`, `test.py`, and `utils.py` provide training, validation, and single-image inference. Run every command below from this directory on Ubuntu Bash.

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

Each dataset directory contains `images/`, `masks/0/`, and the three frozen `split_seed<seed>.json` files. The current local `images` and `masks` entries are links to the existing preprocessed data in `../data/train/`; all three datasets are immediately available in this workspace. The image and mask entries are ignored for publication under this project's rule against committing datasets. On a fresh clone, obtain the datasets under their respective terms and populate the same paths with preprocessed images and masks. The split JSON files are included in this directory. Do not regenerate splits.

## Environment

Requires Ubuntu, Python 3.11, a compatible NVIDIA driver, and a CUDA GPU for full training. The pinned baseline is PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128. The environment files have the same versions; use one of these alternatives:

```bash
cd UVIT3
conda env create -f environment.yml
conda activate uvit3a
```

```bash
cd UVIT3
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Before full training, check the driver and PyTorch CUDA access:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## Data and fixed protocol

The preprocessed images must have these exact names and paired mask conventions:

| CLI dataset | Directory | Images | Image | Mask in `masks/0/` |
| --- | --- | ---: | --- | --- |
| `BUSI` | `BUSI` | 647 lesions | `<id>.png` | `<id>_mask.png` |
| `CVC` | `CVC-ClinicDB` | 612 RGB | `<id>.png` | `<id>.png` |
| `GLAS` | `GlaS` | 165 | `<id>.png` | `<id>.png` |

BUSI excludes 133 normal images. CVC stays three-channel RGB. Images are read as RGB, resized to 256×256 with linear interpolation, and ImageNet-normalized. Masks are resized with nearest-neighbor interpolation and binarized. Training augmentation is random 90-degree rotations and independent horizontal and vertical flips.

All datasets use seeds **2981, 6142, 1187** and the supplied frozen split files. `get_split` verifies that train and validation IDs are disjoint and cover every image; training records a `split_sha256`, and validation checks it against the checkpoint. Run a full audit before training:

```bash
python - <<'PY'
from utils import audit_dataset
for dataset in ('BUSI', 'CVC', 'GLAS'):
    for seed in (2981, 6142, 1187):
        report = audit_dataset(dataset, seed)
        print(dataset, seed, report['images'], report['split_sha256'], report['valid'])
        assert report['valid']
PY
```

The audit rejects missing or empty masks and images that are not three-channel. The expected image counts are in the table above. You may set `UVIT3_DATA_ROOT` to a different directory containing those three named dataset directories and identical splits.

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

Nine pretrained checkpoints are available locally at `weights/<dataset>/seed<seed>/best.pth` for BUSI, CVC, and GLAS with seeds 2981, 6142, and 1187. `weights/manifest.json` records each checkpoint's SHA-256, frozen split hash, selected epoch, and validation IoU. The copied checkpoint metadata uses the public model name `uvit3`; its learned parameter tensors are unchanged. The original training checkpoints remain in the parent project.

Checkpoint binaries are ignored by Git under the project rule against committing checkpoints. When distributing this public repository, provide the nine files separately with the directory layout above; recipients can verify them against `weights/manifest.json`. The commands below work directly in this local workspace.

## Validation and inference

Evaluate the fixed validation split of a saved checkpoint. This verifies the saved `split_sha256` and writes per-case masks and all standard metrics (Dice, IoU, Precision, Recall, Specificity, Accuracy, HD95, ASSD):

```bash
python test.py --dataset BUSI --seed 2981 --batch-size 8 --workers 4
```

Results are written to `output/BUSI/seed2981/metrics.json`, `cases.csv`, and `predictions/`.

The evaluated pretrained checkpoints have per-seed CSV files and three-seed mean ± standard deviation tables in [`results/`](results/README.md). To reproduce all nine validation outputs with the local weights:

```bash
for dataset in BUSI CVC GLAS; do
  for seed in 2981 6142 1187; do
    python test.py --dataset "$dataset" --seed "$seed" --batch-size 1 --workers 0
  done
done
```

Predict one RGB image and restore the mask to the input image size:

```bash
python test.py --dataset BUSI --seed 2981 \
  --image 'data/BUSI/images/benign (16).png' --output output/example_mask.png
```

Replace the image path with any readable image. The output is a binary PNG (0 or 255). If `--dataset` and `--seed` are omitted, `test.py` loads `weights/BUSI/seed2981/best.pth`; use `--checkpoint` to load another checkpoint explicitly.

## Provenance

The architecture in `uvit3.py` is consolidated from the parent project's model module and its required shared modules in `lib/UVIT3.py`; parameter names and computation are unchanged. `ttt_block.py` is copied byte-for-byte from `lib/ttt_block.py`, which records the upstream ViT³ authorship in its header. The data, loss, metrics, training, and checkpoint rules are adapted from the parent project's unified `lib/utils.py` and `TRAIN.py`.

The code is released under MIT. The copied ViT³ component and its original copyright notice are identified in `THIRD_PARTY_NOTICES.md`.
