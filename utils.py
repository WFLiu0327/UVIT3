"""Data, loss, metrics, reproducibility, and I/O shared by TRAIN.py and test.py."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
SEEDS = (2981, 6142, 1187)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
if data_root_override := os.environ.get("UVIT3_DATA_ROOT"):
    DATA_ROOT = Path(data_root_override).expanduser().resolve()
else:
    DATA_ROOT = DEFAULT_DATA_ROOT
DATASETS = {
    "BUSI": {
        "folder": "BUSI",
        "image_extension": ".png",
        "mask_suffix": "_mask",
        "mask_extension": ".png",
        "size": 256,
    },
    "GLAS": {
        "folder": "GlaS",
        "image_extension": ".png",
        "mask_suffix": "",
        "mask_extension": ".png",
        "size": 256,
    },
    "CVC": {
        "folder": "CVC-ClinicDB",
        "image_extension": ".png",
        "mask_suffix": "",
        "mask_extension": ".png",
        "size": 256,
    },
}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
METRICS = ("dice", "iou", "precision", "recall", "specificity", "accuracy", "hd95", "assd")


def set_seed(seed: int) -> None:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataset_info(name: str) -> dict[str, Any]:
    key = name.upper()
    if key not in DATASETS:
        raise ValueError(f"dataset must be one of {tuple(DATASETS)}")
    return DATASETS[key]


def dataset_dir(name: str) -> Path:
    return DATA_ROOT / dataset_info(name)["folder"]


def image_ids(name: str) -> list[str]:
    meta = dataset_info(name)
    paths = sorted((dataset_dir(name) / "images").glob(f"*{meta['image_extension']}"))
    if not paths:
        raise FileNotFoundError(f"no training images found for {name} under {DATA_ROOT}")
    return [path.stem for path in paths]


def split_hash(train_ids: list[str], val_ids: list[str]) -> str:
    payload = json.dumps({"train": train_ids, "val": val_ids}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_split(name: str, seed: int) -> tuple[list[str], list[str], str]:
    """Return one of the three frozen 80/20 splits used by every model."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    root = dataset_dir(name)
    path = root / f"split_seed{seed}.json"
    all_ids = image_ids(name)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少第一版冻结划分：{path}；为保证公平比较，本项目不再自动重新划分数据"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_ids, val_ids = list(payload["train"]), list(payload["val"])
    if set(train_ids) & set(val_ids) or set(train_ids + val_ids) != set(all_ids):
        raise RuntimeError(f"invalid frozen split: {path}")
    return train_ids, val_ids, split_hash(train_ids, val_ids)


def audit_dataset(name: str, seed: int) -> dict[str, Any]:
    root = dataset_dir(name)
    meta = dataset_info(name)
    ids = image_ids(name)
    missing, empty, non_rgb = [], [], []

    def inspect_case(case_id: str) -> tuple[str, bool, bool, bool]:
        image_path = root / "images" / f"{case_id}{meta['image_extension']}"
        mask_path = (
            root
            / "masks"
            / "0"
            / f"{case_id}{meta['mask_suffix']}{meta['mask_extension']}"
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (
            case_id,
            mask is None,
            mask is not None and not np.any(mask > 0),
            image is None or image.ndim != 3 or image.shape[2] != 3,
        )

    # Parallel decoding keeps the full audit efficient at training startup.
    with ThreadPoolExecutor(max_workers=min(8, len(ids))) as executor:
        inspections = executor.map(inspect_case, ids)
        for case_id, mask_missing, mask_empty, image_non_rgb in inspections:
            if mask_missing:
                missing.append(case_id)
            elif mask_empty:
                empty.append(case_id)
            if image_non_rgb:
                non_rgb.append(case_id)
    train_ids, val_ids, digest = get_split(name, seed)
    return {
        "dataset": name.upper(),
        "images": len(ids),
        "train": len(train_ids),
        "val": len(val_ids),
        "seed": seed,
        "split_sha256": digest,
        "missing_masks": missing,
        "empty_masks": empty,
        "non_rgb_images": non_rgb,
        "valid": not missing and not empty and not non_rgb,
    }


class SegmentationDataset(Dataset):
    def __init__(self, name: str, ids: list[str], train: bool, image_size: int | None = None) -> None:
        self.name = name.upper()
        self.root = dataset_dir(name)
        self.meta = dataset_info(name)
        self.ids = ids
        self.train = train
        self.image_size = image_size or int(self.meta["size"])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.ids[index]
        image = cv2.imread(
            str(self.root / "images" / f"{case_id}{self.meta['image_extension']}"),
            cv2.IMREAD_COLOR,
        )
        mask = cv2.imread(
            str(
                self.root
                / "masks"
                / "0"
                / f"{case_id}{self.meta['mask_suffix']}{self.meta['mask_extension']}"
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or mask is None:
            raise FileNotFoundError(f"cannot read image/mask pair: {case_id}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            rotations = random.randrange(4)
            if rotations:
                image, mask = np.rot90(image, rotations).copy(), np.rot90(mask, rotations).copy()
            if random.random() < 0.5:
                image, mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
            if random.random() < 0.5:
                image, mask = np.flip(image, 0).copy(), np.flip(mask, 0).copy()
        size = (self.image_size, self.image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        image = np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1))
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        mask = np.ascontiguousarray((mask > 0).astype(np.float32)[None])
        return torch.from_numpy(image), torch.from_numpy(mask), case_id


def make_loaders(name: str, batch_size: int, workers: int, seed: int) -> tuple[DataLoader, DataLoader, str]:
    train_ids, val_ids, digest = get_split(name, seed)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": _worker_seed,
        "generator": generator,
    }
    train_loader = DataLoader(
        SegmentationDataset(name, train_ids, train=True),
        shuffle=True,
        drop_last=len(train_ids) >= batch_size,
        **common,
    )
    val_loader = DataLoader(SegmentationDataset(name, val_ids, train=False), shuffle=False, **common)
    return train_loader, val_loader, digest


class BCEDiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        probability, target_flat = torch.sigmoid(logits).flatten(1), target.flatten(1)
        intersection = (probability * target_flat).sum(1)
        dice = (2 * intersection + 1e-5) / (probability.sum(1) + target_flat.sum(1) + 1e-5)
        return bce + 1 - dice.mean()


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask ^ ndimage.binary_erosion(mask) if mask.any() else mask


def case_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction.detach().cpu().numpy().astype(bool)
    true = target.detach().cpu().numpy().astype(bool)
    intersection = float(np.logical_and(pred, true).sum())
    pred_sum, true_sum = float(pred.sum()), float(true.sum())
    union = float(np.logical_or(pred, true).sum())
    total = float(pred.size)
    false_positive = pred_sum - intersection
    false_negative = true_sum - intersection
    true_negative = total - intersection - false_positive - false_negative
    pred_surface, true_surface = _surface(pred), _surface(true)
    if not pred_surface.any() and not true_surface.any():
        distances = np.asarray([0.0])
    elif not pred_surface.any() or not true_surface.any():
        distances = np.asarray([np.nan])
    else:
        to_pred = ndimage.distance_transform_edt(~pred_surface)
        to_true = ndimage.distance_transform_edt(~true_surface)
        distances = np.concatenate([to_true[pred_surface], to_pred[true_surface]])
    return {
        "dice": (2 * intersection + 1e-7) / (pred_sum + true_sum + 1e-7),
        "iou": (intersection + 1e-7) / (union + 1e-7),
        "precision": (intersection + 1e-7) / (pred_sum + 1e-7),
        "recall": (intersection + 1e-7) / (true_sum + 1e-7),
        "specificity": (true_negative + 1e-7) / (true_negative + false_positive + 1e-7),
        "accuracy": (intersection + true_negative + 1e-7) / (total + 1e-7),
        "hd95": float(np.nanpercentile(distances, 95)) if np.isfinite(distances).any() else float("nan"),
        "assd": float(np.nanmean(distances)) if np.isfinite(distances).any() else float("nan"),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prediction_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    criterion = BCEDiceLoss().to(device)
    rows, losses = [], []
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for images, masks, case_ids in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        amp_enabled = device.type == "cuda" and not getattr(model, "force_fp32", False)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
        probabilities = torch.sigmoid(logits)
        losses.extend([float(loss)] * images.shape[0])
        for index, case_id in enumerate(case_ids):
            prediction = probabilities[index, 0] >= 0.5
            rows.append({"case_id": case_id, **case_metrics(prediction, masks[index, 0] > 0.5)})
            if prediction_dir:
                cv2.imwrite(str(prediction_dir / f"{case_id}.png"), prediction.byte().cpu().numpy() * 255)
    summary = {"loss": float(np.mean(losses))}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = float(np.nanmean(values))
    return summary, rows


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
