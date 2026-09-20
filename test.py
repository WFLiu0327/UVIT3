"""Evaluate a UVIT3 checkpoint or segment one RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from uvit3 import build_uvit3
from utils import (
    MEAN, STD, SEEDS, DATASETS, SegmentationDataset, audit_dataset,
    evaluate, get_split, save_csv, save_json, set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="可选；默认加载 weights/<dataset>/seed<seed>/best.pth")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default=None,
                        help="选择随仓库提供的权重；默认 BUSI")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None,
                        help="选择随仓库提供的权重；默认 2981")
    parser.add_argument("--image", type=Path, help="单张 RGB 图像；不指定时评估固定验证集")
    parser.add_argument("--output", type=Path, help="单张图像的预测掩膜路径")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.image and not args.output:
        parser.error("--image 需要同时指定 --output")
    if not args.image and args.output:
        parser.error("--output 仅用于 --image")

    checkpoint_path = args.checkpoint or (
        Path(__file__).resolve().parent / "weights" / (args.dataset or "BUSI")
        / f"seed{args.seed or 2981}" / "best.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("model") != "uvit3":
        raise ValueError("checkpoint model must be uvit3")
    seed = int(config["seed"])
    dataset = str(config["dataset"]).upper()
    if seed not in SEEDS or dataset not in DATASETS:
        raise ValueError("checkpoint does not use the fixed dataset and seed protocol")
    if args.dataset is not None and args.dataset != dataset:
        raise ValueError("--dataset does not match checkpoint dataset")
    if args.seed is not None and args.seed != seed:
        raise ValueError("--seed does not match checkpoint seed")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_uvit3().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.image:
        image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(args.image)
        original_size = (image_bgr.shape[1], image_bgr.shape[0])
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        array = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))[None].to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
        mask = (cv2.resize(probability, original_size, interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8) * 255
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), mask):
            raise OSError(f"could not write {args.output}")
        print(f"prediction={args.output}")
        return

    audit = audit_dataset(dataset, seed)
    if not audit["valid"]:
        raise RuntimeError(f"dataset audit failed: {audit}")
    _, val_ids, digest = get_split(dataset, seed)
    if config.get("split_sha256") != digest:
        raise RuntimeError("split_sha256 does not match the checkpoint")
    loader = DataLoader(SegmentationDataset(dataset, val_ids, train=False), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    destination = Path(__file__).resolve().parent / "output" / dataset / f"seed{seed}"
    summary, rows = evaluate(model, loader, device, destination / "predictions")
    summary.update({"dataset": dataset, "seed": seed, "split_sha256": digest,
                    "checkpoint": str(checkpoint_path.resolve()), "data_audit": audit})
    save_json(destination / "metrics.json", summary)
    save_csv(destination / "cases.csv", rows)
    print(f"metrics={destination / 'metrics.json'}")
    print(f"dice={summary['dice']:.4f} iou={summary['iou']:.4f}")


if __name__ == "__main__":
    main()
