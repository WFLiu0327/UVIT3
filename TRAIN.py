"""统一训练入口：所有模型使用项目冻结的三个随机种子划分。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from uvit3 import build_uvit3
from utils import (
    BCEDiceLoss,
    DATASETS,
    DATA_ROOT,
    SEEDS,
    audit_dataset,
    evaluate,
    make_loaders,
    save_csv,
    save_json,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs"
if run_root_override := os.environ.get("UVIT3_RUN_ROOT"):
    RUN_ROOT = Path(run_root_override).expanduser().resolve()
else:
    RUN_ROOT = DEFAULT_RUN_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("uvit3",), default="uvit3")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="BUSI")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=2981)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--run-dir", type=Path, default=None, help="自定义新的空运行目录")
    args = parser.parse_args()
    protocol = {
        "epochs": 200,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "early_stop_patience": 20,
    }
    for setting, expected in protocol.items():
        if getattr(args, setting) != expected:
            parser.error(f"--{setting.replace('_', '-')} is fixed at {expected} by the UVIT3 protocol")
    if not 0 <= args.min_lr <= args.lr:
        parser.error("--min-lr 必须位于 0 和 --lr 之间")
    if args.weight_decay < 0:
        parser.error("--weight-decay 不能为负数")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs 必须位于 0 和 epochs-1 之间")
    if args.early_stop_patience < 1:
        parser.error("--early-stop-patience 必须至少为 1")

    set_seed(args.seed)
    audit = audit_dataset(args.dataset, args.seed)
    if not audit["valid"]:
        raise RuntimeError(f"数据检查失败：{audit}")
    train_loader, val_loader, digest = make_loaders(args.dataset, args.batch_size, args.workers, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_uvit3().to(device)
    amp_enabled = device.type == "cuda" and not getattr(model, "force_fp32", False)
    precision = "amp_fp16" if amp_enabled else "fp32"
    criterion = BCEDiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_factor = args.min_lr / args.lr

    def lr_factor(epoch: int) -> float:
        if args.warmup_epochs and epoch < args.warmup_epochs:
            return float(epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs - 1, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_factor + (1 - min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run_dir = args.run_dir or RUN_ROOT / args.model / args.dataset / f"seed{args.seed}"
    print(f"data_root={DATA_ROOT.resolve()} run_dir={run_dir.resolve()}")
    start_epoch, best_iou, no_improve_epochs = 0, -math.inf, 0
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"已有训练产物：{run_dir}；best-only 协议不自动续训，请保留该目录并用 "
            "--run-dir 指定一个新的空目录从头训练"
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "model_source": getattr(model, "source_project", "UVIT3"),
        "model_source_class": getattr(model, "source_name", type(model).__name__),
        "model_source_commit": getattr(model, "source_commit", None),
        "dataset": args.dataset,
        "seed": args.seed,
        "split_sha256": digest,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": "linear_warmup_cosine",
        "early_stopping_patience": args.early_stop_patience,
        "early_stopping_metric": "val_iou",
        "checkpoint_policy": "best_only_no_auto_resume",
        "protocol": "UVIT3_fixed_200_epoch_official_TTT",
        "precision": precision,
        "loss": "BCE+Dice",
        "image_size": 256,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "data_root": str(DATA_ROOT.resolve()),
        "run_dir": str(run_dir.resolve()),
        "data_audit": audit,
    }
    save_json(run_dir / "config.json", config)
    log_path = run_dir / "log.csv"
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_dice", "val_iou", "val_hd95", "seconds"]
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            begin = time.perf_counter()
            model.train()
            total_loss, samples = 0.0, 0
            progress = tqdm(train_loader, desc=f"{args.model} {epoch + 1}/{args.epochs}", leave=False)
            for images, masks, _ in progress:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    loss = criterion(model(images), masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * images.shape[0]
                samples += images.shape[0]
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            train_loss = total_loss / samples
            val_summary, val_rows = evaluate(model, val_loader, device)
            improved = val_summary["iou"] > best_iou
            if improved:
                best_iou = val_summary["iou"]
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            row = {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_summary["loss"],
                "val_dice": val_summary["dice"],
                "val_iou": val_summary["iou"],
                "val_hd95": val_summary["hd95"],
                "seconds": time.perf_counter() - begin,
            }
            writer.writerow(row)
            handle.flush()
            scheduler.step()
            if improved:
                best_state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_iou": best_iou,
                    "config": config,
                }
                best_path = run_dir / "best.pth"
                temporary_best_path = run_dir / "best.pth.tmp"
                torch.save(best_state, temporary_best_path)
                temporary_best_path.replace(best_path)
                save_json(run_dir / "best.json", val_summary)
                save_csv(run_dir / "best_cases.csv", val_rows)
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.4f} "
                f"val_dice={val_summary['dice']:.4f} val_iou={val_summary['iou']:.4f} "
                f"early_stop={no_improve_epochs}/{args.early_stop_patience}"
            )
            if no_improve_epochs >= args.early_stop_patience:
                print(
                    f"EARLY STOP epoch={epoch + 1}: val_iou 连续 "
                    f"{args.early_stop_patience} 个 epoch 未提升，best_iou={best_iou:.4f}"
                )
                break


if __name__ == "__main__":
    main()
