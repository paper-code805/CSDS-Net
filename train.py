"""Train the semantic research variant with the archived matched protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from csds_net.data import CrackDataset
from csds_net.losses import SegmentationLoss
from csds_net.model import build_model, MODEL_NAMES
from csds_net.tolerance_metrics import COUNT_KEYS, tolerance_counts, tolerance_metrics
from csds_net.engine import FIELDS, run_epoch, seed_all, safe_epoch, rng_state, restore_rng_state
from profile_model import measure_complexity


class SegmentationLogits(nn.Module):
    def __init__(self, model_name: str = "csds") -> None:
        super().__init__()
        self.net = build_model(model_name, pretrained=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)["logits"]


def history_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected history fields: {path}")
        return list(reader)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


TOLERANCE_FIELDS = ["epoch"] + [f"{split}_{metric}" for split in ("train", "val")
                              for metric in ("precision", "recall", "dice", "iou_derived", "accuracy")]


def run_epoch_with_edges(model, loader, optimizer, scaler, device, amp_enabled, criterion,
                         record_tolerance2=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    samples = 0
    tp = fp = fn = tn = pixels = 0.0
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            outputs = model.net(images)
            loss = criterion(outputs, masks)["total"]
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        prediction = outputs["logits"].detach().sigmoid() >= 0.5
        truth = masks.bool()
        if record_tolerance2:
            for key, value in tolerance_counts(prediction, truth).items():
                counts[key] += value
        tp += float((prediction & truth).sum())
        fp += float((prediction & ~truth).sum())
        fn += float((~prediction & truth).sum())
        tn += float((~prediction & ~truth).sum())
        pixels += float(masks.numel())
        batch_size = images.shape[0]
        samples += batch_size
        total_loss += float(loss.detach()) * batch_size
    if samples == 0:
        raise RuntimeError("empty dataloader")
    eps = 1e-7
    result = {"loss": total_loss / samples,
            "precision": (tp + eps) / (tp + fp + eps),
            "recall": (tp + eps) / (tp + fn + eps),
            "dice": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
            "iou": (tp + eps) / (tp + fp + fn + eps),
            "accuracy": (tp + tn + eps) / (pixels + eps)}
    if record_tolerance2:
        result["tolerance2"] = tolerance_metrics(counts)
    return result


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    selected, _ = config_parser.parse_known_args()
    defaults = json.loads(Path(selected.config).read_text(encoding="utf-8")) if selected.config else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--model", choices=MODEL_NAMES, default="csds_semantic")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="runs/csds_semantic")
    parser.add_argument("--complexity-json", help="Optional measured profile; measured automatically when omitted")
    parser.add_argument("--reference-manifest", help="Default: splits/<dataset>.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--loss-mode", choices=("matched", "edge_hardneg"), default="edge_hardneg")
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--boundary-weight", type=float, default=0.15)
    parser.add_argument("--strong-augment", action="store_true")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--record-tolerance2", action="store_true", default=True)
    parser.add_argument("--no-record-tolerance2", action="store_false", dest="record_tolerance2")
    parser.add_argument("--device", default="auto")
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16 or args.workers < 0:
        parser.error("epochs/batch-size must be positive, image-size >=16 and workers >=0")
    if args.config and args.reference_manifest and not Path(args.reference_manifest).is_absolute():
        args.reference_manifest = str(Path(__file__).resolve().parent / args.reference_manifest)
    if args.record_tolerance2 and args.loss_mode != "edge_hardneg":
        parser.error("--record-tolerance2 currently requires --loss-mode edge_hardneg")
    if args.pos_weight <= 0 or args.hard_negative_weight < 0 or args.boundary_weight < 0:
        parser.error("loss weights must be nonnegative and pos-weight must be positive")

    seed_all(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    dataset = Path(args.data_root).name
    out = Path(args.output_root) / dataset / args.model / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if args.reference_manifest is None:
        args.reference_manifest = str(Path(__file__).resolve().parent / "splits" / f"{dataset}.json")
    if args.complexity_json is None:
        args.complexity_json = str(out / "profile.json")
        if not Path(args.complexity_json).exists():
            saved_rng = rng_state()
            measure_complexity(args.model, args.image_size, device, Path(args.complexity_json))
            restore_rng_state(saved_rng)
    complexity = json.loads(Path(args.complexity_json).read_text(encoding="utf-8"))[args.model]
    if not all(isinstance(complexity[key], (int, float)) and complexity[key] > 0
               for key in ("gflops", "params_m", "fps")):
        raise ValueError("CSDS complexity measurements are missing or invalid")

    config = vars(args).copy()
    config.update({
        "dataset": dataset,
        "input": "RGB",
        "pretrained": False,
        "initial_checkpoint": args.init_checkpoint,
        "split": "fixed 80/20, no independent test",
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "metric_tolerance": 0,
        "metric_threshold": 0.5,
        "loss": ("BCEWithLogits + soft Dice; main segmentation logits only"
                 if args.loss_mode == "matched" else
                 "weighted BCE + soft Dice + boundary BCE; hard-negative background weighting"),
        "edge_head_loss": args.loss_mode == "edge_hardneg",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "gradient_clip_norm": 1.0,
        "amp_enabled": amp_enabled,
        "complexity": complexity,
        "checkpoint_selection": "strict validation Dice; unchanged from seven-model protocol",
        "additional_metric_tolerance": 2 if args.record_tolerance2 else None,
    })
    config_path = out / "config.json"
    if args.resume and config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        legacy_defaults = {"loss_mode": "matched", "pos_weight": 1.0,
                           "hard_negative_weight": 0.0, "boundary_weight": 0.15,
                           "strong_augment": False, "record_tolerance2": False}
        for key in ("model", "data_root", "seed", "split_seed", "image_size", "batch_size",
                    "epochs", "lr", "weight_decay", "amp_enabled", "loss_mode", "pos_weight",
                    "hard_negative_weight", "boundary_weight", "strong_augment", "init_checkpoint",
                    "record_tolerance2"):
            if old.get(key, legacy_defaults.get(key)) != config[key]:
                raise ValueError(f"Resume mismatch for {key}: {old.get(key)} vs {config[key]}")
    elif config_path.exists():
        raise FileExistsError(f"Existing run requires --resume: {out}")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not args.resume and (out / "history.csv").exists():
        raise FileExistsError("Existing history requires --resume")

    train_set = CrackDataset(args.data_root, "train", image_size=args.image_size, train=True,
                             split_seed=args.split_seed, train_ratio=0.8, val_ratio=0.2,
                             strong_augment=args.strong_augment)
    val_set = CrackDataset(args.data_root, "val", image_size=args.image_size, train=False,
                           split_seed=args.split_seed, train_ratio=0.8, val_ratio=0.2)
    manifest = {"split_seed": args.split_seed, "train": train_set.sample_ids,
                "val": val_set.sample_ids, "test": []}
    reference = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
    if manifest != reference:
        raise ValueError(f"CSDS split differs from seven-model reference for {dataset}")
    manifest_path = out / "split_manifest.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Dataset split changed since checkpoint")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = SegmentationLogits(args.model).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial.get("config", {}).get("model") != args.model:
            raise ValueError("Initial checkpoint uses a different model")
        model.load_state_dict(initial["model"])
    criterion = (SegmentationLoss(pos_weight=args.pos_weight,
                                  hard_negative_weight=args.hard_negative_weight,
                                  boundary_weight=args.boundary_weight).to(device)
                 if args.loss_mode == "edge_hardneg" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history_path = out / "history.csv"
    tolerance_path = out / "tolerance2_history.csv"
    checkpoint_path = out / "last.pt"
    best_path = out / "best.pt"
    best_dice = -1.0
    start = 1
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"] + 1
        best_dice = checkpoint["best_dice"]
        rows = history_rows(history_path)
        if len(rows) < checkpoint["epoch"]:
            raise RuntimeError("Checkpoint is ahead of history; cannot recover recorded metrics")
        if len(rows) > checkpoint["epoch"]:
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows[:checkpoint["epoch"]])
        if args.record_tolerance2:
            with tolerance_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != TOLERANCE_FIELDS:
                    raise ValueError("Unexpected tolerance history fields")
                tolerance_rows = list(reader)
            if len(tolerance_rows) < checkpoint["epoch"]:
                raise RuntimeError("Checkpoint is ahead of tolerance history")
            with tolerance_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TOLERANCE_FIELDS)
                writer.writeheader()
                writer.writerows(tolerance_rows[:checkpoint["epoch"]])
        if not best_path.exists() or torch.load(best_path, map_location="cpu", weights_only=False)["best_dice"] < best_dice:
            if rows[checkpoint["epoch"] - 1]["val_dice"] == str(best_dice):
                save_checkpoint(best_path, checkpoint)
            else:
                raise RuntimeError("Best checkpoint missing and last checkpoint is not the best")
    elif args.resume and history_path.exists():
        raise RuntimeError("History exists but last.pt is missing")
    else:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(FIELDS)
        if args.record_tolerance2:
            with tolerance_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(TOLERANCE_FIELDS)

    print(f"model={args.model} dataset={dataset} seed={args.seed} split={len(train_set)}/{len(val_set)} "
          f"params={complexity['params_m']:.3f} gflops={complexity['gflops']:.3f} "
          f"fps={complexity['fps']:.2f} device={device}", flush=True)
    if args.resume and checkpoint_path.exists():
        restore_rng_state(checkpoint.get("rng_state"))
    run_epoch_with_edges_unchecked = run_epoch_with_edges
    for number in range(start, args.epochs + 1):
        t0 = time.monotonic()
        if criterion is None:
            train = run_epoch(model, train_loader, optimizer, scaler, device, amp_enabled)
        else:
            train = safe_epoch(run_epoch_with_edges_unchecked, model, train_loader, optimizer, scaler, device,
                               amp_enabled, criterion, args.record_tolerance2, out, number)
        with torch.no_grad():
            if criterion is None:
                val = run_epoch(model, val_loader, None, scaler, device, amp_enabled)
            else:
                val = safe_epoch(run_epoch_with_edges_unchecked, model, val_loader, None, scaler, device,
                                 amp_enabled, criterion, args.record_tolerance2, out, number)
        elapsed = time.monotonic() - t0
        row = {"epoch": number, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": train["loss"], "train_dice": train["dice"],
               "train_iou": train["iou"], "train_precision": train["precision"],
               "train_recall": train["recall"], "train_accuracy": train["accuracy"],
               "val_loss": val["loss"], "val_dice": val["dice"],
               "val_iou": val["iou"], "val_precision": val["precision"],
               "val_recall": val["recall"], "val_accuracy": val["accuracy"],
               "elapsed_seconds": elapsed, "gflops": complexity["gflops"],
               "params_m": complexity["params_m"], "fps": complexity["fps"]}
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError("Non-finite epoch metrics; epoch will not be recorded")
        scheduler.step()
        improved = val["dice"] > best_dice
        best_dice = max(best_dice, val["dice"])
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
        if args.record_tolerance2:
            tolerance_row = {"epoch": number}
            for split, metrics in (("train", train), ("val", val)):
                tolerance_row.update({f"{split}_{key}": value
                                      for key, value in metrics["tolerance2"].items()})
            with tolerance_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=TOLERANCE_FIELDS).writerow(tolerance_row)
        checkpoint = {"epoch": number, "best_dice": best_dice,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "config": config, "rng_state": rng_state()}
        save_checkpoint(checkpoint_path, checkpoint)
        if improved:
            save_checkpoint(best_path, checkpoint)
        print(f"epoch={number:03d} train_loss={train['loss']:.4f} "
              f"val_loss={val['loss']:.4f} dice={val['dice']:.4f} "
              f"iou={val['iou']:.4f} seconds={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
