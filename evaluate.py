"""Evaluate one checkpoint using its fixed split, size and threshold 0.5."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from csds_net.checkpoint import forward_finite, load_model, sha256
from csds_net.data import CrackDataset
from csds_net.tolerance_metrics import COUNT_KEYS, tolerance_counts, tolerance_metrics


@torch.no_grad()
def evaluate(checkpoint, data_root, manifest_path, batch_size=8, device="cpu", amp=True):
    device = torch.device(device)
    model, config, epoch = load_model(checkpoint, device)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["split_seed"] != config.get("split_seed", 42):
        raise ValueError("Checkpoint and manifest split seeds differ")
    data = CrackDataset(data_root, "val", image_size=config.get("image_size", 256),
                        split_seed=manifest["split_seed"], train_ratio=0.8, val_ratio=0.2)
    if data.sample_ids != manifest["val"]:
        raise ValueError("Dataset files differ from the reference validation split")
    exact, tolerant = dict.fromkeys(COUNT_KEYS, 0), dict.fromkeys(COUNT_KEYS, 0)
    retries = 0
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
        images, truth = batch["image"].to(device), batch["mask"].to(device).bool()
        output, retry = forward_finite(model, images, amp and device.type == "cuda")
        retries += retry
        prediction = output["logits"].sigmoid() >= 0.5
        for accumulator, radius in ((exact, 0), (tolerant, 2)):
            for key, value in tolerance_counts(prediction, truth, radius=radius).items():
                accumulator[key] += value
    strict = tolerance_metrics(exact)
    strict["iou"] = strict.pop("iou_derived")
    return {"checkpoint_sha256": sha256(checkpoint), "checkpoint_epoch": epoch,
            "model": config.get("model", "csds_semantic"), "training_seed": config.get("seed"),
            "split_seed": manifest["split_seed"], "partition": "validation; no independent test set",
            "validation_samples": len(data), "image_size": config.get("image_size", 256),
            "threshold": 0.5, "amp": amp and device.type == "cuda", "batch_size": batch_size,
            "fp32_retry_batches": retries, "strict": strict, "tolerance2": tolerance_metrics(tolerant),
            "counts_strict": exact, "counts_tolerance2": tolerant,
            "metric_note": "tolerance2.iou_derived = Dice/(2-Dice); not strict pixel intersection-over-union"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", help="Default: splits/<data-root basename>.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", default="runs/evaluation.json")
    args = parser.parse_args()
    manifest = args.manifest or str(Path(__file__).resolve().parent / "splits" / f"{Path(args.data_root).name}.json")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    result = evaluate(args.checkpoint, args.data_root, manifest, args.batch_size, device, not args.no_amp)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
