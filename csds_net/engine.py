"""Training utilities independent of the private benchmark workspace."""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FIELDS = ["epoch", "lr", "train_loss", "train_dice", "train_iou", "train_precision",
          "train_recall", "train_accuracy", "val_loss", "val_dice", "val_iou",
          "val_precision", "val_recall", "val_accuracy", "elapsed_seconds",
          "gflops", "params_m", "fps"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    if state is None:
        return  # Legacy checkpoints did not record RNG state.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def run_epoch(model, loader, optimizer, scaler, device, amp_enabled):
    """Original BCE+Dice reference path, retained for checkpoint compatibility."""
    training = optimizer is not None
    model.train(training)
    tp = fp = fn = tn = pixels = loss_sum = samples = 0.0
    for batch in loader:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            probability = logits.sigmoid()
            dice = 1 - (2 * (probability * masks).sum((2, 3)) + 1) / ((probability + masks).sum((2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        pred, truth = probability.detach() >= 0.5, masks.bool()
        tp += float((pred & truth).sum())
        fp += float((pred & ~truth).sum())
        fn += float((~pred & truth).sum())
        tn += float((~pred & ~truth).sum())
        pixels += masks.numel()
        samples += len(images)
        loss_sum += float(loss.detach()) * len(images)
    eps = 1e-7
    return {"loss": loss_sum / samples, "precision": (tp+eps)/(tp+fp+eps),
            "recall": (tp+eps)/(tp+fn+eps), "dice": (2*tp+eps)/(2*tp+fp+fn+eps),
            "iou": (tp+eps)/(tp+fp+fn+eps), "accuracy": (tp+tn+eps)/(pixels+eps)}


class FiniteCriterion:
    def __init__(self, criterion, validation):
        self.criterion, self.validation = criterion, validation

    def __call__(self, outputs, masks):
        if self.validation:
            with torch.autocast(device_type=masks.device.type, enabled=False):
                result = self.criterion({k:v.float() for k,v in outputs.items()}, masks.float())
        else:
            result = self.criterion(outputs, masks)
        if not torch.isfinite(result["total"]):
            raise FloatingPointError("Non-finite loss; epoch will not be recorded")
        return result


class SafeValidationNet(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.calls, self.retries = 0, []

    def forward(self, images):
        batch = self.calls
        self.calls += 1
        outputs = self.network(images)
        if not all(torch.isfinite(value).all() for value in outputs.values()):
            self.retries.append(batch)
            with torch.autocast(device_type=images.device.type, enabled=False):
                outputs = self.network(images.float())
            if not all(torch.isfinite(value).all() for value in outputs.values()):
                raise FloatingPointError("Validation remains non-finite after FP32 retry")
        return outputs


def safe_epoch(epoch_function, model, loader, optimizer, scaler, device, amp_enabled,
               criterion, record_tolerance2, output: Path, number: int):
    validation = optimizer is None
    finite = FiniteCriterion(criterion, validation)
    if not validation:
        return epoch_function(model, loader, optimizer, scaler, device, amp_enabled,
                              finite, record_tolerance2)
    network = model.net
    proxy = SafeValidationNet(network)
    model.net = proxy
    try:
        result = epoch_function(model, loader, None, scaler, device, amp_enabled,
                                finite, record_tolerance2)
    finally:
        model.net = network  # Preserve historical net.* checkpoint names.
    with (output / "validation_numeric_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"epoch": number, "fp32_retry_batches": proxy.retries,
                                 "validation_loss_dtype": "float32"}) + "\n")
    return result
