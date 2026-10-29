"""Fixed-threshold bidirectional square-neighborhood crack metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F

COUNT_KEYS = ("pred_match", "pred_positive", "gt_match", "gt_positive",
              "exact_correct", "total")


@torch.no_grad()
def tolerance_counts(prediction, target, radius=2):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    pred, gt = prediction.float(), target.float()
    kernel = 2 * radius + 1
    gt_near = F.max_pool2d(gt, kernel, stride=1, padding=radius)
    pred_near = F.max_pool2d(pred, kernel, stride=1, padding=radius)
    return {"pred_match": int((pred * gt_near).sum().item()),
            "pred_positive": int(pred.sum().item()),
            "gt_match": int((gt * pred_near).sum().item()),
            "gt_positive": int(gt.sum().item()),
            "exact_correct": int((prediction == target).sum().item()),
            "total": target.numel()}


def tolerance_metrics(counts):
    pred, gt = counts["pred_positive"], counts["gt_positive"]
    precision = counts["pred_match"] / pred if pred else (1.0 if not gt else 0.0)
    recall = counts["gt_match"] / gt if gt else (1.0 if not pred else 0.0)
    dice = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "dice": dice,
            "iou_derived": dice / (2 - dice) if dice else 0.0,
            "accuracy": counts["exact_correct"] / counts["total"]}
