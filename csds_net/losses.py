from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def dice_loss(logits: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def boundary_map(mask: Tensor) -> Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        boundary_weight: float = 0.15,
        dice_weight: float = 1.0,
        pos_weight: float = 1.0,
        hard_negative_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.hard_negative_weight = hard_negative_weight
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, outputs: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        logits = outputs["logits"]
        pixel_bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        if self.hard_negative_weight > 0:
            hard_background = (target == 0) & (torch.sigmoid(logits.detach()) > 0.2)
            pixel_bce = pixel_bce * (1.0 + self.hard_negative_weight * hard_background)
        bce = pixel_bce.mean()
        dice = dice_loss(logits, target)
        edge_target = boundary_map(target)
        edge = F.binary_cross_entropy_with_logits(outputs["edge_logits"], edge_target)
        total = bce + self.dice_weight * dice + self.boundary_weight * edge
        return {"total": total, "bce": bce.detach(), "dice": dice.detach(), "edge": edge.detach()}

