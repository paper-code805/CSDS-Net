"""Checkpoint-compatible semantic research variant.

This module preserves the existing trained operators and parameter names.
Its dense gates are not the Gaussian/Top-K attention in manuscript Section 2.
See docs/IMPLEMENTATION.md for the verified architecture mapping.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Type
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class CSDSBlock(nn.Module):
    """Historical block name: dense channel and spatial gating.

    A depthwise local/context pair provides the candidate structure. A channel
    region gate and a spatial pixel gate then select the responses used by the
    interaction branch. The residual path keeps optimization stable when the
    gates are initially uncertain.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.pre = ConvNormAct(channels, hidden, kernel_size=1)
        self.local = ConvNormAct(hidden, hidden, groups=hidden)
        self.context = ConvNormAct(hidden, hidden, dilation=2, groups=hidden)
        self.region_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.pixel_gate = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.mix = ConvNormAct(hidden, channels, kernel_size=1, act=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.pre(x)
        candidate = self.local(h) + self.context(h)
        selected = candidate * self.region_gate(candidate) * self.pixel_gate(candidate)
        # This is an enhancement gate: the ungated candidate remains present.
        y = self.mix(candidate + selected)
        return self.activation(self.out_norm(y) + x)


class DirectionalCSDSBlock(CSDSBlock):
    """Aggregate elongated crack evidence along both image axes."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__(channels, expansion)
        hidden = channels * expansion
        self.horizontal = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.pre(x)
        candidate = (self.local(h) + self.context(h)
                     + 0.5 * (self.horizontal(h) + self.vertical(h)))
        selected = candidate * self.region_gate(candidate) * self.pixel_gate(candidate)
        y = self.mix(candidate + selected)
        return self.activation(self.out_norm(y) + x)


class PlainStructureBlock(nn.Module):
    """Width-matched local/context residual block used when module A is absent."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.pre = ConvNormAct(channels, hidden, kernel_size=1)
        self.local = ConvNormAct(hidden, hidden, groups=hidden)
        self.context = ConvNormAct(hidden, hidden, dilation=2, groups=hidden)
        self.mix = ConvNormAct(hidden, channels, kernel_size=1, act=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.pre(x)
        candidate = self.local(h) + self.context(h)
        y = self.mix(candidate)
        return self.activation(self.out_norm(y) + x)


class Encoder(nn.Module):
    def __init__(self, channels: Sequence[int], block_type: Type[CSDSBlock] = CSDSBlock) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvNormAct(3, c1, kernel_size=3, stride=2)
        self.stage1 = block_type(c1)
        self.down1 = ConvNormAct(c1, c2, stride=2)
        self.stage2 = block_type(c2)
        self.down2 = ConvNormAct(c2, c3, stride=2)
        self.stage3 = block_type(c3)
        self.down3 = ConvNormAct(c3, c4, stride=2)
        self.stage4 = block_type(c4)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(self.down3(f3))
        return [f1, f2, f3, f4]


class CSCA(nn.Module):
    """Cross-scale context aggregation for the first three skip features."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        self.channels = list(channels)
        self.source_projections = nn.ModuleList(
            [
                nn.ModuleList([nn.Conv2d(src, dst, 1, bias=False) for src in channels])
                for dst in channels[:-1]
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c * 2, c, 1),
                    nn.GELU(),
                    nn.Conv2d(c, c, 1),
                    nn.Sigmoid(),
                )
                for c in channels[:-1]
            ]
        )
        self.refine = nn.ModuleList([ConvNormAct(c, c, 3) for c in channels[:-1]])

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        enhanced: List[Tensor] = []
        for target_idx in range(3):
            target = features[target_idx]
            height, width = target.shape[-2:]
            aggregated = torch.zeros_like(target)
            for source_idx, source in enumerate(features):
                projected = self.source_projections[target_idx][source_idx](source)
                projected = F.interpolate(
                    projected, size=(height, width), mode="bilinear", align_corners=False
                )
                aggregated = aggregated + projected
            aggregated = aggregated / len(features)
            gate = self.gates[target_idx](torch.cat([target, aggregated], dim=1))
            enhanced.append(self.refine[target_idx](target + gate * aggregated))
        return enhanced + [features[3]]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 block_type: Type[CSDSBlock] = CSDSBlock) -> None:
        super().__init__()
        self.fuse = ConvNormAct(in_channels + skip_channels, out_channels, 3)
        self.block = block_type(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.fuse(torch.cat([x, skip], dim=1)))


class CSDSNet(nn.Module):
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256),
                 block_type: Type[CSDSBlock] = CSDSBlock) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("CSDSNet expects four channel levels")
        self.channels = tuple(channels)
        self.encoder = Encoder(channels, block_type)
        self.csca = CSCA(channels)
        self.dec3 = DecoderBlock(channels[3], channels[2], channels[2], block_type)
        self.dec2 = DecoderBlock(channels[2], channels[1], channels[1], block_type)
        self.dec1 = DecoderBlock(channels[1], channels[0], channels[0], block_type)
        # The encoder starts at half resolution. Fuse a shallow full-resolution
        # image feature before the final head so one-pixel cracks are not lost
        # during the last interpolation.
        self.input_detail = ConvNormAct(3, channels[0], 3)
        self.full_fuse = ConvNormAct(channels[0] * 2, channels[0], 3)
        self.full_refine = block_type(channels[0])
        self.head = nn.Sequential(
            ConvNormAct(channels[0], channels[0], 3),
            nn.Conv2d(channels[0], 1, 1),
        )
        self.edge_head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_size = x.shape[-2:]
        encoded = self.encoder(x)
        skips = self.csca(encoded)
        y = self.dec3(encoded[3], skips[2])
        y = self.dec2(y, skips[1])
        y = self.dec1(y, skips[0])
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        y = self.full_refine(self.full_fuse(torch.cat([y, self.input_detail(x)], dim=1)))
        logits = self.head(y)
        edge_logits = self.edge_head(
            y
        )
        return {"logits": logits, "edge_logits": edge_logits}


class SemanticGate(nn.Module):
    """Use decoder/context evidence to attenuate a texture feature.

    The default 0.25 floor preserves a route for weak thin cracks; independent
    candidates can test a lower floor. The inference threshold stays fixed.
    """

    def __init__(self, channels: int, floor: float = 0.25) -> None:
        super().__init__()
        if not 0.0 <= floor < 1.0:
            raise ValueError("semantic gate floor must be in [0, 1)")
        self.floor = float(floor)
        hidden = max(channels // 2, 8)
        self.projection = nn.Sequential(
            ConvNormAct(channels * 2, hidden, 1),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.normal_(self.projection[-1].weight, std=0.01)
        nn.init.constant_(self.projection[-1].bias, 1.0)

    def forward(self, detail: Tensor, semantic: Tensor) -> Tensor:
        gate = self.floor + (1.0 - self.floor) * torch.sigmoid(
            self.projection(torch.cat([detail, semantic], dim=1))
        )
        return detail * gate


class TopDownCSCA(nn.Module):
    """Aggregate only coarser scales, then gate each native skip feature.

    Remove self-projection (the target already has a direct path) and fine-to-
    coarse projections. Whether these removed routes were redundant must be
    established experimentally rather than assumed from the architecture.
    """

    def __init__(self, channels: Sequence[int], gate_floor: float = 0.25) -> None:
        super().__init__()
        self.source_projections = nn.ModuleList([
            nn.ModuleList([ConvNormAct(src, dst, 1)
                           for src in channels[index + 1:]])
            for index, dst in enumerate(channels[:-1])
        ])
        self.gates = nn.ModuleList([SemanticGate(c, floor=gate_floor) for c in channels[:-1]])
        self.refine = nn.ModuleList([ConvNormAct(c, c, 3) for c in channels[:-1]])

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        enhanced = []
        for index, target in enumerate(features[:-1]):
            sources = [F.interpolate(projection(source), size=target.shape[-2:],
                                     mode="bilinear", align_corners=False)
                       for projection, source in zip(self.source_projections[index],
                                                     features[index + 1:])]
            semantic = sum(sources) / len(sources)
            enhanced.append(self.refine[index](
                self.gates[index](target, semantic) + semantic
            ))
        return enhanced + [features[-1]]


class IdentitySkips(nn.Module):
    """Pass native encoder skips unchanged when module B is absent."""

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        return list(features)


class ModuleAblationCSDSNet(CSDSNet):
    """Controlled A/B/C variants of the final semantic CSDS architecture.

    A is the directional, channel-gated and pixel-gated CSDS block; B is the
    top-down CSCA skip enhancement; C is the gated full-resolution RGB detail
    branch. All variants retain the same four-level topology and channel width.
    """

    def __init__(self, use_a: bool, use_b: bool, use_c: bool,
                 gate_floor: float = 0.25) -> None:
        block_type = DirectionalCSDSBlock if use_a else PlainStructureBlock
        super().__init__(channels=(48, 96, 192, 384), block_type=block_type)
        self.use_detail_branch = bool(use_c)
        self.csca = (TopDownCSCA(self.channels, gate_floor=gate_floor)
                     if use_b else IdentitySkips())
        if use_c:
            self.detail_gate = SemanticGate(self.channels[0], floor=gate_floor)
        else:
            self.input_detail = nn.Identity()
            self.full_fuse = ConvNormAct(self.channels[0], self.channels[0], 3)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        encoded = self.encoder(x)
        skips = self.csca(encoded)
        y = self.dec3(encoded[3], skips[2])
        y = self.dec2(y, skips[1])
        y = self.dec1(y, skips[0])
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_detail_branch:
            detail = self.detail_gate(self.input_detail(x), y)
            y = self.full_fuse(torch.cat([y, detail], dim=1))
        else:
            y = self.full_fuse(y)
        y = self.full_refine(y)
        return {"logits": self.head(y), "edge_logits": self.edge_head(y)}


class SemanticCSDSNet(CSDSNet):
    """Uniform three-dataset candidate: wide directional core, gated detail.

    Keep the encoder/decoder topology and CSDS interaction. Replace all-scale
    CSCA with top-down context and gate the RGB detail before full-resolution
    fusion. Original CSDS variants and their checkpoint keys remain unchanged.
    """

    def __init__(self, gate_floor: float = 0.25) -> None:
        super().__init__(channels=(48, 96, 192, 384), block_type=DirectionalCSDSBlock)
        self.csca = TopDownCSCA(self.channels, gate_floor=gate_floor)
        self.detail_gate = SemanticGate(self.channels[0], floor=gate_floor)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        encoded = self.encoder(x)
        skips = self.csca(encoded)
        y = self.dec3(encoded[3], skips[2])
        y = self.dec2(y, skips[1])
        y = self.dec1(y, skips[0])
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        detail = self.detail_gate(self.input_detail(x), y)
        y = self.full_refine(self.full_fuse(torch.cat([y, detail], dim=1)))
        return {"logits": self.head(y), "edge_logits": self.edge_head(y)}

MODEL_NAMES = ("csds_semantic", "csds_ablation_m0", "csds_ablation_m1",
               "csds_ablation_m2", "csds_ablation_m3", "csds_ablation_m4")


def build_model(name: str = "csds_semantic", pretrained: bool = False) -> nn.Module:
    """Build a research variant; random initialization only."""
    if pretrained:
        raise ValueError("This release does not use pretrained weights")
    if name == "csds_semantic":
        return SemanticCSDSNet()
    flags = {"csds_ablation_m0": (False, False, False),
             "csds_ablation_m1": (True, False, False),
             "csds_ablation_m2": (True, True, False),
             "csds_ablation_m3": (True, False, True),
             "csds_ablation_m4": (False, True, True)}
    if name not in flags:
        raise ValueError(f"Unknown model: {name}; choose from {MODEL_NAMES}")
    return ModuleAblationCSDSNet(*flags[name])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
