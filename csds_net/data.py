from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
Sample = Tuple[Path, Path]


def _image_files(root: Path, recursive: bool) -> List[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _pair_directories(image_root: Path, mask_root: Path, recursive: bool) -> List[Sample]:
    def key(path: Path, base: Path) -> str:
        return str(path.relative_to(base).with_suffix("")).replace("\\", "/").lower()

    masks = {key(path, mask_root): path for path in _image_files(mask_root, recursive)}
    samples = []
    for image in _image_files(image_root, recursive):
        mask = masks.get(key(image, image_root))
        if mask is not None:
            samples.append((image, mask))
    return samples


def discover_samples(root: Union[str, Path]) -> List[Sample]:
    """Discover paired images and masks in the three project datasets."""
    root = Path(root)
    candidates = [
        (root / "CrackTree260", root / "gt", False),
        (root / "image", root / "label", False),
        (root / "images", root / "masks", False),
        (root / "leftImg8bit" / "train", root / "gtFine" / "train", True),
    ]
    for image_root, mask_root, recursive in candidates:
        if image_root.is_dir() and mask_root.is_dir():
            samples = _pair_directories(image_root, mask_root, recursive)
            if samples:
                return samples
    raise FileNotFoundError(f"Could not discover paired images and masks below {root}")


def partition_samples(
    samples: List[Sample],
    root: Union[str, Path],
    split: str,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
) -> List[Sample]:
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    if split == "all":
        return list(samples)
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to at most one")
    no_test = abs(train_ratio + val_ratio - 1.0) < 1e-8

    root = Path(root)
    strata: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        image = sample[0]
        relative_parent = image.parent.relative_to(root)
        stratum = image.parent.name if len(relative_parent.parts) > 1 else "all"
        strata[stratum].append(sample)

    rng = np.random.default_rng(split_seed)
    partitions: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for stratum in sorted(strata):
        group = sorted(strata[stratum], key=lambda pair: str(pair[0]))
        order = rng.permutation(len(group))
        shuffled = [group[int(index)] for index in order]
        train_count = int(round(len(group) * train_ratio))
        val_count = int(round(len(group) * val_ratio))
        max_train = len(group) - 1 if no_test else len(group) - 2
        train_count = min(max(train_count, 1), max_train)
        max_val = len(group) - train_count if no_test else len(group) - train_count - 1
        val_count = min(max(val_count, 1), max_val)
        partitions["train"].extend(shuffled[:train_count])
        partitions["val"].extend(shuffled[train_count : train_count + val_count])
        partitions["test"].extend(shuffled[train_count + val_count :])
    return sorted(partitions[split], key=lambda pair: str(pair[0]))


class CrackDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        image_dir: str = "images",
        mask_dir: str = "masks",
        image_size: Union[int, Tuple[int, int]] = 512,
        train: bool = False,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        strong_augment: bool = False,
    ) -> None:
        del image_dir, mask_dir
        self.root = Path(root)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)
        self.train = train
        self.strong_augment = strong_augment
        all_samples = discover_samples(self.root)
        self.samples = partition_samples(
            all_samples,
            self.root,
            split,
            split_seed,
            train_ratio,
            val_ratio,
        )
        if not self.samples:
            raise RuntimeError(f"No samples in the {split} split below {self.root}")

    @property
    def sample_ids(self) -> List[str]:
        return [str(image.relative_to(self.root)).replace("\\", "/") for image, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not self.train:
            return image, mask
        if torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if torch.rand(()) < 0.2:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.strong_augment:
            if torch.rand(()) < 0.6:
                scale = float(0.75 + 0.25 * torch.rand(()))
                crop_width = max(1, round(image.width * scale))
                crop_height = max(1, round(image.height * scale))
                left = int(torch.randint(image.width - crop_width + 1, (1,)))
                top = int(torch.randint(image.height - crop_height + 1, (1,)))
                bounds = (left, top, left + crop_width, top + crop_height)
                image, mask = image.crop(bounds), mask.crop(bounds)
            if torch.rand(()) < 0.7:
                image = ImageEnhance.Brightness(image).enhance(float(0.8 + 0.4 * torch.rand(())))
            if torch.rand(()) < 0.7:
                image = ImageEnhance.Contrast(image).enhance(float(0.8 + 0.4 * torch.rand(())))
            if torch.rand(()) < 0.4:
                image = ImageEnhance.Color(image).enhance(float(0.85 + 0.3 * torch.rand(())))
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = self._augment(image, mask)
        image = image.resize(self.image_size[::-1], Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size[::-1], Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        # Public crack datasets encode foreground as bool, 0/1, or 0/255.
        mask_array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        if self.train and self.strong_augment and torch.rand(()) < 0.25:
            image_tensor = (image_tensor + 0.015 * torch.randn_like(image_tensor)).clamp(0.0, 1.0)
        mask_tensor = torch.from_numpy(mask_array[None])
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": str(image_path.relative_to(self.root)).replace("\\", "/"),
        }
