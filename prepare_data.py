"""Validate a dataset and create or check its fixed 80/20 split manifest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from csds_net.data import CrackDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference", help="Check against a supplied benchmark split")
    parser.add_argument("--output", help="Write a new manifest; fails if it already exists")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    sets = [CrackDataset(args.data_root, split, image_size=256, split_seed=args.split_seed,
                         train_ratio=0.8, val_ratio=0.2) for split in ("train", "val")]
    manifest = {"split_seed": args.split_seed, "train": sets[0].sample_ids,
                "val": sets[1].sample_ids, "test": []}
    if set(manifest["train"]) & set(manifest["val"]):
        raise ValueError("Training and validation partitions overlap")
    if args.reference and json.loads(Path(args.reference).read_text(encoding="utf-8")) != manifest:
        raise ValueError("Dataset differs from the reference manifest; do not silently substitute data")
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    print(json.dumps({key:len(manifest[key]) for key in ("train", "val", "test")}, indent=2))
