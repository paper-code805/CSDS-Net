"""Rebuild unadjusted mean, sample SD and sample variance from per-seed results."""
from __future__ import annotations
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(source, output):
    groups = defaultdict(list)
    with Path(source).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[(row["dataset"], row["model"])].append(row)
    metrics = ["strict_dice", "strict_iou", "precision", "recall", "dice", "iou_derived", "pa"]
    rows = []
    for (dataset, model), values in sorted(groups.items()):
        seeds = [int(value["seed"]) for value in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds for {dataset}/{model}")
        row = {"dataset":dataset, "model":model, "n":len(values), "seeds":";".join(map(str, sorted(seeds)))}
        for metric in metrics:
            if metric not in values[0]:
                continue
            observations = [float(value[metric]) for value in values]
            row[metric+"_mean"] = statistics.mean(observations)
            row[metric+"_sd"] = statistics.stdev(observations) if len(values)>1 else ""
            row[metric+"_variance"] = statistics.variance(observations) if len(values)>1 else ""
        rows.append(row)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/per_seed.csv")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()
    for row in summarize(args.input, args.output):
        print(row["dataset"], row["model"], row["n"], row["dice_mean"])
