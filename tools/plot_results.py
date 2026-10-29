"""Plot unadjusted 3-seed ablation Dice curves from shipped epoch histories."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="figs/ablation_curves.png")
    args = parser.parse_args()
    audits = json.loads((ROOT/"results/history_audit.json").read_text())["experiments"]
    datasets = ("CrackTree260", "CrackSC", "UAV-Crack500")
    models = ("csds_ablation_m0", "csds_ablation_m1", "csds_ablation_m2", "csds_ablation_m3", "csds_ablation_m4", "csds_semantic")
    labels = ("M0", "M1", "M2 (w/o C)", "M3 (w/o B)", "M4 (w/o A)", "M5 (full)")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for axis, dataset in zip(axes, datasets):
        for index, (model, label) in enumerate(zip(models, labels)):
            runs = []
            for audit in audits:
                if audit["dataset"] == dataset and audit["model"] == model:
                    path = ROOT / "results/experiments" / dataset / model / f"seed_{audit['seed']}" / "tolerance2_history.csv"
                    with path.open(newline="",encoding="utf-8") as handle:
                        runs.append([float(row["val_dice"]) for row in csv.DictReader(handle)])
            array = np.array(runs)
            assert array.shape == (3,100)
            mean, sd = array.mean(0), array.std(0,ddof=1)
            epochs = np.arange(1,101)
            line, = axis.plot(epochs,mean,label=label,lw=1.8 if index == 5 else 1.1)
            axis.fill_between(epochs,mean-sd,mean+sd,color=line.get_color(),alpha=0.10)
        axis.set_title(dataset,fontsize=11,fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_xlim(1,100)
        axis.set_ylim(0.6,0.96)
        axis.grid(alpha=0.2)
        axis.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Validation Dice (2-pixel tolerance)")
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=6,frameon=False,bbox_to_anchor=(0.5,0.015),fontsize=9)
    fig.suptitle("Unadjusted epoch histories: mean and sample SD across seeds 42, 123, 3407",fontsize=12)
    fig.subplots_adjust(left=0.065,right=0.99,top=0.82,bottom=0.24,wspace=0.12)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor="white")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
