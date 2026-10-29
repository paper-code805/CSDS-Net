# Training and evaluation protocol

The paper's Table 1 is reproduced in the README without replacing its settings
with a new experiment. The reference hardware is Windows 64-bit, Intel
Core i5-13600K (3.50 GHz), 32 GB RAM, and one NVIDIA RTX 3090 (24 GB).
[Paper reference](PAPER_REFERENCE.md) summarizes the manuscript. The operational
details below describe the included trainer and archived measurements; paper
table transcriptions are stored separately under `results/paper/`.

Each dataset has an independent model. Default recipes use seed 42, 100 epochs,
RGB 256x256, batch 8, AdamW learning rate/weight decay 1e-4, cosine annealing,
gradient norm clipping at 1.0, no pretraining, and CUDA AMP. CPU runs use FP32.
The fixed 80/20 split uses split_seed 42 with no independent test set. Use
training seeds 42/123/3407 for the archived three-run design.

| Dataset | Positive BCE weight | Hard-negative weight | Boundary weight | Augmentation |
|---|---:|---:|---:|---|
| CrackTree260 | 8 | 2 | 0.15 | Horizontal and vertical flips |
| CrackSC | 4 | 1 | 0.15 | Flips, crop, brightness/contrast/color, noise |
| UAV-Crack500 experiment | 4 | 1 | 0.15 | Same strong augmentation as CrackSC |

Loss = weighted BCE + soft Dice + 0.15 x boundary BCE. A hard negative is a
background pixel with detached current crack probability >0.2. It receives
BCE multiplier `1 + hard_negative_weight`; positive pixels use `pos_weight`.
The boundary target is a 3x3 dilation minus erosion of the unmodified binary
mask. Soft Dice uses sigmoid logits and per-image smoothing 1. This is a
different loss/augmentation recipe from the seven plain BCE+Dice baselines;
comparison does not isolate architecture alone.

`best.pt` maximizes **strict-pixel validation Dice**. `last.pt` is for resume.
Both train and validation are recorded every epoch; train metrics use that
epoch's augmented minibatches and changing weights, rather than a second
evaluation pass on the final epoch state. LR is the rate used for that epoch;
elapsed_seconds covers its train+validation computation before checkpoint I/O.
Static GFLOPs/Params/FPS are measured once per run and repeated in history.

Validation loss is calculated in FP32; non-finite AMP validation outputs are
retried in FP32 and logged. The Tree seed42 archive used the earlier validation
path, whereas CrackSC/UAV used the finite-validation repair. No failed history
was backfilled. Public defaults apply finite validation to new runs.

## Metrics

Prediction threshold is fixed at 0.5; no threshold search, TTA, postprocessing,
or mixing metrics from different checkpoints. Counts accumulate over the entire
validation partition before metrics are computed.

Strict metrics use TP/FP/FN pixel overlap: Dice=2TP/(2TP+FP+FN),
IoU=TP/(TP+FP+FN). PA always uses exact pixel agreement.

For tolerance metrics, N2 is a 5x5 square dilation, allowing a maximum
Chebyshev displacement of 2 pixels on the resized grid:

```
Pr2 = |prediction intersect N2(truth)| / |prediction|
Re2 = |truth intersect N2(prediction)| / |truth|
Dice2 = 2 Pr2 Re2 / (Pr2 + Re2)
iou_derived = Dice2 / (2 - Dice2)
```

`iou_derived` is a monotonic transform of tolerant Dice; it is **not strict
pixel IoU**. Output keeps both fields distinct. When both foreground sets are
empty, tolerance Pr/Re/Dice are 1; when exactly one is empty they are 0.

Across seeds, calculate each metric for each run first, then take its arithmetic
mean, sample SD (ddof=1), and sample variance. Do not derive an IoU mean from a
rounded Dice mean or invent uncertainty for a single observation.

## Raw evidence

`results/experiments/` preserves 54 unmodified 100-row strict and tolerance
histories for six configurations, three datasets and three seeds. `per_seed.csv`
and `summary.csv` are rebuilt from their strict-best epochs. `seed42/` provides
the three selected final-model histories and portable recipes for convenience.
`history_audit.json` records history hashes, epochs, split equality and finiteness.
`baseline_per_seed.csv` holds the 63 archived unmodified baseline evaluations;
the baseline code/dependencies/checkpoints are not bundled in this model release.

Archived performance is evidence from existing runs, not a guarantee that this
new portable trainer will produce bitwise-identical training on another system.
`verification.json` records the actual package checks. No full training was
performed during packaging.
