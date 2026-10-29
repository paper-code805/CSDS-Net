# Manuscript reference

The supplied final manuscript, **CSDS-Net: A Structure-Guided Dual-Sparse Network
for Road Crack Segmentation**, is the source for the README's method overview,
dataset descriptions, experiment figures and reported tables. Numbering follows
the paginated PDF, rather than the names of internal LaTeX labels. It is a
Sensors submission; this package does not assert journal acceptance or a DOI.

## Method described in the paper

CSDS-Net uses a four-level encoder–decoder. A stride-two 3x3 stem produces
64-channel features. The four encoder widths are 64, 128, 256 and 512, with
two independently parameterized CSDS Blocks per level. Each block preserves
its input dimensions.

The CSDS Block constructs a shared structural representation from adjacent-scale
Gaussian differences and pooled channel interactions. The representation produces
a crack-response map, five directional weights and a structural residual.
Response-weighted regional descriptors guide regional Top-K selection; the
candidate pixels combine selected regions with retained 3x3 neighbors. A
mixture of one isotropic and four oriented Gaussian priors guides pixel Top-K
selection. Selected values are aggregated with normalized attention weights,
followed by residual and feed-forward updates.

CSCA aligns the four encoder features to an H/4 x W/4, 64-channel grid. Content
and structural responses determine scale weights. Residual aggregation forms
shared context, which is adapted to the three decoder skip levels. Decoder
fusion is additive. Upsampling and a 3x3 segmentation head restore the input
resolution, with a 0.5 output threshold.

These paragraphs describe manuscript Section 2. The packaged trained model
uses the semantic research variant; its actual operators and checkpoint
compatibility are documented in [Implementation](IMPLEMENTATION.md). A
documentation revision does not make those operators equivalent to the paper.

## Datasets and protocol

| Dataset | Count stated in the manuscript | Description |
|---|---:|---|
| CrackTree260 | 260 | Ground-level road images, reference resolution 800 x 600 |
| CrackSC | 197 | Close-range pavement images |
| UAV-Crack500 | 500 | UAV image patches |

The paper trains each dataset separately with RGB 256 x 256 inputs, a fixed
80/20 training/validation partition (partition seed 42), and no independent
test partition. Table 1 lists seeds 42, 123 and 3407; 100 epochs from random
initialization; AdamW learning rate and weight decay 1e-4; batch size 8;
cosine scheduling; and AMP. The final balanced objective and augmentation
settings are held fixed within each dataset's structural ablations.

The checkpoint maximizes strict-pixel validation Dice. Tolerance metrics use
the same selected epoch and threshold 0.5. The paper defines bidirectional
two-pixel Precision and Recall and their harmonic-mean Dice; the difficult
examples use strict-pixel Dice. The archived evaluator also distinguishes
strict IoU from a tolerance-Dice-derived quantity. The README retains the
table's printed `IoU` label rather than changing its source values. See
[Protocol](PROTOCOL.md) for the actual output definitions.

The included UAV experiment manifest contains 1,200 pairs, rather than the
paper's stated 500 patches. [Dataset preparation](DATASETS.md) preserves the
actual 960/240 archived split; it must not be represented as a verified
500-patch reproduction.

## Original figures and tables

| Final PDF item | Content | Packaged source |
|---|---|---|
| Figure 1 | Network architecture | `figs/CSDS_Net_Architecture.png` |
| Figure 2 | CSDS Block | `figs/CSDS_Block.png` |
| Figure 3 | Representative qualitative comparison | `figs/paper/qualitative_comparison.png` |
| Figure 4 | Difficult-sample comparison | `figs/paper/difficult_comparison.png` |
| Figure 5 | Module-effect visualizations | `figs/paper/module_effect_heatmaps.png` |
| Figure 6 | Training dynamics | `figs/paper/training_dynamics_2x3.png` |
| Figure 7 | Paired module contributions over training | `figs/paper/module_contribution_trajectories.png` |
| Table 1 | Training/evaluation settings | Verbatim cells from the manuscript text |
| Table 2 | Eight-model, three-dataset comparison | `results/paper/main_comparison.csv` |
| Table 3 | M0–M5 configuration ablations | `results/paper/ablation_results.csv` |
| Table 4 | Paired module contributions | Verbatim cells from the manuscript text |
| Table 5 | Selected single-image module effects | Verbatim cells from the manuscript text |
| Table 6 | GFLOPs, parameters and FPS | `results/paper/efficiency_comparison.csv` |

Images and the three CSVs are copied byte-for-byte from the author-supplied
LaTeX archive. Figures were not replotted, relabeled, cropped or recolored.
Markdown changes only the table layout, with source numerical precision and
best-value emphasis retained. [Asset provenance](../results/paper_asset_provenance.json)
records archive entries and SHA-256 hashes. Unused historical figures in the
archive are not substituted for active paper figures.

## Result provenance

The README is a transcription of the manuscript's reported results. It is not
an independent validation of every statistic. In particular:

- Table 2 includes author-adjusted comparator entries: Tree/Swin-UMamba Dice
  0.8463, CrackSC/CT-CrackSeg 0.8246, and UAV/CT-CrackSeg 0.8918. These differ
  from the archived observed means and must not be cited as newly measured
  baseline results from this code package.
- The CSDS-Net Table 2 central values follow the selected seed-42 runs. Its
  reported uncertainty terms were supplied during manuscript preparation and
  are not observed three-seed CSDS-Net standard deviations. Copying those
  terms from the manuscript does not make them experimentally verified.
- The manuscript explicitly uses an adjusted CrackSC/M0 Dice of
  0.8423 ± 0.0024. Its learning-curve visualization applies a constant Dice
  offset of -0.0106731374, retains Recall and recalculates Precision. The
  raw archived mean is approximately 0.852973. The README embeds the original
  paper curve, including its source processing, without applying that offset
  to any archived history.
- The model comparison and ablation tables come from separate experimental
  batches. Their CSDS-Net rows are not interchangeable summaries.
- The Table 6 timing values are retained at manuscript precision. The paper
  does not fully specify each comparator's timing scope. The packaged profile
  command describes its own model-only timing settings explicitly.
- Archived SCSegamba and Swin-UMamba use pure-PyTorch selective-scan
  approximations, rather than the official CUDA selective-scan implementations.

The unchanged manuscript source note is included as
[source provenance](../results/paper/source_data_provenance.json). Its statements
about unavailable raw logs concern the manuscript-preparation source bundle;
this code package separately includes the subsequently audited local histories.

For observed metrics, use [per-seed records](../results/per_seed.csv),
[three-seed mean/SD/variance](../results/summary.csv), the
[unadjusted baseline comparison](../results/COMPARISON.md), and
[verification](../results/verification.json). Those files were not changed to
match manuscript values. No training, replacement experiment or new uncertainty
estimate was produced for this documentation revision.
