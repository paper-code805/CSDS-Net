# Dataset preparation

The README's overview follows the manuscript counts: CrackTree260 260 images,
CrackSC 197 images, and UAV-Crack500 500 patches. The table and procedures
below describe the data actually used by the included archived manifests.
See [Paper reference](PAPER_REFERENCE.md) for the manuscript description.

Download imagery and annotations from the original providers; data are not
included here. Preserve the relative filenames in `splits/*.json` to reproduce
the archived splits. Conversion to another layout changes sample IDs.

| Dataset | Original source | Archived discovered pairs | Train | Validation |
|---|---|---:|---:|---:|
| CrackTree260 | [DeepCrack download](https://github.com/qinnzou/DeepCrack#download) | 260 | 208 | 52 |
| CrackSC | [Transformer-Crack](https://github.com/jonguo111/Transformer-Crack) | 197 | 158 | 39 |
| UAV experiment | [UAV-CrackX source study](https://github.com/SHAN-JH/UAV-CrackX-Datasets) | 1200 | 960 | 240 |

CrackSC's publicly provided files may be a partial release. Exact availability
of the archived 197 pairs has not been independently established. The local
experiment is named `UAV-Crack500`, but its actual manifest contains **1200**
pairs from `leftImg8bit/train`, stratified by subfolder. The manuscript describes
500 patches. The linked UAV-CrackX collection is the source-study repository;
equivalence to a specific 500-patch distribution is not established. Neither a
500-image subset nor provider-defined validation files are silently substituted.
Resolve these data-version differences before claiming paper reproduction.

Recognized layouts, in order of precedence:

```text
datasets/
  CrackTree260/
    CrackTree260/      # images
    gt/               # masks with identical stems
  CrackSC/
    image/
    label/
  UAV-Crack500/
    leftImg8bit/train/<subset>/*.{jpg,png}
    gtFine/train/<subset>/*.{png,jpg}
```

A generic `images/` and `masks/` layout is also accepted for custom datasets.
Images and masks pair by the same relative stem (extension may differ). The
original discovery rule only includes UAV `train` folders; provider `val`
folders are not part of these experiments. The loaded pool is split 80/20 with
NumPy default_rng seed 42, stratified by nested image folder. There is no
independent test partition. Filenames, not data files, are shipped in manifests.

```bash
python prepare_data.py --data-root datasets/CrackTree260 --reference splits/CrackTree260.json
python prepare_data.py --data-root datasets/CrackSC --reference splits/CrackSC.json
python prepare_data.py --data-root datasets/UAV-Crack500 --reference splits/UAV-Crack500.json
```

The commands fail on a split mismatch. For a new dataset, create a separately
named manifest and pass it explicitly; results then describe a new experiment:

```bash
python prepare_data.py --data-root datasets/MyCracks --output splits/MyCracks.json
python train.py --config configs/CrackTree260.json --data-root datasets/MyCracks --reference-manifest splits/MyCracks.json
```

Images are RGB, bilinearly resized to 256x256 and scaled to [0,1]. Masks are
grayscale, resized with nearest-neighbor sampling, and binarized by `mask > 0`.
Class-index annotations with ignore/other nonzero labels require an explicit
conversion to binary masks before using this loader. The preprocessing is
unchanged from the archived experiments.
