# Archived, unadjusted validation comparison

All rows use three actual training seeds; ± is sample SD. IoU-derived is the
per-seed transform of tolerant Dice and is not strict pixel IoU. Semantic
variant loss/augmentation differs from the seven baseline recipes. No paper
scenario edits or borrowed uncertainty are used.

## CrackTree260

| Method | PA | Pr (2px) | Re (2px) | Dice (2px) | IoU-derived |
|---|---:|---:|---:|---:|---:|
| SCSegamba (PyTorch approximation) | 0.9917 ± 0.0003 | 0.8630 ± 0.0163 | 0.3802 ± 0.0793 | 0.5252 ± 0.0800 | 0.3587 ± 0.0734 |
| SegFormer-B3 | 0.9914 ± 0.0001 | 0.8810 ± 0.0034 | 0.6041 ± 0.0046 | 0.7168 ± 0.0033 | 0.5586 ± 0.0040 |
| TransFuse-S | 0.9902 ± 0.0004 | 0.8872 ± 0.0036 | 0.5701 ± 0.0173 | 0.6940 ± 0.0117 | 0.5315 ± 0.0138 |
| TransUNet | 0.9928 ± 0.0002 | 0.8677 ± 0.0081 | 0.7688 ± 0.0044 | 0.8153 ± 0.0050 | 0.6882 ± 0.0072 |
| Swin-UMamba (PyTorch approximation) | 0.9940 ± 0.0002 | 0.8721 ± 0.0030 | 0.8753 ± 0.0139 | 0.8736 ± 0.0056 | 0.7757 ± 0.0088 |
| CT-CrackSeg | 0.9919 ± 0.0004 | 0.8388 ± 0.0179 | 0.8119 ± 0.0306 | 0.8249 ± 0.0192 | 0.7023 ± 0.0280 |
| CrackFormer-II | 0.9945 ± 0.0000 | 0.8816 ± 0.0013 | 0.8131 ± 0.0168 | 0.8459 ± 0.0092 | 0.7330 ± 0.0139 |
| CSDS-Net semantic variant | 0.9932 ± 0.0000 | 0.8539 ± 0.0108 | 0.8781 ± 0.0177 | 0.8658 ± 0.0139 | 0.7635 ± 0.0214 |

## CrackSC

| Method | PA | Pr (2px) | Re (2px) | Dice (2px) | IoU-derived |
|---|---:|---:|---:|---:|---:|
| SCSegamba (PyTorch approximation) | 0.9616 ± 0.0206 | 0.6524 ± 0.3322 | 0.6033 ± 0.0918 | 0.6071 ± 0.2232 | 0.4589 ± 0.2158 |
| SegFormer-B3 | 0.9765 ± 0.0010 | 0.8743 ± 0.0109 | 0.7378 ± 0.0182 | 0.8001 ± 0.0062 | 0.6668 ± 0.0086 |
| TransFuse-S | 0.9767 ± 0.0005 | 0.8918 ± 0.0081 | 0.6967 ± 0.0185 | 0.7821 ± 0.0085 | 0.6422 ± 0.0115 |
| TransUNet | 0.9754 ± 0.0008 | 0.8089 ± 0.0036 | 0.7335 ± 0.0130 | 0.7693 ± 0.0069 | 0.6251 ± 0.0091 |
| Swin-UMamba (PyTorch approximation) | 0.9794 ± 0.0003 | 0.8719 ± 0.0024 | 0.7939 ± 0.0045 | 0.8311 ± 0.0033 | 0.7110 ± 0.0048 |
| CT-CrackSeg | 0.9755 ± 0.0013 | 0.8657 ± 0.0179 | 0.8640 ± 0.0158 | 0.8646 ± 0.0051 | 0.7616 ± 0.0079 |
| CrackFormer-II | 0.9773 ± 0.0001 | 0.8531 ± 0.0080 | 0.7478 ± 0.0238 | 0.7969 ± 0.0168 | 0.6626 ± 0.0233 |
| CSDS-Net semantic variant | 0.9742 ± 0.0005 | 0.8409 ± 0.0049 | 0.8449 ± 0.0026 | 0.8429 ± 0.0022 | 0.7284 ± 0.0032 |

## UAV-Crack500

| Method | PA | Pr (2px) | Re (2px) | Dice (2px) | IoU-derived |
|---|---:|---:|---:|---:|---:|
| SCSegamba (PyTorch approximation) | 0.9621 ± 0.0013 | 0.9295 ± 0.0061 | 0.8173 ± 0.0202 | 0.8698 ± 0.0121 | 0.7697 ± 0.0189 |
| SegFormer-B3 | 0.9604 ± 0.0004 | 0.8949 ± 0.0067 | 0.8649 ± 0.0100 | 0.8796 ± 0.0019 | 0.7851 ± 0.0031 |
| TransFuse-S | 0.9597 ± 0.0009 | 0.9383 ± 0.0165 | 0.8055 ± 0.0291 | 0.8665 ± 0.0096 | 0.7645 ± 0.0150 |
| TransUNet | 0.9651 ± 0.0006 | 0.8963 ± 0.0109 | 0.8581 ± 0.0141 | 0.8767 ± 0.0026 | 0.7804 ± 0.0041 |
| Swin-UMamba (PyTorch approximation) | 0.9662 ± 0.0003 | 0.8947 ± 0.0073 | 0.8705 ± 0.0077 | 0.8824 ± 0.0005 | 0.7896 ± 0.0008 |
| CT-CrackSeg | 0.9693 ± 0.0007 | 0.9193 ± 0.0039 | 0.8908 ± 0.0008 | 0.9048 ± 0.0016 | 0.8262 ± 0.0026 |
| CrackFormer-II | 0.9644 ± 0.0002 | 0.9039 ± 0.0049 | 0.8701 ± 0.0049 | 0.8866 ± 0.0014 | 0.7964 ± 0.0023 |
| CSDS-Net semantic variant | 0.9679 ± 0.0001 | 0.9124 ± 0.0025 | 0.8973 ± 0.0036 | 0.9048 ± 0.0013 | 0.8261 ± 0.0022 |

Source tables: `baseline_per_seed.csv` / `per_seed.csv`; regenerate summaries with `tools/summarize.py`. Pure-PyTorch selective-scan approximations are not official CUDA implementations. Baseline code is not bundled.
