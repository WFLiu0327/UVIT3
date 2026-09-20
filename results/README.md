# UVIT3 pretrained checkpoint evaluation

These are **frozen validation-split results**, not independent test-set results. The same validation splits selected the best checkpoints.

- Datasets: BUSI (130 validation cases/seed), CVC-ClinicDB (123), GlaS (33).
- Seeds: 2981, 6142, 1187; all nine checkpoint split hashes matched the frozen JSON files.
- Inference: CPU FP32, 256×256 RGB input, batch size 1, threshold 0.5.
- Each seed metric is the mean over its validation cases. Values below are the mean ± sample standard deviation across three seeds.
- HD95 and ASSD are measured in pixels on the 256×256 evaluation masks.
- All nine evaluations had complete prediction and case files, with no undefined metric values.

| Dataset | Dice | IoU | Precision | Recall | Specificity | Accuracy | HD95 (px) | ASSD (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BUSI | 0.8103 ± 0.0283 | 0.7295 ± 0.0342 | 0.8315 ± 0.0268 | 0.8257 ± 0.0195 | 0.9808 ± 0.0048 | 0.9644 ± 0.0060 | 22.73 ± 2.15 | 8.97 ± 1.07 |
| CVC | 0.9281 ± 0.0074 | 0.8753 ± 0.0096 | 0.9337 ± 0.0164 | 0.9338 ± 0.0091 | 0.9936 ± 0.0023 | 0.9881 ± 0.0014 | 11.27 ± 2.48 | 3.04 ± 0.72 |
| GLAS | 0.9222 ± 0.0018 | 0.8624 ± 0.0025 | 0.9156 ± 0.0124 | 0.9364 ± 0.0160 | 0.9107 ± 0.0049 | 0.9274 ± 0.0084 | 16.52 ± 5.40 | 3.35 ± 0.89 |

Per-seed values and split hashes are in `pretrained_per_seed.csv`. Exact unrounded aggregate values are in `pretrained_summary.csv`.
