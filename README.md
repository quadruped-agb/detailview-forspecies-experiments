# DetailView Inference — For-Species-20K Dev Sample

## Overview
Running DetailView species classification model on a sample of 99 trees 
from the For-Species-20K dataset.

## Hardware
- HP EliteBook x360 1030 G3
- 16GB RAM, no GPU (CPU only)

## Dataset
- Source: For-Species-20K dev sample
- Total trees sampled: 99
- Trees successfully run: 94
- Trees excluded: 5 (large .laz files >1MB caused OOM kills)

## Inference Settings
- Model: model_202305171452_60 (original DetailView pretrained model)
- n_aug: 1 (reduced from default 10 due to RAM constraints)
- projection_backend: numpy (switched from torch to reduce memory)
- Batches of 5 trees with 60s cooldown between runs

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 68.75% |
| Weighted F1 | 0.72 |
| Trees matched with ground truth | 48 |

## Notes
- Accuracy is lower than expected (~79.5%) due to n_aug=1 instead of default 10
- A GPU would allow n_aug=3 and full dataset inference without memory issues

## Original Model
DetailView by Julian Frey: https://github.com/JulFrey/DetailView
