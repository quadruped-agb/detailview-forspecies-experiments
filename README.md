# DetailView Inference — For-Species-20K Dev Sample

## Overview
Running DetailView species classification model on a sample of 99 trees from the For-Species-20K dataset.

## Dataset
- Source: For-Species-20K dev sample
- Total trees sampled: 99
- Trees successfully run: 94
- Trees excluded: 5 (large .laz files over 1MB caused memory issues during processing)

## Inference Settings
- Model: model_202305171452_60 (original DetailView pretrained model)
- n_aug: 1 (reduced from default 10 due to memory constraints)
- projection_backend: numpy (switched from torch to reduce memory usage)
- Processed in batches of 5 trees with a 60 second cooldown between batches

## Results

| Metric | Value |
|---|---|
| Accuracy | 78.72% |
| Weighted F1 | 0.78 |
| Macro F1 | 0.77 |
| Trees matched with ground truth | 94 |

## Notes
This was a baseline pipeline validation, not a replication of the paper's official benchmark, which used 50 augmentations per tree. With n_aug=1, results are reasonable but less robust than the paper's reported figures, since predictions rely on a single augmented view per tree rather than an averaged ensemble. A machine with more available memory or a GPU would allow higher n_aug values and full dataset inference without the file-size and batch-size workarounds used here.

## Original Model
DetailView by Julian Frey: https://github.com/JulFrey/DetailView
