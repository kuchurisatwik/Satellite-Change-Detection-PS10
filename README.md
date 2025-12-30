# 🛰️ Satellite Change Detection using Deep Learning (PS10)

A clean, modular implementation for pixel-level change detection between pairs of high‑resolution satellite images. The project demonstrates a U-Net–based semantic segmentation pipeline to generate binary change masks highlighting changed regions between a "before" and "after" image.

This repository is intended for learning, research, and portfolio use.

---

## Table of contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Model & Approach](#model--approach)
- [Preprocessing](#preprocessing)
- [Inference pipeline](#inference-pipeline)
- [Post-processing](#post-processing)
- [Outputs](#outputs)
- [Quickstart](#quickstart)
- [Experiments & Notebooks](#experiments--notebooks)
- [Future work](#future-work)
- [License & Contact](#license--contact)

---

## Overview

Change detection from satellite imagery is used for monitoring urban growth, infrastructure changes, environmental impacts, and disaster assessment. This project automates change detection via a deep learning semantic segmentation approach to produce binary maps that indicate where change has occurred at the pixel level.

---

## Highlights

- U-Net based segmentation for pixel-accurate change detection
- Patch-based preprocessing to support high-resolution images
- End-to-end inference pipeline (patch extraction → prediction → reconstruction → cleanup)
- Post-processing to reduce noise and false positives
- Modular code structure suitable for experimentation and extension

---

## Repository structure

satellite-change-detection-ps10/
│
├── data/
│   └── LEVIR-CD/                  # Dataset directory (before, after, labels) — not included
│
├── models/
│   └── unet_model.py              # U-Net model architecture
│
├── preprocessing/
│   └── patch_extraction.py        # Patch generation and utilities
│
├── inference/
│   ├── detect_changes.py          # Main inference pipeline (load model, run prediction)
│   └── post_processing.py         # Morphological filtering & cleanup utilities
│
├── notebooks/
│   └── experiments.ipynb          # Experiments, visualization, and analysis
│
├── output/
│   └── predicted_masks/           # Generated change detection outputs
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore



---

## Dataset

- Dataset used: LEVIR-CD (high-resolution building change detection dataset)
- Task: Binary change detection (change / no-change)
- Note: Dataset files are not included due to size. After downloading, place them at:

```
data/LEVIR-CD/
```

Expect the folder to contain paired "before" and "after" images and ground-truth change masks (for evaluation/training).

---

## Model & Approach

- Base architecture: U-Net (encoder-decoder segmentation network with skip-connections)
- Input: A pair of images (Before, After). Common approaches:
  - Stack channels (e.g., concatenate before and after images) as input to U-Net
  - Use Siamese or difference-based input variants
- Output: Binary mask where white (1) indicates detected change and black (0) indicates no change

Why U-Net?
- Effective for dense segmentation tasks
- Preserves spatial detail via skip connections

---

## Preprocessing

Key steps:
- Patch extraction: Large images are split into 256 × 256 overlapping/non-overlapping patches to reduce memory footprint and improve batch processing.
- Normalization: Per-channel normalization / scaling to improve model stability.
- (Optional) Data augmentation: flips, rotations, color jitter for robustness.

See: preprocessing/patch_extraction.py

---

## Inference pipeline

Main steps performed by inference/detect_changes.py:

1. Load trained U-Net weights
2. Extract patches from input before/after images
3. Run model prediction on patches (batch inference)
4. Reconstruct full-size prediction from patches (stitching/averaging overlaps)
5. Apply post-processing to clean the mask
6. Save final binary mask to output/predicted_masks/

Usage (Quickstart below) shows how to run the inference script.

---

## Post-processing

Common techniques applied after prediction to improve map quality:

- Morphological opening and closing to remove small artifacts and fill holes
- Small object removal (connected-component analysis) to filter out tiny false positives
- Optional median/gaussian filtering for smoothing

Refer to inference/post_processing.py for utilities and configurable parameters.

---

## Outputs

- Binary PNG/TIFF masks saved under output/predicted_masks/
  - White (255) = change
  - Black (0) = no change
- Outputs can be:
  - Visualized directly
  - Converted to vector formats (e.g., shapefiles) for GIS workflows
  - Used for evaluation against ground truth (IoU, F1-score, precision/recall)

---

## Quickstart

1. Create environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate     # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Prepare the dataset

Download the LEVIR-CD dataset (or your chosen dataset) and place it here:

```
data/LEVIR-CD/
```

3. Run change detection (inference)

```bash
python inference/detect_changes.py --model-path path/to/model.pth \
    --input-before data/LEVIR-CD/before/some_image.png \
    --input-after data/LEVIR-CD/after/some_image.png \
    --output-dir output/predicted_masks/
```

Notes:
- The CLI arguments in the example are illustrative; check `inference/detect_changes.py --help` for exact options.
- If you want to train a model, add training scripts that load data patches and run optimization (not included here).

---

## Experiments & Notebooks

- The `notebooks/experiments.ipynb` contains exploratory analysis, visualization of predictions, and evaluation routines.
- Use the notebook to inspect intermediate results (patches, predictions, post-processing effects).

---

## Future improvements

Potential extensions and research directions:

- Add training scripts with checkpointing and logging
- Multi-class change detection (different change types)
- Improve architecture (Siamese encoders, attention modules)
- Convert outputs to vector formats (GeoJSON / Shapefiles)
- Cloud deployment or an API for large-scale inference
- Add unit tests and CI/CD pipeline

---

## Contributing

Contributions, issues, and feature requests are welcome. For major changes, please open an issue first to discuss the plan. Pull requests should include tests or reproduceable examples where applicable.

---

## License & Contact

This repository is for educational and research purposes. Include your chosen license file (e.g., MIT, Apache-2.0) in the root if you want to make reuse terms explicit.

Author: kuchurisatwik  
Repository: https://github.com/kuchurisatwik/Satellite-Change-Detection-PS10

If you have questions or suggestions, open an issue in the repo.

