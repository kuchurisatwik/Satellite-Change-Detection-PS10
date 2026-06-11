# Geospatial Change Detection Engine

## Overview

A deep learning-based geospatial analytics pipeline for automated change detection between high-resolution satellite image pairs.

This project leverages a U-Net semantic segmentation architecture to identify and localize changes at the pixel level between temporal satellite observations. The system processes "before" and "after" imagery, generates binary change masks, and applies post-processing techniques to improve prediction quality and reduce noise.

The solution is designed for remote sensing research, environmental monitoring, urban development analysis, infrastructure assessment, and disaster impact evaluation.

---

## Key Features

### Deep Learning-Based Change Detection

* U-Net encoder-decoder architecture for semantic segmentation
* Pixel-level binary classification of changed versus unchanged regions
* Support for high-resolution satellite imagery

### Scalable Image Processing

* Patch-based processing for memory-efficient inference
* Reconstruction of full-resolution predictions from image patches
* Batch inference support

### Automated Post-Processing

* Morphological filtering
* Connected component analysis
* Noise reduction and false-positive suppression

### Research-Friendly Design

* Modular codebase
* Easy experimentation and model replacement
* Notebook-based visualization and analysis workflow

---

## Architecture

```text
Satellite Image Pair
(Before + After)
          │
          ▼
   Patch Extraction
          │
          ▼
      U-Net Model
          │
          ▼
  Binary Predictions
          │
          ▼
 Patch Reconstruction
          │
          ▼
   Post-Processing
          │
          ▼
 Final Change Mask
```

---

## Use Cases

* Urban expansion monitoring
* Building and infrastructure tracking
* Disaster damage assessment
* Land-use and land-cover analysis
* Environmental monitoring
* Agricultural change detection

---

## Dataset

The project uses the LEVIR-CD dataset, a benchmark dataset for high-resolution building change detection.

Dataset structure:

```text
data/
└── LEVIR-CD/
    ├── before/
    ├── after/
    └── labels/
```

Dataset files are not included in the repository due to size constraints.

---

## Model Architecture

### U-Net Segmentation Network

The model follows an encoder-decoder architecture with skip connections that preserve spatial information during downsampling and upsampling operations.

Input:

* Before image
* After image

Output:

* Binary change mask

Benefits:

* Accurate localization
* Efficient training
* Strong performance on dense prediction tasks

---

## Inference Pipeline

The inference workflow consists of:

1. Loading trained model weights
2. Extracting image patches
3. Running batch predictions
4. Reconstructing full-resolution masks
5. Applying post-processing filters
6. Saving final change maps

Generated outputs are stored under:

```text
output/predicted_masks/
```

---

## Repository Structure

```text
geospatial-change-detection-engine/
│
├── data/
├── models/
├── preprocessing/
├── inference/
├── notebooks/
├── output/
├── requirements.txt
└── README.md
```

---

## Quick Start

### Clone Repository

```bash
git clone https://github.com/kuchurisatwik/geospatial-change-detection-engine.git

cd geospatial-change-detection-engine
```

### Create Environment

```bash
python -m venv venv

source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Inference

```bash
python inference/detect_changes.py \
  --model-path models/model.pth \
  --input-before before_image.png \
  --input-after after_image.png \
  --output-dir output/predicted_masks/
```

---

## Future Enhancements

* Siamese U-Net architectures
* Attention-based segmentation networks
* Transformer-based remote sensing models
* Multi-class change detection
* GIS export formats (GeoJSON, Shapefile)
* REST API deployment
* Docker support
* CI/CD integration

---

## Author

Satwik Kuchuri

Machine Learning • Computer Vision • Cloud & DevOps

---

## License

This project is released for educational and research purposes.
