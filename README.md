# 🛰️ Satellite Change Detection using Deep Learning (PS10)

This repository implements a **Satellite Image Change Detection system** using deep learning to automatically identify changes between satellite images captured at different times.

The project follows a **clean, modular, and practical structure**, making it suitable for:

- Learning change detection concepts  
- Academic reference  
- Portfolio and resume projects  
- Real-world remote sensing workflows  

---

## 📌 Problem Statement

Satellite imagery is widely used to monitor:

- Urban expansion  
- Infrastructure development  
- Environmental changes  
- Disaster impacts (floods, earthquakes, deforestation)  

Manually comparing satellite images is **slow, expensive, and error-prone**.

This project automates the process using a **deep learning–based semantic segmentation approach** to detect changes at the **pixel level**.

---

## 🧠 Solution Overview

The solution uses a **U-Net convolutional neural network** to generate **binary change maps** between two satellite images captured at different times.

### High-Level Workflow

1. Input satellite image pairs (Before & After)  
2. Preprocess images and extract patches  
3. Perform inference using a trained U-Net model  
4. Apply post-processing to clean predictions  
5. Generate final change detection masks  

---

## 📂 Repository Structure

satellite-change-detection-ps10/
│
├── data/
│   └── LEVIR-CD/                  # Dataset directory (before, after, labels)
│
├── models/
│   └── unet_model.py              # U-Net model architecture
│
├── preprocessing/
│   └── patch_extraction.py        # Patch generation (256x256 sliding window)
│
├── inference/
│   ├── detect_changes.py          # Main inference pipeline
│   └── post_processing.py         # Morphological filtering & cleanup
│
├── notebooks/
│   └── experiments.ipynb          # Experiments and visual analysis
│
├── output/
│   └── predicted_masks/           # Generated change detection outputs
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore
---

## 📊 Dataset

- **Dataset:** LEVIR-CD  
- **Type:** High-resolution satellite imagery  
- **Task:** Binary change detection  
- **Labels:** Change / No-change masks  

⚠️ Dataset files are **not included** in this repository due to size constraints.

After downloading the dataset, place it inside:

data/LEVIR-CD/

---

## 🏗️ Model Architecture

- **Model Used:** U-Net  

### Why U-Net?

- Designed for pixel-level segmentation  
- Skip connections preserve spatial details  
- Widely used in satellite and medical imaging tasks  

**Input:** Before and After images (stacked or paired)  
**Output:** Binary segmentation mask highlighting changed regions  

---

## ⚙️ Preprocessing

### Patch Extraction

- Large satellite images are split into **256 × 256 patches**
- Advantages:
  - Efficient memory usage  
  - Faster inference  
  - Better model generalization  

### Normalization

- Pixel values are normalized to improve model stability and prediction quality  

---

## 🔍 Inference Pipeline

1. Load the trained U-Net model  
2. Process satellite image patches  
3. Predict binary change masks  
4. Reconstruct full-size output from patches  
5. Apply post-processing filters  

---

## 🧹 Post-Processing

To improve prediction quality, the following steps are applied:

- Morphological opening and closing  
- Noise reduction  
- Removal of small false-positive regions  

This results in **cleaner and more realistic change detection maps**.

---

## 📤 Output

- Binary change detection masks  
- **White pixels:** Change detected  
- **Black pixels:** No change  

Outputs can be:

- Visualized directly  
- Used for GIS-based analysis  
- Extended to vector representations (e.g., shapefiles)  

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

2️⃣ Prepare Dataset

Place the dataset in:

data/LEVIR-CD/

3️⃣ Run Change Detection

python inference/detect_changes.py

4️⃣ View Results

Check the generated outputs inside:

output/predicted_masks/


---

🧪 Experiments

Jupyter notebooks are included for:

Model testing

Visualization

Experimental analysis


Location:

notebooks/


---

🛠️ Technologies Used

Python

PyTorch / TensorFlow (depending on implementation)

NumPy

OpenCV

Scikit-image

Jupyter Notebook



---

📌 Future Improvements

Model training scripts

Multi-class change detection

GIS vector output (Shapefiles)

Cloud deployment (AWS / GCP)

CI/CD pipeline integration



---

📜 License

This project is intended for educational and research purposes.

