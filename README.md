# 🛡️ Deepfake Video Detection Using Machine Learning & Facial Feature Analysis

A dual-stage deepfake detection system combining facial feature extraction, a custom 4-stage **Convolutional Neural Network (CNN)** baseline, and a hybrid **CoAtNet (Convolution + Self-Attention Transformer)** architecture.

Includes an interactive **Streamlit Forensic Web App** for live video analysis, a **CLI tool**, and end-to-end training and evaluation pipelines.

---

## 📑 Architecture Overview

```text
Raw Video (.mp4 / .avi)
         │
         ▼
[ Frame Sampling ] (20 frames distributed across video)
         │
         ▼
[ Face Detection & Extraction ] (OpenCV Haar Cascade + HSV Equalization)
         │
         ▼
[ Deep Learning Classifier ]
   ├── 1. CoAtNet: MBConv Depthwise Blocks + Multi-Head Self-Attention (PyTorch)
   └── 2. CNN: 4-Stage Conv2D + BatchNorm + Dropout (Keras / TensorFlow)
         │
         ▼
[ Forensic Verdict & Confidence Meter ] (Authentic vs. Deepfake)
```

---

## 🌟 Key Features

- **Interactive Streamlit Web Dashboard (`app.py`)**:
  - Drag-and-drop video upload (`.mp4`, `.avi`, `.mov`).
  - Synthetic test video generator for rapid testing.
  - Frame-by-frame facial extraction & bounding box visualization.
  - Interactive temporal manipulation risk timeline graph.
  - Downloadable JSON Forensic Audit Report.
- **Modular Deep Learning Models (`models/`)**:
  - `models/coatnet.py`: PyTorch CoAtNet combining `MBConv` with `TransformerBlock`.
  - `models/cnn.py`: 4-stage Keras CNN with batch normalization and dropout.
- **Standalone CLI Inference Tool (`predict.py`)**:
  - Run predictions on any video directly from the terminal.
- **Training Pipelines (`.ipynb`)**:
  - `coatnet_uadfv_pipeline.ipynb`: PyTorch training on UADFV video dataset.
  - `cnn_udafv_new.ipynb`: Keras training on UADFV video dataset.
  - `cnn_fakeavceleb_streamsplit.ipynb`: 4-class multi-modal detection on FakeAVCeleb v1.2 with disk-cached streaming generator.

---

## 🚀 Quick Start Guide

### 1. Installation
Install all dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
To run the interactive UI:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can upload any `.mp4`/`.avi` video or click **"Use Synthetic Demo Video"** to test immediately.

### 3. Run Inference from Terminal
```bash
# Using CoAtNet (PyTorch)
python predict.py --video path/to/video.mp4 --model coatnet

# Using 4-Stage CNN (TensorFlow/Keras)
python predict.py --video path/to/video.mp4 --model cnn

# With custom trained weights
python predict.py --video path/to/video.mp4 --model coatnet --weights coatnet_uadfv.pt
```

---

## 📁 Dataset Folder Organization

To train using the notebooks, organize your video datasets as follows:

```text
data/
└── uadfv/
    ├── Real/
    │   ├── 0001.mp4
    │   └── ...
    └── Fake/
        ├── 0001_fake.mp4
        └── ...
```

---

## 📊 Evaluation Metrics
Models are evaluated across:
- **Accuracy**
- **Precision, Recall, F1-Score**
- **ROC-AUC Curve**
- **Confusion Matrix**
