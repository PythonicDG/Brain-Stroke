# 🧠 Stroke CT Classifier — EfficientNet-B3 Ensemble
A high-accuracy deep-learning system for **binary classification of Brain CT scans**  
(**Normal vs Stroke**), built with state-of-the-art modeling techniques, strong regularization,  
and robust production-ready deployment.

---

## 🚀 Overview

This project delivers a **full end-to-end pipeline** for training, validating, and deploying  
an EfficientNet-B3 model with advanced strategies including:

- 5-fold cross-validation  
- Hard Negative Mining (HNM)  
- Focal Loss to counter class imbalance  
- Test-Time Augmentation (TTA)  
- Ensemble averaging  
- Flask inference API for production  
- Optimized threshold tuning for clinical use  

---

## ✨ Key Features

- 🧩 **EfficientNet-B3 (Noisy Student)** pretrained backbone  
- 🔄 **5-fold ensemble** for stable predictions  
- 🎚️ **Optimized threshold**: `0.425` for best sensitivity–specificity balance  
- 🧪 **TTA** (horizontal & vertical flips)  
- 🎯 **Hard Negative Mining** applied at epoch 6  
- ⚖️ **Focal Loss** for imbalanced classes  
- 📈 **Complete evaluation** (ROC, PR, OOF curves, confusion matrix)  
- 🔥 **Fast inference** + optional **Grad-CAM explainability**  

---

## 📊 Final Holdout Performance

| Metric | Score |
|--------|--------|
| **Accuracy** | **95.49%** |
| **ROC-AUC** | **0.9908** |
| **Optimal Threshold** | **0.425** |
| **Holdout Size** | **886 CT slices** |

### Class-wise Metrics

| Class | Description | Precision | Recall | F1-Score | Support |
|-------|-------------|-----------|--------|----------|---------|
| **0** | Normal | 0.9786 | 0.9300 | 0.9537 | 443 |
| **1** | Stroke | 0.9333 | 0.9797 | 0.9559 | 443 |

---

## 📈 Visualization

### 🔢 Confusion Matrix  
<img src="assets/confusion_matrix.png" width="550">

### 📉 ROC Curve  
<img src="assets/roc_curve.png" width="550">

### 📊 Probability Distribution  
<img src="assets/prob_distribution.png" width="550">

---

## 🧠 Why This Model?

- ✔ **End-to-end CNN with modern EfficientNet backbone**  
- ✔ **Stable predictions** via TTA + model ensembling  
- ✔ **Clinical safety emphasis** — Focal Loss + HNM reduces false negatives  
- ✔ **Lightweight & fast** compared to multi-CNN genetic-algorithm pipelines  
- ✔ **Grad-CAM heatmaps** for explainability  
- ✔ **Easy deployment** with a simple Flask API  

---

---

## 🔧 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Stroke-CT-Classifier.git
cd Stroke-CT-Classifier
pip install -r requirements.txt
cd app
python app.py
