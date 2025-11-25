# 🧠 Stroke CT Classifier — EfficientNet-B3 Ensemble

A high-accuracy deep-learning system for **binary classification of Brain CT scans** (Normal vs Stroke).  
Built using **EfficientNet-B3**, trained with **5-fold cross-validation**, **Hard Negative Mining**,  
**Focal Loss**, and **Test-Time Augmentation (TTA)**.

This project includes:

- End-to-end training notebook  
- Flask inference API  
- Production threshold tuning  
- Robust preprocessing pipeline  
- Ensemble-based prediction for stability  

---

# 🚀 Key Features

- 🔍 **EfficientNet-B3 (Noisy Student) backbone**  
- 📦 **5-fold model ensemble** for high stability  
- 🎯 **Optimized decision threshold** (0.425)  
- 💡 **Hard Negative Mining** (HNM) at epoch 6  
- 🧪 **Test-Time Augmentation** (horizontal + vertical flips)  
- ⚙️ **Focal Loss** to reduce class imbalance impact  
- 📊 **Comprehensive evaluation with ROC, PR, OOF curves**

---

# 📊 Final Performance (Holdout Set)

| Metric | Score |
|--------|--------|
| **Accuracy** | **95.49%** |
| **ROC-AUC** | **0.9908** |
| **Best Threshold** | **0.425** |
| **Holdout Size** | **886 CT slices** |

### Class-wise Metrics

| Class | Description | Precision | Recall | F1-Score | Support |
|-------|-------------|-----------|--------|----------|---------|
| 0 | Normal | 0.9786 | 0.9300 | 0.9537 | 443 |
| 1 | Abnormal (Stroke) | 0.9333 | 0.9797 | 0.9559 | 443 |

---

# 📈 Evaluation Plots

Upload your generated plots into:  


### **Confusion Matrix**
<img src="assets/confusion_matrix.png" width="550">

### **ROC Curve**
<img src="assets/roc_curve.png" width="550">

### **Precision-Recall Curve**
<img src="assets/pr_curve.png" width="550">

### **Probability Distribution**
<img src="assets/prob_distribution.png" width="550">
---

- ✔ **End-to-end CNN** 
- ✔ **Modern EfficientNet-B3 backbone**,
- ✔ **Simple production deployment** (Flask API)  
- ✔ **TTA + Ensemble** improves generalization  
- ✔ **Focal Loss + HNM** reduces false negatives (critical for stroke)  
- ✔ **Explainable Grad-CAM heatmaps** possible directly on CNN  
- ✔ **Faster, lighter, scalable** compared to multi-CNN GA pipelines  

---

# 📂 Project Structure

# 🔧 Installation

### Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/Stroke-CT-Classifier.git
cd Stroke-CT-Classifier

pip install -r requirements.txt


cd app
python app.py
http://127.0.0.1:5000
