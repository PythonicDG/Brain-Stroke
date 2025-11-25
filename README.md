
Then connect them here:

### **Confusion Matrix**
<img src="assets/confusion_matrix.png" width="550">

### **ROC Curve**
<img src="assets/roc_curve.png" width="550">

### **Precision-Recall Curve**
<img src="assets/pr_curve.png" width="550">

### **Probability Distribution**
<img src="assets/prob_distribution.png" width="550">

### **Threshold vs F1-Score**
<img src="assets/threshold_curve.png" width="550">

---

# 🧠 Why This Approach (Our Flow) Works Better Than Research-Paper Baselines

- ✔ **End-to-end CNN**, not GA + BiLSTM complexity  
- ✔ **Modern EfficientNet-B3 backbone**, not old AlexNet/VGG  
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
