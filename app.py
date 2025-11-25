import os
import cv2
import numpy as np
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import albumentations as A
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 384
MODEL_PATHS = [
    "best_fold_1.pth",
    "best_fold_2.pth",
    "best_fold_3.pth",
    "best_fold_4.pth",
    "best_fold_5.pth",
]

# ------------------------------------------------------
# PREPROCESSING — EXACTLY FROM NOTEBOOK
# ------------------------------------------------------

val_aug = A.Compose([
    A.Normalize()
])

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    augmented = val_aug(image=img)
    img = augmented["image"]          # (H,W)

    img = np.stack([img, img, img], axis=-1)  # (H,W,3)
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))        # (3,H,W)

    return torch.tensor(img).unsqueeze(0).to(device)


# ------------------------------------------------------
# MODEL LOADING (EXACT NOTEBOOK ARCHITECTURE)
# ------------------------------------------------------

def create_model():
    backbone = timm.create_model(
        "tf_efficientnet_b3_ns",
        pretrained=False,
        num_classes=0,
        global_pool="avg"
    )
    in_features = backbone.num_features
    head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )
    model = nn.Sequential(backbone, head)
    return model.to(device)

models = []
for p in MODEL_PATHS:
    m = create_model()
    m.load_state_dict(torch.load(p, map_location=device))
    m.eval()
    models.append(m)

# ------------------------------------------------------
# INFERENCE (ENSEMBLE)
# ------------------------------------------------------

def predict(img_tensor):
    probs = []
    with torch.no_grad():
        for m in models:
            logits = m(img_tensor)
            if logits.ndim > 2:
                logits = logits.view(-1, 1)
            p = torch.sigmoid(logits).item()
            probs.append(p)

    mean_prob = sum(probs) / len(probs)
    label = "Abnormal (Stroke)" if mean_prob > 0.5 else "Normal"
    return label, mean_prob


# ------------------------------------------------------
# FLASK APP
# ------------------------------------------------------

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]

        path = "uploaded.png"
        file.save(path)

        img_tensor = preprocess_image(path)
        label, prob = predict(img_tensor)

        prediction = f"{label} — Confidence: {prob:.4f}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
