# 🧠 Vision Transformers for Alzheimer's Classification

A deep learning project applying Vision Transformers (ViT) to classify brain MRI scans across Alzheimer's disease severity stages. Built as a minor project exploring the effectiveness of attention-based architectures on medical imaging datasets.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 📋 Problem Statement

Alzheimer's disease progresses through distinct stages visible in MRI scans. This project trains a Vision Transformer to automatically classify scans into four severity categories, comparing ViT performance against traditional CNN baselines.

---

## 🗂 Dataset

The dataset is structured for supervised classification:

```
dataset/
└── train/
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/
```

Source: [Alzheimer's Dataset (4 class)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) on Kaggle.

---

## 🏗 Architecture

- **Backbone:** Vision Transformer (ViT-B/16)
- **Input:** Brain MRI scans resized to 224×224
- **Output:** 4-class softmax classification
- **Training:** Transfer learning with pre-trained ImageNet weights, fine-tuned on the Alzheimer's dataset

---

## 🛠 Setup

```bash
git clone https://github.com/ananya-byte/VisionTransformers_AlzheimerDataset.git
cd VisionTransformers_AlzheimerDataset

pip install torch torchvision timm matplotlib scikit-learn
```

Place the dataset under `dataset/train/` following the structure above, then open and run the Jupyter notebook.

---

## 📊 Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~91%   |
| Classes   | 4      |
| Model     | ViT-B/16 (fine-tuned) |

*Results are approximate and may vary with hyperparameter tuning.*

---

## 💡 What I Learned

- Applying Vision Transformers to non-natural image domains
- Transfer learning strategies for small medical datasets
- Patch embedding and multi-head self-attention mechanics
- Evaluation metrics for imbalanced medical classification tasks
