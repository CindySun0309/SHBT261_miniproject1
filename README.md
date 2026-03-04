# SHBT261 Mini Project 1 — Image Classification on Caltech-101

**Course:** SHBT 261 AI in Medicine  
**Author:** Cindy Sun  
**Institution:** Harvard University, Spring 2026

---

## Overview

This project systematically compares four image classification approaches on the [Caltech-101 dataset](https://www.kaggle.com/datasets/imbikramsaha/caltech-101):

| Model | Type | Params |
|---|---|---|
| SVM + HOG | Classical ML | RBF kernel, hand-crafted HOG features |
| ResNet-18 | CNN | ~11M, pretrained on ImageNet |
| EfficientNet-B0 | CNN | ~5.3M, pretrained on ImageNet |
| ViT-Small/16 | Vision Transformer | ~22M, pretrained on ImageNet-21k |

An exhaustive **grid search across 107 configurations** (4 SVM + 103 deep learning) is performed to identify the best hyperparameter settings for each model. Four ablation studies are conducted to isolate the effect of image size, data augmentation, optimizer, and learning rate.

---

## Key Results

| Model | Accuracy | Top-5 Accuracy | Macro F1 | Best Config | Train Time |
|---|---|---|---|---|---|
| **ViT-Small** | **98.23%** | 99.69% | 0.974 | img=224, ep=10, lr=1e-3, AdamW, aug=False | 7.5 min |
| EfficientNet-B0 | 97.39% | 99.85% | 0.962 | img=224, ep=10, lr=1e-3, Adam, aug=True | 4.5 min |
| ResNet-18 | 97.0% | 99.77% | 0.954 | img=224, ep=5, lr=1e-3, AdamW, aug=False | 1.3 min |
| SVM + HOG | 67.36% | 84.18% | 0.531 | img=64, C=10, aug=False | 3.8 min |

> **Best efficiency-accuracy tradeoff:** ResNet-18 achieves 97% accuracy in just 1.3 minutes of training.

---

## Dataset

- **Source:** [Caltech-101 on Kaggle](https://www.kaggle.com/datasets/imbikramsaha/caltech-101)
- **Classes:** 101 object categories
- **Total images:** ~8,677
- **Split:** 70% train / 15% validation / 15% test (stratified, random seed 42)

---

## Project Structure

```
SHBT261_miniproject1/
├── data/
│   └── caltech-101/                  ← dataset (download separately)
├── figures/
│   └── experiments/                  ← all generated plots
│       ├── comparison_summary.png
│       ├── ablation_summary.png
│       ├── top_bottom5_per_class_accuracy.png
│       ├── all_models_per_class_accuracy.png
│       ├── top_misclassified_pairs.png
│       ├── confusion_matrix_worst_classes.png
│       ├── *_best_confusion_matrix.png
│       └── *_curves.png
├── models/
│   ├── experiments/                  ← all trained model weights (.pth / .pkl)
│   └── best_models/                  ← best model per architecture
│       ├── best_svm.pkl
│       ├── best_svm_scaler.pkl
│       ├── best_resnet18.pth
│       ├── best_efficientnet_b0.pth
│       ├── best_vit_small.pth
│       └── README.md
├── results/
│   ├── all_results.json              ← raw results for all 107 runs
│   └── summary_table.csv             ← sortable metrics table
└── final_code.ipynb                  ← main notebook
```

---

## Grid Search Configuration

### SVM
| Parameter | Values |
|---|---|
| Image size | 64×64, 128×128 |
| C | 10 |
| Augmentation | True, False |

### Deep Learning
| Parameter | Values |
|---|---|
| Models | ResNet-18, EfficientNet-B0, ViT-Small |
| Image size | 128×128, 224×224 (ViT: 224 only) |
| Epochs (per phase) | 5, 10 |
| Learning rate | 1e-3, 1e-4 |
| Optimizer | Adam, AdamW, SGD |
| Augmentation | True, False |
| Batch size | 32 (fixed) |

**Total combinations:** 107 (4 SVM + 103 DL)

---

## Training Strategy

Deep learning models use **2-phase transfer learning:**

- **Phase 1** — Freeze pretrained backbone, train classification head only
- **Phase 2** — Unfreeze entire network, fine-tune at lr/10

Both phases use cosine annealing scheduling and cross-entropy loss with label smoothing (ε=0.1). The grid search is **resume-safe** — completed runs are skipped automatically so training can be interrupted and restarted without loss of progress.

---

## Ablation Studies

| Ablation | Variable | Finding |
|---|---|---|
| Image size | 128×128 vs 224×224 | 224×224 consistently better; largest gain for EfficientNet-B0 (+9%) |
| Augmentation | With vs without | Neutral or slightly negative across all models on this dataset |
| Optimizer | Adam vs AdamW vs SGD | Adam/AdamW >> SGD; SGD drops to ~50% avg for EfficientNet-B0 |
| Learning rate | 1e-3 vs 1e-4 | 1e-3 dominates by ~20–25% for all DL models |

---

## Requirements

```bash
pip install torch torchvision timm
pip install scikit-learn scikit-image
pip install numpy pandas matplotlib seaborn
pip install tqdm joblib Pillow rich
```

> **Environment:** The notebook is designed to run on an **OnDemand server** with `NUM_WORKERS=12`. Adjust this value in Cell 5 if running locally.

---

## How to Run

1. Download the Caltech-101 dataset from [Kaggle](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) and place it at:
   ```
   data/caltech-101/
   ```

2. The notebook auto-detects your project directory from `os.path.expanduser("~")`. Ensure the folder is named `SHBT261_miniproject1` in your home directory, or update `PROJECT_DIR` in Cell 2:
   ```python
   PROJECT_DIR = Path("/your/path/to/SHBT261_miniproject1")
   ```

3. Run all cells in order. The notebook will:
   - Extract and cache HOG features on first run (reused on subsequent runs)
   - Skip any already-completed experiment runs automatically
   - Save all results to `results/all_results.json` after each run
   - Generate all figures and export a summary CSV

> **Estimated runtime:** ~8–12 hours on an NVIDIA GPU (e.g. Colab T4) · ~20–30 hours on Apple Silicon MPS · Varies on OnDemand depending on available resources.

---

## Outputs

| File | Description |
|---|---|
| `results/summary_table.csv` | All 107 runs sorted by accuracy |
| `results/all_results.json` | Full raw results including per-class accuracy and confusion matrices |
| `figures/experiments/comparison_summary.png` | Best accuracy per model + accuracy vs time scatter + augmentation effect |
| `figures/experiments/ablation_summary.png` | 4-panel ablation study |
| `figures/experiments/top_bottom5_per_class_accuracy.png` | Top 5 & bottom 5 classes per model |
| `figures/experiments/confusion_matrix_worst_classes.png` | Confusion matrix zoomed to worst 20 classes |
| `figures/experiments/top_misclassified_pairs.png` | Top 15 most frequent misclassification pairs |
| `figures/experiments/all_models_per_class_accuracy.png` | Full per-class accuracy for all 101 classes |
| `figures/experiments/*_best_confusion_matrix.png` | Full 101×101 confusion matrix per model (Appendix) |
| `models/best_models/` | Best model weights per architecture |
