# SHBT261 Mini Project 1 — Image Classification on Caltech-101

**Course:** SHBT 261 AI in Medicine  
**Author:** Cindy Sun  
**Institution:** Harvard University, Spring 2026

---

## Overview

This project trains and evaluates multiple image classification models on the [Caltech-101 dataset](https://www.kaggle.com/datasets/imbikramsaha/caltech-101), comparing classical machine learning and deep learning approaches. A systematic **grid search** is performed across model architectures and hyperparameters to identify the best configuration for each model type.

---

## Models

| Model | Type | Parameters |
|---|---|---|
| SVM + HOG | Classical ML | Hand-crafted HOG features + RBF kernel SVM |
| ResNet-18 | Deep Learning (CNN) | ~11M, pretrained on ImageNet |
| EfficientNet-B0 | Deep Learning (CNN) | ~5.3M, pretrained on ImageNet |
| ViT-Small/16 | Deep Learning (Transformer) | ~22M, pretrained on ImageNet-21k |

---

## Dataset

- **Source:** [Caltech-101](https://www.kaggle.com/datasets/imbikramsaha/caltech-101)
- **Classes:** 101 object categories
- **Total images:** ~9,000
- **Split:** 70% train / 15% validation / 15% test (stratified)

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
| Epochs | 5, 10 (per phase) |
| Learning rate | 1e-3, 1e-4 |
| Optimizer | Adam, AdamW, SGD |
| Augmentation | True, False |
| Batch size | 32 |

**Total combinations:** 107 (4 SVM + 103 DL)

---

## Training Strategy

Deep learning models use **2-phase transfer learning**:
- **Phase 1** — Freeze pretrained backbone, train only the classification head
- **Phase 2** — Unfreeze entire network, fine-tune with 10× smaller learning rate

---

## Evaluation Metrics

Each model is evaluated using:
- Overall accuracy
- Top-5 accuracy
- Per-class accuracy
- Confusion matrix
- Precision, Recall, F1-Score (macro & weighted)
- Training time

---

## Ablation Studies

Four ablation experiments are conducted across all models:
1. **Image size** — 128×128 vs 224×224
2. **Data augmentation** — with vs without (random flip, rotation, color jitter)
3. **Optimizer** — Adam vs AdamW vs SGD
4. **Learning rate** — 1e-3 vs 1e-4

---

## Project Structure

```
SHBT261_miniproject1/
├── data/
│   └── caltech-101/          ← dataset (download separately)
├── figures/
│   └── experiments/          ← all generated plots
│       ├── comparison_summary.png
│       ├── ablation_summary.png
│       ├── *_best_confusion_matrix.png
│       └── *_curves.png
├── models/
│   ├── experiments/          ← all trained model weights (.pth / .pkl)
│   └── best_models/          ← best model per architecture
│       ├── best_svm.pkl
│       ├── best_svm_scaler.pkl
│       ├── best_resnet18.pth
│       ├── best_efficientnet_b0.pth
│       ├── best_vit_small.pth
│       └── README.md
├── results/
│   ├── all_results.json      ← raw results for all 107 runs
│   └── summary_table.csv     ← sortable metrics table
└── experiment_grid.ipynb     ← main notebook
```

---

## Requirements

```bash
pip install torch torchvision timm
pip install scikit-learn scikit-image
pip install numpy pandas matplotlib seaborn
pip install tqdm joblib Pillow
```

---

## How to Run

1. Download the Caltech-101 dataset from [Kaggle](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) and place it at `data/caltech-101/`

2. Update `PROJECT_DIR` in Cell 2 to your local path:
```python
PROJECT_DIR = Path("/your/path/to/SHBT261_miniproject1")
```

3. Run all cells in order. The notebook will:
   - Automatically skip any already-completed runs (resume-safe)
   - Save all results to `results/all_results.json` after each run
   - Generate all figures and export a summary CSV

> **Note:** The full grid search takes several hours. On an NVIDIA GPU (e.g. Colab T4) expect ~8–12 hours total. On Apple Silicon MPS expect ~20–30 hours.

---

## Key Results

See `results/summary_table.csv` for full results. Best configurations per model are saved in `models/best_models/` with details in `models/best_models/README.md`.

---

## Outputs

| File | Description |
|---|---|
| `results/summary_table.csv` | All 107 runs sorted by accuracy |
| `figures/experiments/comparison_summary.png` | Accuracy vs time scatter + augmentation effect |
| `figures/experiments/ablation_summary.png` | 4-panel ablation study plots |
| `figures/experiments/*_best_confusion_matrix.png` | Per-model confusion matrices |
| `figures/experiments/all_models_per_class_accuracy.png` | Per-class accuracy for all models |
