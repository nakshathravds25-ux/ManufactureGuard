# ManufactureGuard — Surface Defect Classification with Texture Features & ML

ManufactureGuard is an end-to-end computer-vision project that classifies **six steel surface defect types** from the NEU Surface Defect Database using handcrafted texture descriptors and classical machine-learning models.

> **Task:** multiclass classification — crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.  
> The NEU-DET dataset contains defect images only, so this project does **not** perform defect-vs-non-defect classification.

## Highlights

- Multi-scale **Local Binary Patterns (LBP)**
- **GLCM** texture statistics
- **Gabor** filter-bank features
- ANOVA **SelectKBest** feature selection
- **SVM (RBF)**, **Random Forest**, and **Gradient Boosting**
- Held-out train/validation/test workflow
- Multiclass metrics, confusion matrices, one-vs-rest ROC curves, and feature importance
- Streamlit inference application

## Dataset

**NEU Surface Defect Database**

The dataset contains 1,800 grayscale images across six surface-defect classes, with 300 images per class.

Dataset source: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

## Pipeline

```text
NEU-DET images
      ↓
LBP + GLCM + Gabor feature extraction
      ↓
Standardization
      ↓
ANOVA SelectKBest
      ↓
SVM / Random Forest / Gradient Boosting
      ↓
6-class defect prediction
      ↓
Accuracy + Macro-F1 + ROC-AUC + confusion matrices
```

## Feature Engineering

### Local Binary Patterns
LBP descriptors are extracted at multiple radii to capture local texture micro-patterns.

### Gray-Level Co-occurrence Matrix
GLCM-derived features include energy, contrast, dissimilarity, homogeneity, ASM, and correlation across multiple pixel distances.

### Gabor Filters
A multi-frequency, multi-orientation Gabor filter bank captures oriented texture patterns that are useful for defects such as scratches and rolled-in scale.

The combined feature vector is reduced using `SelectKBest(f_classif)` before model training.

## Models

- **SVM (RBF kernel)**
- **Random Forest**
- **Gradient Boosting**

All models are trained as native multiclass classifiers.

## Evaluation

The evaluation pipeline reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- One-vs-rest ROC-AUC
- Per-class classification report
- Confusion matrices
- Feature-importance visualizations where supported

Exact metrics should be reproduced by running the training pipeline on the documented dataset split rather than treated as fixed benchmark values.

## Project Structure

```text
ManufactureGuard/
├── app.py
├── train.py
├── requirements.txt
├── models/
├── output/
└── src/
    ├── config.py
    ├── data_loader.py
    ├── evaluation.py
    ├── feature_extraction.py
    ├── models.py
    ├── training.py
    └── utils.py
```

## Run Locally

```bash
git clone https://github.com/nakshathravds25-ux/ManufactureGuard.git
cd ManufactureGuard
pip install -r requirements.txt
```

Place the NEU-DET dataset under `./data/`, then run:

```bash
python train.py
streamlit run app.py
```

## Live Demo

https://manufactureguard.streamlit.app/

## Tech Stack

Python · scikit-learn · scikit-image · NumPy · Pandas · Matplotlib · Streamlit

## Scope

This project demonstrates how engineered texture features and classical machine-learning models can be used for multiclass industrial surface-defect recognition. It is an academic/portfolio implementation and is not presented as a production quality-control system.
