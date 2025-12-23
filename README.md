# Small ML Tasks

Machine Learning Übungsprojekte mit Python und scikit-learn.

## Projekte

### 01 - ML Landscape
Einführung in die ML-Landschaft

### 02 - End-to-End ML Project
Vollständiges ML-Projekt mit dem California Housing Dataset:
- Data Loading & Exploration
- Feature Engineering
- Multiple Models (Linear Regression, Decision Tree, Random Forest, SVR)
- Hyperparameter Tuning mit GridSearch & RandomizedSearch
- Cross-Validation

### 03 - Classification
Klassifikation mit MNIST Dataset:
- Binary Classification (5 vs nicht-5)
- Multiclass Classification (alle 10 Ziffern)
- Error Analysis
- Multilabel Classification
- Multioutput Classification (Denoising)
- KNN GridSearch für >97% Accuracy
- Data Augmentation durch Image Shifting
- Titanic Survival Prediction

## Setup

```bash
# Virtual Environment erstellen
python -m venv .SmallMLTasks

# Aktivieren (Windows)
.SmallMLTasks\Scripts\activate

# Dependencies installieren
pip install scikit-learn pandas matplotlib numpy scipy
```

## Ausführen

```bash
python 02_End2EndML.py
python 03_Classification.py
```
