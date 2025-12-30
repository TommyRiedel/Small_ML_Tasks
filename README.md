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

### 04 - Linear Models
Lineare Modelle für Regression und Klassifikation:
- Linear Regression
- Polynomial Regression (Under-/Overfitting)
- Regularization (Ridge L2, Lasso L1, Elastic Net)
- Logistic Regression (Binary Classification)
- Softmax Regression (Multiclass Classification)

### 05 - Decision Trees
Entscheidungsbäume für Klassifikation und Regression:
- Decision Tree Classifier (Iris Dataset)
- Decision Tree Regressor (Quadratische Funktion)
- Overfitting vermeiden (Regularization Parameter)
- Hyperparameter: max_depth, min_samples_leaf, max_features
- Vor- und Nachteile von Decision Trees

### 06 - Ensemble Learning
Ensemble-Methoden für bessere Vorhersagen:
- Voting Classifier (Hard Voting)
- Bagging (Bootstrap Aggregating)
- Random Forest (Feature Importances)
- AdaBoost (Adaptive Boosting)

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
python 04_LinModels.py
python 05_DecisionTrees.py
python 06_EnsembleLearning.py
```
