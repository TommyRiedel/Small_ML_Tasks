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
- Voting Classifier (Hard Voting - Mehrheitsentscheidung)
- Bagging (Bootstrap Aggregating mit OOB Score)
- Random Forest (Feature Importances auf Iris Dataset)
- AdaBoost (Adaptive Boosting mit Weak Learners)
- Vergleich: Einzelner Tree vs. Ensemble-Methoden
- Wann welche Methode: Overfitting → Bagging, Underfitting → Boosting

### 07 - Dimensionality Reduction
Dimensionsreduktion mit verschiedenen Techniken:
- PCA (Principal Component Analysis) - Varianzmaximierung
- Incremental PCA - Für große Datasets (batch-weise)
- Random Projection - Schnelle Approximation (Johnson-Lindenstrauss)
- LLE (Locally Linear Embedding) - Manifold Learning
- MDS, Isomap, t-SNE - Weitere Manifold-Methoden
- Kernel PCA - Nichtlineare Dimensionsreduktion
- Anwendung auf MNIST (784 → 154 Dimensionen bei 95% Varianz)

### 08 - Unsupervised Learning
Clustering-Algorithmen und unüberwachtes Lernen:
- K-Means Clustering (Elbow Method, Silhouette Score)
- Mini-Batch K-Means (für große Datasets)
- Image Segmentation (Color Quantization)
- Semi-Supervised Learning (Label Propagation)
- DBSCAN (Density-Based Clustering, Outlier Detection)
- Spectral Clustering (Graph-Based)
- Agglomerative Clustering (Hierarchical)
- Gaussian Mixture Models (Soft Clustering, Anomaly Detection)
- Model Selection (BIC, AIC)
- Bayesian GMM (Automatic Component Selection)

### 09 - Artificial Neural Networks
Künstliche neuronale Netze mit scikit-learn:
- Perceptron (Binary Classification auf Iris)
- MLP Regressor (California Housing mit 3 Hidden Layers)
- MLP Classifier (Fashion MNIST mit 2 Hidden Layers)
- Early Stopping zur Vermeidung von Overfitting
- Hyperparameter Tuning (hidden_layer_sizes, alpha, learning_rate)
- Probability Predictions und Confidence Analysis

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
python 07_DimReduction.py
python 08_UnsupervisedLearning.py
python 09_ANN.py
```
