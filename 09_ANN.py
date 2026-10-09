# 09 - Artificial Neural Networks (ANNs)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing, fetch_openml, fetch_covtype
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. PERCEPTRON - Simple neuronal net

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)  # Binary: Setosa vs. Rest

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)
print(f"Perceptron Predictions: {y_pred}")

# 2. MLP REGRESSOR - California Housing

try:
    housing = fetch_california_housing()
except Exception as e:
    print(f"Warning: {e}")
    from sklearn.datasets import make_regression
    X_data, y_data = make_regression(n_samples=20640, n_features=8, noise=10, random_state=42)
    class Housing:
        def __init__(self, data, target):
            self.data = data
            self.target = target
    housing = Housing(X_data, y_data)

X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)

# MLP with 3 hidden layers (50 Neuronen per layer)
mlp_reg = MLPRegressor(
    hidden_layer_sizes=[50, 50, 50],
    early_stopping=True,  # Stoppt bei Overfitting
    random_state=42
)

pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)

print(f"Best Validation Score: {mlp_reg.best_validation_score_:.4f}")
y_pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}")

# 3. MLP CLASSIFIER - Fashion MNIST

fashion_mnist = fetch_openml('Fashion-MNIST', as_frame=False)
targets = fashion_mnist.target.astype(int)
X_train, y_train = fashion_mnist.data[:60000], targets[:60000]
X_test, y_test = fashion_mnist.data[60000:], targets[60000:]

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# MLP with 2 hidden layers (200 and 100 Neuronen)
mlp_clf = MLPClassifier(
    hidden_layer_sizes=[200, 100],
    early_stopping=True,
    random_state=42
)

pipeline = make_pipeline(MinMaxScaler(), mlp_clf)
pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Best Validation Score: {mlp_clf.best_validation_score_:.4f}")

# Predictions on the first 15 test-samples
X_new = X_test[:15]
y_pred = mlp_clf.predict(X_new)
print(f"\nPredictions: {y_pred}")
print(f"Actual:      {y_test[:15]}")

# Analyze probabilities for uncertain predictions
y_proba = mlp_clf.predict_proba(X_test)
uncertain_predictions = (y_proba.max(axis=1) < 0.99).sum()
print(f"\nUnsichere Predictions (<99% Confidence): {uncertain_predictions}")

#Exerxise: Use the Covertype dataset to train a MLPClassifier and evaluate its performance.
covtype = fetch_covtype()
X_train, X_test, y_train, y_test = train_test_split(
    covtype.data, covtype.target, random_state=42)

mlp_clf = MLPClassifier(hidden_layer_sizes=[200, 100, 50], early_stopping=True,
                        verbose=True, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)

print(f"\nTest Accuracy: {pipeline.score(X_test, y_test):.4f}")