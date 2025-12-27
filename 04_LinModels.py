import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. LINEAR REGRESSION - Basis für alle linearen Modelle
# ============================================================================
rng = np.random.default_rng(seed=42)
X = 2 * rng.random((200, 1))
y = 4 + 3 * X + rng.standard_normal((200, 1))

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)
print(f"Intercept: {lin_reg.intercept_[0]:.4f} (erwartet: 4)")
print(f"Coefficient: {lin_reg.coef_[0][0]:.4f} (erwartet: 3)")

# ============================================================================
# 2. POLYNOMIAL REGRESSION - Für nicht-lineare Beziehungen
# Wandelt x in [x, x², x³, ...] um, dann lineare Regression
# ============================================================================
print("\n" + "=" * 60)
print("POLYNOMIAL REGRESSION")
print("=" * 60)

# Daten: y = 0.5x² + x + 2 + Noise
X_poly = 6 * rng.random((200, 1)) - 3
y_poly = 0.5 * X_poly**2 + X_poly + 2 + rng.standard_normal((200, 1))

# Verschiedene Polynomial-Grade testen
for degree in [1, 2, 10]:
    poly_reg = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),  # Wichtig bei hohen Graden!
        LinearRegression()
    )
    poly_reg.fit(X_poly, y_poly)
    y_pred = poly_reg.predict([[0]])
    print(f"Degree {degree:2d}: Prediction für X=0: {y_pred[0][0]:.4f}")

print("Erwartet: ~2.0 | Degree 1: Underfitting | Degree 10: Overfitting")

# ============================================================================
# 3. REGULARIZATION - Verhindert Overfitting
# ============================================================================
print("\n" + "=" * 60)
print("REGULARIZATION")
print("=" * 60)

# Kleine Datenmenge (anfällig für Overfitting)
X_small = 3 * rng.random((20, 1))
y_small = 1 + 0.5 * X_small + rng.standard_normal((20, 1)) / 1.5

# Ridge (L2): Bestraft große Gewichte quadratisch
# Lasso (L1): Bestraft große Gewichte linear, kann Features auf 0 setzen
print("\nRidge (L2 Regularization):")
for alpha in [0, 0.1, 10]:
    ridge = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
    ridge.fit(X_small, y_small.ravel())
    print(f"  Alpha={alpha:4.1f}: Coef={ridge.coef_[0]:.4f}")

print("\nLasso (L1 Regularization):")
for alpha in [0, 0.1, 1.0]:
    lasso = Lasso(alpha=alpha) if alpha > 0 else LinearRegression()
    lasso.fit(X_small, y_small.ravel())
    print(f"  Alpha={alpha:4.1f}: Coef={lasso.coef_[0]:.4f}")

print("\nHöheres Alpha = stärkere Regularization = kleinere Coefficients")

# ============================================================================
# 4. LOGISTIC REGRESSION - Für Klassifikation (nicht Regression!)
# Gibt Wahrscheinlichkeiten aus: P(y=1|X)
# ============================================================================
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (Binary Classification)")
print("=" * 60)

# Iris Dataset: Virginica vs Rest
iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == "virginica"
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Vorhersagen für verschiedene Petal Widths
for width in [1.0, 1.5, 2.0]:
    proba = log_reg.predict_proba([[width]])[0, 1]
    pred = "Virginica" if proba >= 0.5 else "Nicht Virginica"
    print(f"Width={width:.1f}cm: P(Virginica)={proba:.2f} → {pred}")

# ============================================================================
# 5. SOFTMAX REGRESSION - Multiclass Classification
# Logistic Regression für >2 Klassen
# ============================================================================
print("\n" + "=" * 60)
print("SOFTMAX REGRESSION (Multiclass)")
print("=" * 60)

# Alle 3 Iris-Arten klassifizieren
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

# Vorhersage für eine Blume
test_flower = [[5, 2]]  # Length=5cm, Width=2cm
prediction = softmax_reg.predict(test_flower)[0]
probas = softmax_reg.predict_proba(test_flower)[0]

print(f"\nTest Flower (Length=5cm, Width=2cm):")
print(f"Prediction: {iris.target_names[prediction]}")
print(f"Probabilities:")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {probas[i]:.2f}")

# Accuracy auf Test Set
accuracy = softmax_reg.score(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2%}")

print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
print("Linear Regression:    y = wx + b")
print("Polynomial:           Erweitert Features zu [x, x², x³, ...]")
print("Ridge (L2):           Bestraft große Gewichte quadratisch")
print("Lasso (L1):           Kann Features eliminieren (Coef → 0)")
print("Logistic Regression:  Binary Classification mit Wahrscheinlichkeiten")
print("Softmax:              Multiclass Classification")
