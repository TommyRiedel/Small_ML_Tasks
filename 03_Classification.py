from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

# ============================================================================
# DATEN LADEN & SPLIT
# ============================================================================
print("Lade MNIST Dataset...")
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist.data, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# ============================================================================
# 1. BINARY CLASSIFICATION: 5 vs nicht-5
# ============================================================================
print("\n" + "=" * 60)
print("BINARY CLASSIFICATION (5 vs nicht-5)")
print("=" * 60)

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# SGD mit max_iter um Convergence Warning zu vermeiden
sgd_clf = SGDClassifier(max_iter=5, tol=None, random_state=42, n_jobs=-1)
sgd_clf.fit(X_train, y_train_5)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, n_jobs=-1)

print(f"Precision: {precision_score(y_train_5, y_train_pred):.4f}")
print(f"Recall:    {recall_score(y_train_5, y_train_pred):.4f}")
print(f"F1-Score:  {f1_score(y_train_5, y_train_pred):.4f}")

# Random Forest (schneller mit weniger Bäumen)
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba", n_jobs=-1)
y_train_pred_forest = (y_probas_forest[:, 1] >= 0.5)

print(f"\nRandom Forest F1: {f1_score(y_train_5, y_train_pred_forest):.4f}")

# ============================================================================
# 2. MULTICLASS CLASSIFICATION: Alle 10 Ziffern
# ============================================================================
print("\n" + "=" * 60)
print("MULTICLASS CLASSIFICATION")
print("=" * 60)

# Scaling für bessere Performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(float))
X_test_scaled = scaler.transform(X_test.astype(float))

# SGD mit max_iter
sgd_clf = SGDClassifier(max_iter=50, tol=None, random_state=42, n_jobs=-1)
sgd_clf.fit(X_train_scaled, y_train)
print(f"SGD Accuracy: {cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean():.4f}")

# SVM nur auf kleinem Subset (sehr langsam)
print("\nTrainiere SVM (nur 2000 Samples)...")
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train_scaled[:2000], y_train[:2000])
print(f"SVM Prediction: {svm_clf.predict([X_test_scaled[0]])[0]} (Label: {y_test[0]})")

# ============================================================================
# 3. ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3, n_jobs=-1)
cm = confusion_matrix(y_train, y_train_pred)
print(f"\nHäufigste Verwechslungen:")
print(f"3 als 5: {cm[3, 5]} mal")
print(f"5 als 3: {cm[5, 3]} mal")
print(f"8 als 3: {cm[8, 3]} mal")

# ============================================================================
# 4. MULTILABEL CLASSIFICATION
# ============================================================================
print("\n" + "=" * 60)
print("MULTILABEL CLASSIFICATION")
print("=" * 60)

# Zwei Labels: "groß (>=7)" und "ungerade"
y_train_large = (y_train.astype('int8') >= 7)
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

# KNN auf kleinem Subset (sehr langsam auf vollem Dataset)
print("Trainiere KNN (nur 1000 Samples)...")
knn_clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn_clf.fit(X_train[:1000], y_multilabel[:1000])
print(f"Prediction für Test[0]: {knn_clf.predict([X_test[0]])}")
print(f"Bedeutung: [groß>=7, ungerade]")

# ============================================================================
# 5. MULTIOUTPUT CLASSIFICATION (Noise Removal)
# ============================================================================
print("\n" + "=" * 60)
print("MULTIOUTPUT CLASSIFICATION (Denoising)")
print("=" * 60)

# Noise hinzufügen (nur auf kleinem Subset)
rng = np.random.default_rng(seed=42)
noise_train = rng.integers(0, 100, (1000, 784))
X_train_noisy = X_train[:1000] + noise_train
y_train_clean = X_train[:1000]

noise_test = rng.integers(0, 100, (len(X_test), 784))
X_test_noisy = X_test + noise_test

print("Trainiere Denoising KNN (1000 Samples)...")
knn_clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn_clf.fit(X_train_noisy, y_train_clean)
clean_digit = knn_clf.predict([X_test_noisy[0]])
print(f"Noise Removal erfolgreich (Output Shape: {clean_digit.shape})")

print("\n" + "=" * 60)
print("FERTIG!")
print("=" * 60)