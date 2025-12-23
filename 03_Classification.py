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


from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# ============================================================================
# AUFGABE 1: KNN Classifier mit >97% Accuracy
# ============================================================================
print("=" * 60)
print("AUFGABE 1: KNN mit GridSearch für >97% Accuracy")
print("=" * 60)

# Daten laden
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist.data, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# GridSearch für beste Hyperparameter
print("\nGridSearch läuft (kann einige Minuten dauern)...")
param_grid = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': [3, 4, 5]
}

knn_clf = KNeighborsClassifier(n_jobs=-1)
grid_search = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"\nBeste Parameter: {grid_search.best_params_}")
print(f"Beste CV Accuracy: {grid_search.best_score_:.4f}")

# Test auf Test Set
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

if test_accuracy > 0.97:
    print("✓ Ziel erreicht: >97% Accuracy!")
else:
    print(f"✗ Noch {0.97 - test_accuracy:.4f} bis zum Ziel")

# ============================================================================
# AUFGABE 2: Data Augmentation durch Shifting
# ============================================================================
print("\n" + "=" * 60)
print("AUFGABE 2: Data Augmentation (Image Shifting)")
print("=" * 60)

def shift_image(image, dx, dy):
    # Verschiebt ein 28x28 Bild um dx, dy Pixel
    image = image.reshape((28, 28))
    shifted = np.roll(image, dy, axis=0)
    shifted = np.roll(shifted, dx, axis=1)
    
    # Ränder auf 0 setzen
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    
    return shifted.reshape(-1)

# Augmentierte Daten erstellen (nur auf kleinem Subset für Demo)
print("\nErstelle augmentierte Daten (nur 5000 Samples für Demo)...")
X_train_subset = X_train[:5000]
y_train_subset = y_train[:5000]

X_train_augmented = [X_train_subset]
y_train_augmented = [y_train_subset]

# 4 Richtungen: links, rechts, oben, unten
shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
for dx, dy in shifts:
    X_shifted = np.array([shift_image(img, dx, dy) for img in X_train_subset])
    X_train_augmented.append(X_shifted)
    y_train_augmented.append(y_train_subset)

X_train_augmented = np.vstack(X_train_augmented)
y_train_augmented = np.hstack(y_train_augmented)

print(f"Original: {X_train_subset.shape}")
print(f"Augmentiert: {X_train_augmented.shape}")

# Training mit augmentierten Daten
print("\nTrainiere KNN mit augmentierten Daten...")
knn_aug = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
knn_aug.fit(X_train_augmented, y_train_augmented)

y_pred_aug = knn_aug.predict(X_test)
test_accuracy_aug = accuracy_score(y_test, y_pred_aug)
print(f"Test Accuracy mit Augmentation: {test_accuracy_aug:.4f}")
print(f"Verbesserung: {test_accuracy_aug - test_accuracy:.4f}")

# ============================================================================
# AUFGABE 3: Titanic Dataset
# ============================================================================
print("\n" + "=" * 60)
print("AUFGABE 3: Titanic Survival Prediction")
print("=" * 60)

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets", filter="data")
    return [pd.read_csv(Path("datasets/titanic") / filename) for filename in ("train.csv", "test.csv")]

train_data, test_data = load_titanic_data()

# Features auswählen
num_features = ['Age', 'Fare', 'SibSp', 'Parch']
cat_features = ['Pclass', 'Sex', 'Embarked']

# Pipeline erstellen
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
])

# Modell trainieren
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42)),
])

X_train = train_data[num_features + cat_features]
y_train = train_data['Survived']

model.fit(X_train, y_train)
model_accuracy = model.score(X_train, y_train)
print(f"\nTrainingsaccuracy auf Titanic-Daten: {model_accuracy:.4f}")
predictions = model.predict(test_data[num_features + cat_features])
print(f"Vorhersagen für Testdaten (erste 10): {predictions[:10]}")