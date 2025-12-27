import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ============================================================================
# 1. DECISION TREE CLASSIFICATION - Iris Dataset
# ============================================================================
print("=" * 60)
print("DECISION TREE CLASSIFICATION")
print("=" * 60)

# Iris Dataset laden
iris = load_iris(as_frame=True)
X_iris = iris.data[['petal length (cm)', 'petal width (cm)']].values
y_iris = iris.target

# Decision Tree trainieren
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

# Vorhersage für neue Blume
test_flower = [[5, 1.5]]  # Length=5cm, Width=1.5cm
prediction = tree_clf.predict(test_flower)[0]
probabilities = tree_clf.predict_proba(test_flower)[0]

print(f"\nTest Flower (Length=5cm, Width=1.5cm):")
print(f"Prediction: {iris.target_names[prediction]}")
print(f"Probabilities: {probabilities.round(3)}")

# Tree Struktur
print(f"\nTree Info:")
print(f"Depth: {tree_clf.tree_.max_depth}")
print(f"Nodes: {tree_clf.tree_.node_count}")
print(f"Leaves: {tree_clf.tree_.n_leaves}")

# ============================================================================
# 2. OVERFITTING VERMEIDEN - Regularization Parameter
# ============================================================================
print("\n" + "=" * 60)
print("REGULARIZATION (Overfitting vermeiden)")
print("=" * 60)

# Make Moons Dataset (nicht-linear)
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

# Ohne Regularization (Overfitting)
tree_overfit = DecisionTreeClassifier(random_state=42)
tree_overfit.fit(X_moons, y_moons)

# Mit Regularization (min_samples_leaf)
tree_regularized = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_regularized.fit(X_moons, y_moons)

# Test auf neuem Dataset
X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)

print(f"\nOhne Regularization:")
print(f"  Train Accuracy: {tree_overfit.score(X_moons, y_moons):.4f}")
print(f"  Test Accuracy:  {tree_overfit.score(X_moons_test, y_moons_test):.4f}")
print(f"  Nodes: {tree_overfit.tree_.node_count}")

print(f"\nMit Regularization (min_samples_leaf=5):")
print(f"  Train Accuracy: {tree_regularized.score(X_moons, y_moons):.4f}")
print(f"  Test Accuracy:  {tree_regularized.score(X_moons_test, y_moons_test):.4f}")
print(f"  Nodes: {tree_regularized.tree_.node_count}")

print("\nHinweis: Regularisierter Tree hat bessere Test Accuracy!")

# ============================================================================
# 3. DECISION TREE REGRESSION
# ============================================================================
print("\n" + "=" * 60)
print("DECISION TREE REGRESSION")
print("=" * 60)

# Quadratische Daten generieren: y = x²
rng = np.random.default_rng(seed=42)
X_quad = rng.random((200, 1)) - 0.5
y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))

# Regression Trees mit verschiedenen Depths
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)

tree_reg1.fit(X_quad, y_quad)
tree_reg2.fit(X_quad, y_quad)

# Vorhersagen
test_points = [[0.0], [0.3], [-0.3]]
print(f"\nVorhersagen für verschiedene Depths:")
print(f"{'X':>6s} {'Depth=2':>10s} {'Depth=3':>10s} {'True (≈x²)':>12s}")
for x in test_points:
    pred1 = tree_reg1.predict([x])[0]
    pred2 = tree_reg2.predict([x])[0]
    true_val = x[0] ** 2
    print(f"{x[0]:6.1f} {pred1:10.4f} {pred2:10.4f} {true_val:12.4f}")

print(f"\nTree 1 (depth=2): {tree_reg1.tree_.node_count} nodes")
print(f"Tree 2 (depth=3): {tree_reg2.tree_.node_count} nodes")

# ============================================================================
# 4. REGULARIZATION PARAMETER
# ============================================================================
print("\n" + "=" * 60)
print("WICHTIGE HYPERPARAMETER")
print("=" * 60)

print("""
Regularization Parameter (verhindern Overfitting):

1. max_depth: Maximale Tiefe des Baums
   - Niedrig (2-5): Underfitting, einfaches Modell
   - Hoch (>10): Overfitting, komplexes Modell
   - Default: None (unbegrenzt)

2. min_samples_split: Minimum Samples für Split
   - Hoch (>20): Weniger Splits, einfacher
   - Niedrig (2): Mehr Splits, komplexer
   - Default: 2

3. min_samples_leaf: Minimum Samples pro Blatt
   - Hoch (>10): Glatteres Modell
   - Niedrig (1): Detaillierteres Modell
   - Default: 1

4. max_leaf_nodes: Maximum Anzahl Blätter
   - Begrenzt Komplexität direkt
   - Default: None (unbegrenzt)

5. max_features: Maximum Features pro Split
   - Reduziert Overfitting
   - Erhöht Diversität (wichtig für Random Forests)
   - Default: None (alle Features)""")

# ============================================================================
# 5. PRAKTISCHES BEISPIEL - Verschiedene Konfigurationen
# ============================================================================
print("=" * 60)
print("VERGLEICH VERSCHIEDENER KONFIGURATIONEN")
print("=" * 60)

configs = [
    ("Default (Overfitting)", {}),
    ("max_depth=3", {"max_depth": 3}),
    ("min_samples_leaf=10", {"min_samples_leaf": 10}),
    ("Kombiniert", {"max_depth": 5, "min_samples_leaf": 5}),
]

for name, params in configs:
    tree = DecisionTreeClassifier(random_state=42, **params)
    tree.fit(X_moons, y_moons)
    train_acc = tree.score(X_moons, y_moons)
    test_acc = tree.score(X_moons_test, y_moons_test)
    
    print(f"\n{name}:")
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | Nodes: {tree.tree_.node_count}")

# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
print("""
Decision Trees:
✅ Einfach zu verstehen und interpretieren
✅ Wenig Data Preprocessing nötig (kein Scaling)
✅ Funktioniert mit numerischen und kategorischen Daten
✅ Kann nicht-lineare Beziehungen lernen

❌ Anfällig für Overfitting (ohne Regularization)
❌ Instabil (kleine Datenänderung → anderer Baum)
❌ Nicht gut für lineare Beziehungen
❌ Bias zu Features mit vielen Werten

Best Practices:
🎯 Immer max_depth oder min_samples_leaf setzen
🎯 Cross-Validation für Hyperparameter-Tuning
🎯 Für Produktion: Random Forests statt einzelner Tree""")

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
import numpy as np
from scipy.stats import mode

X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.2, random_state=42)
params = {
    'max_leaf_nodes': list(range(2, 100)),
    'max_depth': [1, 2, 3, 4, 5, 6], 
    'min_samples_split': [2, 3, 4]
}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=3)
grid_search_cv.fit(X_train, y_train)
print(grid_search_cv.best_estimator_)
y_pred = grid_search_cv.predict(X_test)
print(accuracy_score(y_test, y_pred))

n_trees =  1000
n_instances = 100
mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]
accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
np.mean(accuracy_scores)

Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))