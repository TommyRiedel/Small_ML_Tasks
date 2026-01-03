import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, 
                              RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. VOTING CLASSIFIER - Kombiniert verschiedene Modelle
# ============================================================================
print("=" * 60)
print("VOTING CLASSIFIER")
print("=" * 60)

# Daten
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Einzelne Classifiers
log_clf = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# Hard Voting: Mehrheitsentscheidung
voting_hard = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('svc', svm_clf)],
    voting='hard'  # Mehrheit gewinnt
)
voting_hard.fit(X_train, y_train)

print("\nHard Voting (Mehrheitsentscheidung):")
for name, clf in voting_hard.named_estimators_.items():
    clf.fit(X_train, y_train)
    print(f"  {name}: {clf.score(X_test, y_test):.4f}")
print(f"  Voting: {voting_hard.score(X_test, y_test):.4f} ← Oft besser!")

# Soft Voting: Durchschnitt der Wahrscheinlichkeiten
voting_soft = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('svc', svm_clf)],
    voting='soft'  # Durchschnitt der Probabilities
)
voting_soft.fit(X_train, y_train)

print(f"\nSoft Voting (Probability Average): {voting_soft.score(X_test, y_test):.4f}")

# ============================================================================
# 2. BAGGING - Bootstrap Aggregating
# ============================================================================
print("\n" + "=" * 60)
print("BAGGING (Bootstrap Aggregating)")
print("=" * 60)

# Einzelner Decision Tree (Overfitting)
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Bagging: Viele Trees auf verschiedenen Subsets
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,      # 500 Trees
    max_samples=100,       # Jeder Tree: 100 Samples
    n_jobs=-1,             # Parallelisierung
    random_state=42
)
bag_clf.fit(X_train, y_train)

print(f"\nEinzelner Tree: {tree_clf.score(X_test, y_test):.4f}")
print(f"Bagging (500 Trees): {bag_clf.score(X_test, y_test):.4f} ← Besser!")

# Out-of-Bag Evaluation (kostenlose Validation)
bag_clf_oob = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    oob_score=True,  # Nutzt nicht-verwendete Samples für Validation
    n_jobs=-1,
    random_state=42
)
bag_clf_oob.fit(X_train, y_train)

print(f"\nOOB Score: {bag_clf_oob.oob_score_:.4f}")
print(f"Test Score: {bag_clf_oob.score(X_test, y_test):.4f}")
print("Hinweis: OOB Score ≈ Test Score (ohne extra Validation Set!)")

# ============================================================================
# 3. RANDOM FOREST - Bagging + Random Feature Selection
# ============================================================================
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

# Random Forest = Bagging + max_features
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,  # Regularization
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_train)

print(f"\nRandom Forest: {rf_clf.score(X_test, y_test):.4f}")

# Feature Importance
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
rf_iris = RandomForestClassifier(n_estimators=500, random_state=42)
rf_iris.fit(iris.data, iris.target)

print("\nFeature Importances (Iris):")
for score, name in zip(rf_iris.feature_importances_, iris.data.columns):
    print(f"  {name:20s}: {score:.3f}")

# ============================================================================
# 4. ADABOOST - Adaptive Boosting
# ============================================================================
print("\n" + "=" * 60)
print("ADABOOST (Adaptive Boosting)")
print("=" * 60)

# AdaBoost: Fokussiert auf schwierige Samples
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # Weak learners (Stumps)
    n_estimators=30,
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)

print(f"\nAdaBoost: {ada_clf.score(X_test, y_test):.4f}")
print("Prinzip: Jeder Tree fokussiert auf Fehler des vorherigen")

# ============================================================================
# 5. GRADIENT BOOSTING - Für Regression
# ============================================================================
print("\n" + "=" * 60)
print("GRADIENT BOOSTING (Regression)")
print("=" * 60)

# Daten: y = 3x² + noise
m = 100
rng = np.random.default_rng(seed=42)
X_reg = rng.random((m, 1)) - 0.5
y_reg = 3 * X_reg[:, 0] ** 2 + 0.05 * rng.standard_normal(m)

# Manuelles Gradient Boosting (zum Verständnis)
tree1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree1.fit(X_reg, y_reg)

# Tree 2: Lernt Residuals von Tree 1
y2 = y_reg - tree1.predict(X_reg)
tree2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree2.fit(X_reg, y2)

# Tree 3: Lernt Residuals von Tree 1 + Tree 2
y3 = y2 - tree2.predict(X_reg)
tree3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree3.fit(X_reg, y3)

# Finale Vorhersage: Summe aller Trees
X_new = np.array([[-0.4], [0.], [0.5]])
y_pred = sum(tree.predict(X_new) for tree in (tree1, tree2, tree3))
print(f"\nManuelle Gradient Boosting Predictions: {y_pred}")

# Scikit-Learn Gradient Boosting
gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0,
    random_state=42
)
gbrt.fit(X_reg, y_reg)
print(f"Scikit-Learn Predictions: {gbrt.predict(X_new)}")

# Mit Early Stopping
gbrt_best = GradientBoostingRegressor(
    max_depth=2,
    learning_rate=0.05,
    n_estimators=500,
    n_iter_no_change=10,  # Early stopping
    random_state=42
)
gbrt_best.fit(X_reg, y_reg)
print(f"\nMit Early Stopping: {gbrt_best.n_estimators_} Trees verwendet (von 500)")

# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)

print("""
Ensemble Methods:

1. VOTING
   - Kombiniert verschiedene Modelle
   - Hard: Mehrheitsentscheidung
   - Soft: Durchschnitt der Wahrscheinlichkeiten
   - Gut wenn Modelle unterschiedlich sind

2. BAGGING (Bootstrap Aggregating)
   - Trainiert gleiche Modelle auf verschiedenen Subsets
   - Reduziert Variance (Overfitting)
   - OOB Score: Kostenlose Validation
   - Parallelisierbar

3. RANDOM FOREST
   - Bagging + Random Feature Selection
   - Jeder Split: Nur Subset der Features
   - Mehr Diversität → Bessere Performance
   - Feature Importances verfügbar

4. BOOSTING
   - Sequentiell: Jeder Learner korrigiert vorherigen
   - AdaBoost: Fokussiert auf schwierige Samples
   - Gradient Boosting: Lernt Residuals
   - Reduziert Bias (Underfitting)
   - NICHT parallelisierbar

Wann was verwenden?
✅ Overfitting → Bagging/Random Forest
✅ Underfitting → Boosting (AdaBoost/Gradient Boosting)
✅ Verschiedene Modelle → Voting
✅ Schnell & gut → Random Forest (Default-Wahl)
✅ Beste Performance → Gradient Boosting (aber langsamer)
""")

### Exercises:
from sklearn.datasets import fetch_openml

X_mnist, y_mnist = fetch_openml('mnist_784', return_X_y=True, as_frame=False,
                                parser='auto')

X_train, y_train = X_mnist[:50000], y_mnist[:50000]
X_valid, y_valid = X_mnist[50000:60000], y_mnist[50000:60000]
X_test, y_test = X_mnist[60000:], y_mnist[60000:]

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, 
                              RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingRegressor)
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, dual=True, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print(f"Training the", estimator)
    estimator.fit(X_train, y_train)

print(estimator.score(X_valid, y_valid) for estimator in estimators)
from sklearn.ensemble import VotingClassifier
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]
voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_valid, y_valid))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_valid_encoded = encoder.fit_transform(y_valid)
y_valid_encoded = y_valid.astype(np.int64)

print([estimator.score(X_valid, y_valid_encoded) for estimator in voting_clf.estimators_])
voting_clf.set_params(svm_clf="drop")
print(voting_clf.estimators)
print(voting_clf.estimators_)
voting_clf.named_estimators_
svm_clf_trained = voting_clf.named_estimators_.pop("svm_clf")
voting_clf.estimators_.remove(svm_clf_trained)
print(voting_clf.score(X_valid, y_valid))
voting_clf.voting = "soft"
print(voting_clf.score(X_valid, y_valid))
voting_clf.voting = "hard"
print(voting_clf.score(X_test, y_test))

print([estimator.score(X_test, y_test.astype(np.int64)) for estimator in voting_clf.estimators_])

X_valid_predictions = np.empty((len(X_valid), len(estimators)), dtype=object)
for index, estimator in enumerate(estimators):
    X_valid_predictions[:, index] = estimator.predict(X_valid)

print(X_valid_predictions)
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_valid_predictions, y_valid)
print(rnd_forest_blender.oob_score_)
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=object)
for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
y_pred = rnd_forest_blender.predict(X_test_predictions)
print(accuracy_score(y_test, y_pred))

X_train_full, y_train_full = X_mnist[:60000], y_mnist[:60000]

stack_clf = StackingClassifier(named_estimators, final_estimator=rnd_forest_blender)
stack_clf.fit(X_train_full, y_train_full)
print(stack_clf.score(X_test, y_test))