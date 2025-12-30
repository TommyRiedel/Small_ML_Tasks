from sklearn.datasets import make_moons, load_iris
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, 
                              RandomForestClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. VOTING CLASSIFIER - Kombiniert verschiedene Modelle
# ============================================================================
print("VOTING CLASSIFIER")
print("=" * 50)

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ],
    voting='hard'  # Mehrheitsentscheidung
)
voting_clf.fit(X_train, y_train)

for name, clf in voting_clf.named_estimators_.items():
    print(f"{name}: {clf.score(X_test, y_test):.3f}")
print(f"Voting: {voting_clf.score(X_test, y_test):.3f} ← Ensemble!")

# ============================================================================
# 2. BAGGING - Bootstrap Aggregating
# ============================================================================
print("\nBAGGING")
print("=" * 50)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    oob_score=True,  # Out-of-Bag Score
    n_jobs=-1,
    random_state=42
)
bag_clf.fit(X_train, y_train)

print(f"Einzelner Tree: {tree_clf.score(X_test, y_test):.3f}")
print(f"Bagging (500):  {bag_clf.score(X_test, y_test):.3f}")
print(f"OOB Score:      {bag_clf.oob_score_:.3f}")

# ============================================================================
# 3. RANDOM FOREST - Bagging + Random Features
# ============================================================================
print("\nRANDOM FOREST")
print("=" * 50)

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
print(f"Accuracy: {rf_clf.score(X_test, y_test):.3f}")

# Feature Importances
iris = load_iris(as_frame=True)
rf_iris = RandomForestClassifier(n_estimators=100, random_state=42)
rf_iris.fit(iris.data, iris.target)

print("\nFeature Importances (Iris):")
for name, score in zip(iris.data.columns, rf_iris.feature_importances_):
    print(f"  {name}: {score:.3f}")

# ============================================================================
# 4. ADABOOST - Adaptive Boosting
# ============================================================================
print("\nADABOOST")
print("=" * 50)

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
print(f"Accuracy: {ada_clf.score(X_test, y_test):.3f}")

print("""

ZUSAMMENFASSUNG:
- Voting: Kombiniert verschiedene Modelle
- Bagging: Reduziert Overfitting durch Bootstrap
- Random Forest: Beste Allround-Wahl
- AdaBoost: Fokussiert auf schwierige Samples
""")
