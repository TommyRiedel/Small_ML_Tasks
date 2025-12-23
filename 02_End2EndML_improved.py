# VERBESSERUNGEN:
# 1. Imports konsolidiert (keine Duplikate mehr)
# 2. Unnötige Zwischenschritte entfernt
# 3. Hauptproblem am Ende behoben: .keys() gibt dict_keys zurück, kein Index möglich
# 4. Code in logische Sektionen unterteilt

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32

# Sklearn imports konsolidiert
from sklearn.model_selection import (train_test_split, StratifiedShuffleSplit, 
                                      cross_val_score, GridSearchCV, RandomizedSearchCV)
from sklearn.svm import SVR
from scipy.stats import loguniform, uniform
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder, MinMaxScaler, 
                                   StandardScaler, FunctionTransformer)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import binom
from pandas.plotting import scatter_matrix

np.random.seed(42)

# ============================================================================
# 1. DATEN LADEN
# ============================================================================
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# ============================================================================
# 2. TRAIN/TEST SPLIT - Nur die beste Methode verwenden
# VERBESSERUNG: Alte manuelle Split-Funktionen entfernt, nur stratified split behalten
# ============================================================================
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

# Income category wieder entfernen
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# ============================================================================
# 3. EXPLORATIVE DATENANALYSE (optional, auskommentiert für Performance)
# ============================================================================
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, grid=True,
#              s=housing["population"]/100, label="population",
#              c="median_house_value", cmap="jet", colorbar=True, legend=True)
# plt.show()

# corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# ============================================================================
# 4. DATEN VORBEREITEN
# ============================================================================
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# ============================================================================
# 5. CUSTOM TRANSFORMERS
# ============================================================================
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Berechnet RBF-Ähnlichkeit zu Cluster-Zentren"""
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):  # BUGFIX: Typo korrigiert
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]

# ============================================================================
# 6. PREPROCESSING PIPELINE
# VERBESSERUNG: Direkt die finale Pipeline definieren, Zwischenschritte entfernt
# ============================================================================
def column_ratio(X):
    """Berechnet Verhältnis zwischen zwei Spalten"""
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
], remainder=default_num_pipeline)

# ============================================================================
# 7. MODELLE TRAINIEREN UND EVALUIEREN
# VERBESSERUNG: Nur relevante Modelle, Cross-Validation für bessere Schätzung
# ============================================================================
print("=" * 60)
print("MODELL EVALUATION MIT CROSS-VALIDATION")
print("=" * 60)

# Linear Regression
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_rmses = -cross_val_score(lin_reg, housing, housing_labels, 
                              scoring="neg_root_mean_squared_error", cv=10)
print("\nLinear Regression:")
print(pd.Series(lin_rmses).describe())

# Decision Tree
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, 
                               scoring="neg_root_mean_squared_error", cv=10)
print("\nDecision Tree:")
print(pd.Series(tree_rmses).describe())

# Random Forest
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, 
                                 scoring="neg_root_mean_squared_error", cv=10)
print("\nRandom Forest:")
print(pd.Series(forest_rmses).describe())

# ============================================================================
# 8. HYPERPARAMETER TUNING
# VERBESSERUNG: Kleineres Grid für schnellere Ausführung
# HAUPTPROBLEM BEHOBEN: .keys() gibt dict_keys zurück, nicht indexierbar!
# ============================================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (Grid Search)")
print("=" * 60)

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

# VERBESSERUNG: Kleineres Grid für schnellere Ausführung
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 10],  # Reduziert von [5, 8, 10]
     'random_forest__max_features': [4, 6]},     # Reduziert von [4, 6, 8]
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, 
                          scoring='neg_root_mean_squared_error',
                          verbose=2)  # Verbose für Progress-Anzeige
grid_search.fit(housing, housing_labels)

print("\nBeste Parameter:")
print(grid_search.best_params_)
print(f"\nBester Score: {-grid_search.best_score_:.2f}")

# BUGFIX: Das war das Problem! .keys() gibt dict_keys zurück, nicht indexierbar
# Alte Zeile (FEHLER): print(str(full_pipeline.get_params().keys()[1000:] + "..."))
# Neue Zeile (KORREKT):
params_list = list(full_pipeline.get_params().keys())
print(f"\nAnzahl verfügbarer Parameter: {len(params_list)}")
print("Erste 10 Parameter:", params_list[:10])

# ============================================================================
# 9. FINALES MODELL AUF TEST SET
# ============================================================================
print("\n" + "=" * 60)
print("FINALE EVALUATION AUF TEST SET")
print("=" * 60)

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = grid_search.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
print(f"\nFinaler Test RMSE: {final_rmse:.2f}")

print("\n" + "=" * 60)
print("FERTIG!")
print("=" * 60)

# ============================================================================
# 10. SUPPORT VECTOR REGRESSION (SVR)
# Nur auf ersten 5000 Samples wegen Performance, 3-fold CV
# ============================================================================
print("\n" + "=" * 60)
print("SVR HYPERPARAMETER TUNING (auf 5000 Samples)")
print("=" * 60)

# Nur erste 5000 Samples für SVR (skaliert schlecht)
housing_small = housing.iloc[:5000]
housing_labels_small = housing_labels.iloc[:5000]

svr_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("svr", SVR()),
])

# Hyperparameter Grid für SVR
param_distributions = [
    {
        'svr__kernel': ['linear'],
        'svr__C': loguniform(20, 200_000),  # Log-uniform zwischen 20 und 200k
    },
    {
        'svr__kernel': ['rbf'],
        'svr__C': loguniform(20, 200_000),
        'svr__gamma': loguniform(0.001, 0.1),  # Gamma für RBF kernel
    },
]

# RandomizedSearchCV statt GridSearchCV (schneller)
random_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions,
    n_iter=10,  # Nur 10 zufällige Kombinationen testen
    cv=3,  # 3-fold CV wie empfohlen
    scoring='neg_root_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1  # Parallelisierung
)

print("\nTraining SVR (kann einige Minuten dauern)...")
random_search.fit(housing_small, housing_labels_small)

print("\nBeste SVR Parameter:")
print(random_search.best_params_)
print(f"\nBester SVR Score (auf 5000 Samples): {-random_search.best_score_:.2f}")

# Test auf vollem Test Set
print("\nEvaluierung auf vollem Test Set...")
svr_test_pred = random_search.predict(X_test)
svr_test_rmse = root_mean_squared_error(y_test, svr_test_pred)
print(f"SVR Test RMSE: {svr_test_rmse:.2f}")

# Vergleich mit Random Forest
print("\n" + "=" * 60)
print("VERGLEICH: SVR vs Random Forest")
print("=" * 60)
print(f"Random Forest Test RMSE: {final_rmse:.2f}")
print(f"SVR Test RMSE:           {svr_test_rmse:.2f}")
print(f"Differenz:               {svr_test_rmse - final_rmse:.2f}")

if svr_test_rmse < final_rmse:
    print("\n✓ SVR performt besser!")
else:
    print("\n✗ Random Forest performt besser.")

print("\n" + "=" * 60)
print("ALLE AUFGABEN ABGESCHLOSSEN!")
print("=" * 60)
