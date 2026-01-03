import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ============================================================================
# 1. PCA - Principal Component Analysis
# ============================================================================
print("="*60)
print("PCA - PRINCIPAL COMPONENT ANALYSIS")
print("="*60)

# MNIST Dataset laden
mnist = fetch_openml("mnist_784", as_frame=False, parser='auto')

X_train, y_train = mnist.data[:60000], mnist.target[:60000]
X_test, y_test = mnist.data[60000:], mnist.target[60000:]

# PCA: Finde Dimensionen für 95% Varianz
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(f"\nDimensionen für 95% Varianz: {d} (von 784)")

# Direkt mit n_components=0.95
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(f"Komponenten: {pca.n_components_}")
print(f"Erklärte Varianz: {pca.explained_variance_ratio_.sum():.4f}")

# Rekonstruktion
X_recovered = pca.inverse_transform(X_reduced)
print(f"Original Shape: {X_train.shape}")
print(f"Reduziert: {X_reduced.shape}")
print(f"Rekonstruiert: {X_recovered.shape}")

# ============================================================================
# 2. INCREMENTAL PCA - Für große Datasets
# ============================================================================
print("\n" + "="*60)
print("INCREMENTAL PCA")
print("="*60)

# Batch-weise Training
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced_inc = inc_pca.transform(X_train)
print(f"\nIncremental PCA Shape: {X_reduced_inc.shape}")
print("Vorteil: Passt in Memory bei großen Datasets")

# ============================================================================
# 3. RANDOM PROJECTION - Schnelle Dimensionsreduktion
# ============================================================================
print("\n" + "="*60)
print("RANDOM PROJECTION")
print("="*60)

m, ε = 5_000, 0.1
n = 20000
rng = np.random.default_rng(seed=42)
X = rng.standard_normal((m, n))

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced_rp = gaussian_rnd_proj.fit_transform(X)
print(f"\nOriginal: {X.shape}")
print(f"Reduziert: {X_reduced_rp.shape}")
print(f"Epsilon (Fehlertoleranz): {ε}")
print("Vorteil: Sehr schnell, keine Fit-Zeit")

# ============================================================================
# 4. LLE - Locally Linear Embedding
# ============================================================================
print("\n" + "="*60)
print("LLE - LOCALLY LINEAR EMBEDDING")
print("="*60)

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)

print(f"\nSwiss Roll: {X_swiss.shape}")
print(f"Entrollt: {X_unrolled.shape}")
print("LLE: Erhält lokale Struktur (Manifold Learning)")

# ============================================================================
# 5. ANDERE MANIFOLD LEARNING METHODEN
# ============================================================================
print("\n" + "="*60)
print("MANIFOLD LEARNING VERGLEICH")
print("="*60)

# MDS - Multidimensional Scaling
mds = MDS(n_components=2, normalized_stress=False, random_state=42)
X_mds = mds.fit_transform(X_swiss)
print(f"\nMDS: {X_mds.shape}")

# Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X_swiss)
print(f"Isomap: {X_isomap.shape}")

# t-SNE
tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
X_tsne = tsne.fit_transform(X_swiss)
print(f"t-SNE: {X_tsne.shape}")

# ============================================================================
# 6. KERNEL PCA - Nichtlineare Dimensionsreduktion
# ============================================================================
print("\n" + "="*60)
print("KERNEL PCA")
print("="*60)

# RBF Kernel
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, random_state=42)
X_kpca = rbf_pca.fit_transform(X_swiss)
print(f"\nKernel PCA (RBF): {X_kpca.shape}")
print("Kernel: Kann nichtlineare Strukturen erfassen")

# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================
print("\n" + "="*60)
print("ZUSAMMENFASSUNG")
print("="*60)

print("""
Dimensionsreduktion:

1. PCA (Principal Component Analysis)
   - Linear, schnell, interpretierbar
   - Maximiert Varianz
   - Gut für: Kompression, Noise Reduction

2. Incremental PCA
   - Wie PCA, aber batch-weise
   - Gut für: Große Datasets (nicht in Memory)

3. Random Projection
   - Sehr schnell, keine Fit-Zeit
   - Johnson-Lindenstrauss Lemma
   - Gut für: Schnelle Approximation

4. LLE (Locally Linear Embedding)
   - Manifold Learning
   - Erhält lokale Struktur
   - Gut für: Nichtlineare Manifolds

5. MDS, Isomap, t-SNE
   - MDS: Erhält Distanzen
   - Isomap: Geodätische Distanzen
   - t-SNE: Visualisierung, Cluster

6. Kernel PCA
   - Nichtlinear durch Kernel Trick
   - RBF, Polynomial, Sigmoid
   - Gut für: Komplexe nichtlineare Strukturen

Wann was verwenden?
✅ Schnell & Linear → PCA
✅ Große Daten → Incremental PCA
✅ Sehr schnell → Random Projection
✅ Manifolds → LLE, Isomap
✅ Visualisierung → t-SNE
✅ Nichtlinear → Kernel PCA
""")


### Exercises:
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Original Daten
start = time.time()
rnd_clf.fit(X_train, y_train)
time_original = time.time() - start

print(time_original)
y_pred = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# PCA-reduzierte Daten
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

rnd_clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
start = time.time()
rnd_clf_pca.fit(X_train_reduced, y_train)
time_pca = time.time() - start
print(time_pca)

y_pred = rnd_clf_pca.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred))

sgd_clf = SGDClassifier(random_state=42)
start = time.time()
sgd_clf.fit(X_train, y_train)
time_sgd = time.time() - start
print(time_sgd)

y_pred = sgd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

sgd_clf_pca = SGDClassifier(random_state=42)
start = time.time()
sgd_clf_pca.fit(X_train_reduced, y_train)
time_sgd_pca = time.time() - start
print(time_sgd_pca)
y_pred = sgd_clf_pca.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred))

X_sample, y_sample = X_train[:5000], y_train[:5000]
tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)

start = time.time()
X_reduced = tsne.fit_transform(X_sample)
time_tsne = time.time() - start
print(time_tsne)

plt.figure(figsize=(13, 10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            c=y_sample.astype(np.int8), cmap="jet", alpha=0.5)
plt.axis('off')
plt.colorbar()
plt.show()

plt.figure(figsize=(9, 9))
cmap = plt.cm.jet
for digit in ('4', '9'):
    plt.scatter(X_reduced[y_sample == digit, 0], X_reduced[y_sample == digit, 1],
                c=[cmap(float(digit) / 9)], alpha=0.5)
plt.axis('off')
plt.show()

idx = (y_sample == '4') | (y_sample == '9')
X_subset = X_sample[idx]
y_subset = y_sample[idx]

tsne_subset = TSNE(n_components=2, init="random", learning_rate="auto",
                   random_state=42)
X_subset_reduced = tsne_subset.fit_transform(X_subset)

plt.figure(figsize=(9, 9))
for digit in ('4', '9'):
    plt.scatter(X_subset_reduced[y_subset == digit, 0],
                X_subset_reduced[y_subset == digit, 1],
                c=[cmap(float(digit) / 9)], alpha=0.5)
plt.axis('off')
plt.show()

def plot_digits(X, y, min_distance=0.04, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = plt.cm.jet
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1],
                    c=[cmap(float(digit) / 9)], alpha=0.5)
    plt.axis("off")
    ax = plt.gca()  # get current axes
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(neighbors - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(float(y[index]) / 9),
                         fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"),
                                          image_coord)
                ax.add_artist(imagebox)
plot_digits(X_reduced, y_sample)
plot_digits(X_reduced, y_sample, images=X_sample, figsize=(35, 25))
plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))

pca = PCA(n_components=2, random_state=42)
start = time.time()
X_pca_reduced = pca.fit_transform(X_sample)
time_pca = time.time() - start
print(time_pca)
plot_digits(X_pca_reduced, y_sample)
plt.show()

lle = LocallyLinearEmbedding(n_components=2, random_state=42)
start = time.time()
X_lle_reduced = lle.fit_transform(X_sample)
time_lle = time.time() - start
print(time_lle)
plot_digits(X_lle_reduced, y_sample)
plt.show()

pca_lle = make_pipeline(
    PCA(n_components=0.95),
    LocallyLinearEmbedding(n_components=2, random_state=42)
)
start = time.time()
X_pca_lle_reduced = pca_lle.fit_transform(X_sample)
time_pca_lle = time.time() - start
print(time_pca_lle)
plot_digits(X_pca_lle_reduced, y_sample)
plt.show()

lda = LinearDiscriminantAnalysis(n_components=2)
start = time.time()
X_lda_reduced = lda.fit_transform(X_sample, y_sample)
time_lda = time.time() - start
print(time_lda)
plot_digits(X_lda_reduced, y_sample, figsize=(12, 12))
plt.show()

