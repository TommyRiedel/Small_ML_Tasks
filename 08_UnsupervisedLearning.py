import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN, 
                             SpectralClustering, AgglomerativeClustering)
from sklearn.datasets import make_blobs, fetch_openml, load_digits, make_moons
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from pathlib import Path
import urllib.request
import PIL

# ============================================================================
# 1. K-MEANS CLUSTERING - Basic Example
# ============================================================================
print("="*60)
print("K-MEANS CLUSTERING")
print("="*60)

# Generate blob data with 5 clusters
blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

# Fit K-Means with k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

print(f"\nCluster Centers:\n{kmeans.cluster_centers_}")
print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")

# Predict new points
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
print(f"\nPredictions for new points: {kmeans.predict(X_new)}")
print(f"Distances to centroids:\n{kmeans.transform(X_new).round(2)}")

# ============================================================================
# 2. FINDING OPTIMAL K - Elbow Method & Silhouette Score
# ============================================================================
print("\n" + "="*60)
print("FINDING OPTIMAL K")
print("="*60)

# Try different values of k
kmeans_per_k = [KMeans(n_clusters=k, random_state=43).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

print("\nInertia for different k:")
for k, inertia in enumerate(inertias, 1):
    print(f"  k={k}: {inertia:.2f}")

# Silhouette score: measures cluster quality [-1, 1], higher is better
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

print("\nSilhouette scores for different k:")
for k, score in enumerate(silhouette_scores, 2):
    print(f"  k={k}: {score:.4f}")

best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nBest k based on silhouette score: {best_k}")

# ============================================================================
# 3. MINI-BATCH K-MEANS - For Large Datasets
# ============================================================================
print("\n" + "="*60)
print("MINI-BATCH K-MEANS")
print("="*60)

# Mini-Batch K-Means is faster for large datasets
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
print(f"Mini-Batch K-Means inertia: {minibatch_kmeans.inertia_:.2f}")

# Example with MNIST (using memory mapping for large data)
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X_train, y_train = mnist.data[:60000], mnist.target[:60000]

filename = "my_mnist.mmap"
X_memmap = np.memmap(filename, dtype="float32", mode="write", shape=X_train.shape)
X_memmap[:] = X_train
X_memmap.flush()

minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
minibatch_kmeans.fit(X_memmap)
print(f"MNIST Mini-Batch K-Means trained on {len(X_train)} samples")

# ============================================================================
# 4. IMAGE SEGMENTATION - Color Quantization
# ============================================================================
print("\n" + "="*60)
print("IMAGE SEGMENTATION")
print("="*60)

# Download and load ladybug image
homlp_root = "https://github.com/ageron/handson-mlp/raw/main/"
filename = "ladybug.png"
filepath = Path(f"my_{filename}")
if not filepath.is_file():
    print("Downloading", filename)
    url = f"{homlp_root}/images/unsupervised_learning/{filename}"
    urllib.request.urlretrieve(url, filepath)

image = np.asarray(PIL.Image.open(filepath))
print(f"\nImage shape: {image.shape}")

# Reduce colors using K-Means
X_img = image.reshape(-1, 3)
n_colors = [10, 8, 6, 4, 2]

for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_img)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    print(f"Reduced to {n_clusters} colors")

# ============================================================================
# 5. SEMI-SUPERVISED LEARNING - Label Propagation
# ============================================================================
print("\n" + "="*60)
print("SEMI-SUPERVISED LEARNING")
print("="*60)

# Load digits dataset
X_digits, y_digits = load_digits(return_X_y=True)
X_train_dig, y_train_dig = X_digits[:1400], y_digits[:1400]
X_test_dig, y_test_dig = X_digits[1400:], y_digits[1400:]

# Baseline: Train with only 50 labeled samples
n_labeled = 50
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_dig[:n_labeled], y_train_dig[:n_labeled])
score_baseline = log_reg.score(X_test_dig, y_test_dig)
print(f"\nBaseline (50 labeled): {score_baseline:.4f}")

# Use K-Means to find representative samples
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train_dig)

# Find closest sample to each centroid
representative_digit_idx = X_digits_dist.argmin(axis=0)
X_representative_digits = X_train_dig[representative_digit_idx]

# Manually label these 50 representative samples
y_representative_digits = np.array([
    8, 0, 1, 3, 6, 7, 5, 4, 2, 8,
    2, 3, 9, 5, 3, 9, 1, 7, 9, 1,
    4, 6, 9, 7, 5, 2, 2, 1, 3, 3,
    6, 0, 4, 9, 8, 1, 8, 4, 2, 4,
    2, 3, 9, 7, 8, 9, 6, 5, 6, 4,
])

# Train on representative samples
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_representative_digits, y_representative_digits)
score_representative = log_reg.score(X_test_dig, y_test_dig)
print(f"Representative samples: {score_representative:.4f}")

# Propagate labels to all training samples
y_train_propagated = np.empty(len(X_train_dig), dtype=np.int64)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_dig, y_train_propagated)
score_full_propagation = log_reg.score(X_test_dig, y_test_dig)
print(f"Full label propagation: {score_full_propagation:.4f}")

# Partial propagation: Only propagate to closest 50% of samples
percentile_closest = 50
X_cluster_dist = X_digits_dist[np.arange(len(X_train_dig)), kmeans.labels_]

for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partial = X_train_dig[partially_propagated]
y_train_partial = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_partial, y_train_partial)
score_partial = log_reg.score(X_test_dig, y_test_dig)
print(f"Partial propagation (50%): {score_partial:.4f}")

# Check accuracy of propagated labels
accuracy_propagated = (y_train_partial == y_train_dig[partially_propagated]).mean()
print(f"Propagation accuracy: {accuracy_propagated:.4f}")

# ============================================================================
# 6. DBSCAN - Density-Based Clustering
# ============================================================================
print("\n" + "="*60)
print("DBSCAN - DENSITY-BASED CLUSTERING")
print("="*60)

# DBSCAN can find arbitrary-shaped clusters and detect outliers
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X_moons)

print(f"\nLabels (first 10): {dbscan.labels_[:10]}")
print(f"Core samples (first 10): {dbscan.core_sample_indices_[:10]}")
print(f"Number of clusters: {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}")
print(f"Number of outliers: {list(dbscan.labels_).count(-1)}")

# DBSCAN with different eps
dbscan2 = DBSCAN(eps=0.2)
dbscan2.fit(X_moons)
print(f"\nWith eps=0.2, clusters: {len(set(dbscan2.labels_)) - (1 if -1 in dbscan2.labels_ else 0)}")

# Predict new points using KNN on core samples
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan2.components_, dbscan2.labels_[dbscan2.core_sample_indices_])

X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
print(f"\nPredictions for new points: {knn.predict(X_new)}")
print(f"Prediction probabilities:\n{knn.predict_proba(X_new)}")

# ============================================================================
# 7. SPECTRAL CLUSTERING - Graph-Based Clustering
# ============================================================================
print("\n" + "="*60)
print("SPECTRAL CLUSTERING")
print("="*60)

# Spectral clustering uses graph theory (similarity matrix)
sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
sc1.fit(X_moons)
print(f"\nSpectral Clustering (gamma=100) labels: {sc1.labels_[:10]}")

sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
sc2.fit(X_moons)
print(f"Spectral Clustering (gamma=1) labels: {sc2.labels_[:10]}")

# ============================================================================
# 8. AGGLOMERATIVE CLUSTERING - Hierarchical Clustering
# ============================================================================
print("\n" + "="*60)
print("AGGLOMERATIVE CLUSTERING")
print("="*60)

# Hierarchical clustering builds a tree (dendrogram)
X_simple = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
agg = AgglomerativeClustering(linkage="complete").fit(X_simple)

print(f"\nChildren (merge history): {agg.children_}")
print(f"Number of clusters: {agg.n_clusters_}")
print(f"Labels: {agg.labels_}")

# ============================================================================
# 9. GAUSSIAN MIXTURE MODELS - Probabilistic Clustering
# ============================================================================
print("\n" + "="*60)
print("GAUSSIAN MIXTURE MODELS")
print("="*60)

# GMM assumes data comes from mixture of Gaussian distributions
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X_gmm = np.r_[X1, X2]
y_gmm = np.r_[y1, y2]

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X_gmm)

print(f"\nWeights: {gm.weights_.round(3)}")
print(f"Means:\n{gm.means_.round(2)}")
print(f"Converged: {gm.converged_}")
print(f"Iterations: {gm.n_iter_}")

# Predict cluster probabilities (soft clustering)
print(f"\nPredictions: {gm.predict(X_gmm[:5])}")
print(f"Probabilities:\n{gm.predict_proba(X_gmm[:5]).round(3)}")

# Generate new samples from the model
X_new_samples, y_new_samples = gm.sample(6)
print(f"\nGenerated samples:\n{X_new_samples.round(2)}")
print(f"Generated labels: {y_new_samples}")

# Anomaly detection using density
densities = gm.score_samples(X_gmm)
density_threshold = np.percentile(densities, 2)
anomalies = X_gmm[densities < density_threshold]
print(f"\nNumber of anomalies (bottom 2%): {len(anomalies)}")

# ============================================================================
# 10. MODEL SELECTION - BIC & AIC
# ============================================================================
print("\n" + "="*60)
print("MODEL SELECTION - BIC & AIC")
print("="*60)

# BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion)
# Lower is better - penalizes model complexity
print(f"\nBIC: {gm.bic(X_gmm):.2f}")
print(f"AIC: {gm.aic(X_gmm):.2f}")

# Try different numbers of components
gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X_gmm)
             for k in range(1, 11)]
bics = [model.bic(X_gmm) for model in gms_per_k]
aics = [model.aic(X_gmm) for model in gms_per_k]

print("\nBIC and AIC for different k:")
for k, (bic, aic) in enumerate(zip(bics, aics), 1):
    print(f"  k={k}: BIC={bic:.2f}, AIC={aic:.2f}")

best_k_bic = bics.index(min(bics)) + 1
best_k_aic = aics.index(min(aics)) + 1
print(f"\nBest k (BIC): {best_k_bic}")
print(f"Best k (AIC): {best_k_aic}")

# ============================================================================
# 11. BAYESIAN GAUSSIAN MIXTURE - Automatic Component Selection
# ============================================================================
print("\n" + "="*60)
print("BAYESIAN GAUSSIAN MIXTURE")
print("="*60)

# Bayesian GMM automatically determines optimal number of components
bgm = BayesianGaussianMixture(n_components=10, n_init=10, max_iter=500, random_state=42)
bgm.fit(X_gmm)

# Weights close to 0 indicate unnecessary components
print(f"\nWeights: {bgm.weights_.round(2)}")
print(f"Non-zero components: {(bgm.weights_ > 0.01).sum()}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("""
Clustering Algorithms:

1. K-MEANS
   - Fast, scalable
   - Assumes spherical clusters
   - Must specify k
   - Use: Large datasets, known k

2. MINI-BATCH K-MEANS
   - Faster than K-Means
   - Uses mini-batches
   - Use: Very large datasets

3. DBSCAN
   - Finds arbitrary-shaped clusters
   - Detects outliers
   - No need to specify k
   - Use: Non-spherical clusters, outliers

4. SPECTRAL CLUSTERING
   - Graph-based
   - Good for non-convex clusters
   - Slow for large datasets
   - Use: Complex cluster shapes

5. AGGLOMERATIVE CLUSTERING
   - Hierarchical (dendrogram)
   - No need to specify k upfront
   - Slow for large datasets
   - Use: Hierarchical structure

6. GAUSSIAN MIXTURE MODELS
   - Probabilistic (soft clustering)
   - Can generate new samples
   - Anomaly detection
   - Use: Overlapping clusters, probabilities

7. BAYESIAN GMM
   - Automatic component selection
   - Regularizes weights
   - Use: Unknown number of clusters

Choosing the Right Algorithm:
✅ Known k, spherical clusters → K-Means
✅ Very large data → Mini-Batch K-Means
✅ Arbitrary shapes, outliers → DBSCAN
✅ Complex shapes → Spectral Clustering
✅ Hierarchical structure → Agglomerative
✅ Soft clustering, probabilities → GMM
✅ Unknown k → DBSCAN, Bayesian GMM
""")
