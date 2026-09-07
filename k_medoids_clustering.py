import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

# Read the CSV file
file_path = ""  # Replace with the path to your CSV file
df = pd.read_csv(file_path, header=None)

# Extract the feature data (drop the ID column)
X = df.iloc[:, 1:].values
sample_ids = df.iloc[:, 0].values  # Sample IDs

# Compute the Hamming distance matrix
hamming_distances = squareform(pdist(X, metric='hamming'))

# Initialise the PAM clustering parameters
np.random.seed(22)  # Set the random seed
n_clusters = 9  # Number of clusters
n_samples = X.shape[0]
medoids = np.random.choice(n_samples, n_clusters, replace=False)  # Randomly pick the initial medoids
prev_medoids = np.zeros_like(medoids)
clusters = None

# PAM clustering iterations
while not np.array_equal(medoids, prev_medoids):
    prev_medoids = medoids.copy()
    # Assign every point to its nearest medoid
    clusters = {i: [] for i in range(n_clusters)}
    for i in range(n_samples):
        distances = [hamming_distances[i, medoid] for medoid in medoids]
        cluster = np.argmin(distances)
        clusters[cluster].append(i)
    # Update the medoid of every cluster
    for cluster_id in range(n_clusters):
        cluster_points = clusters[cluster_id]
        if cluster_points:  # If the cluster is not empty
            medoids[cluster_id] = min(
                cluster_points,
                key=lambda m: sum(hamming_distances[m, p] for p in cluster_points)
            )
# Build the cluster labels
labels = np.zeros(n_samples, dtype=int)
for cluster_id, cluster_points in clusters.items():
    for point in cluster_points:
        labels[point] = cluster_id + 1  # Clusters are numbered from 1

# Compute the centre features of every cluster
cluster_centers = {}
for cluster_id, cluster_points in clusters.items():
    cluster_data = X[cluster_points]
    cluster_centers[cluster_id + 1] = np.mean(cluster_data, axis=0)  # Clusters are numbered from 1

# Compute the average silhouette width
silhouette_avg = silhouette_score(hamming_distances, labels, metric='precomputed')

# Compute the silhouette coefficient of every sample
silhouette_values = silhouette_samples(hamming_distances, labels, metric='precomputed')

# Compute the sum of squared errors (SSE)
sse = 0
for cluster_id, cluster_points in clusters.items():
    medoid = X[medoids[cluster_id]]
    for point in cluster_points:
        sse += np.sum((X[point] - medoid) ** 2)

# Print the results
print(f"Cluster labels: {labels}")
print(f"Cluster centres (medoids): {medoids}")
print(f"Average silhouette width: {silhouette_avg:.4f}")
print(f"Sum of squared errors (SSE): {sse:.4f}")

# Print the centre features and the sample IDs of every cluster
print("\nCentre features and sample IDs of every cluster:")
for cluster_id, cluster_points in clusters.items():
    cluster_sample_ids = sample_ids[cluster_points]
    cluster_center = cluster_centers[cluster_id + 1]
    print(f"Cluster {cluster_id}:")
    print(f"  Sample IDs: {cluster_sample_ids}")
    print(f"  Centre features: {cluster_center}")

# Enlarge the fonts of the figure
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # Global font size
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 20  # Axis label font size
plt.rcParams['xtick.labelsize'] = 20  # x-axis tick font size
plt.rcParams['ytick.labelsize'] = 20  # y-axis tick font size
plt.rcParams['legend.fontsize'] = 20  # Legend font size

# Plot the silhouette diagram
plt.figure(figsize=(12, 8))  # Enlarge the figure
y_lower = 10
for i in range(1, n_clusters + 1):  # Clusters are numbered from 1
    ith_cluster_silhouette_values = silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        alpha=0.7,
        label=f"Cluster {i}"
    )
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}", fontsize=20)
    y_lower = y_upper + 10  # Add a blank gap between clusters

plt.axvline(x=silhouette_avg, color="red", linestyle="--", label="ASW")
plt.xlabel("Silhouette coefficient", fontsize=24)
plt.ylabel("Sample size", fontsize=24)
plt.legend(fontsize=24)
plt.tight_layout()  # Adjust the layout automatically
plt.show()
