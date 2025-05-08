import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import pdist

file_path = "data/interim/2/word-embeddings.feather"
df = pd.read_feather(file_path)
embeddings = np.vstack(df.iloc[:, 1].values)

linkage_methods = ['single', 'complete', 'average', 'ward', 'centroid']
distance_metrics = ['euclidean', 'cityblock', 'cosine']

def plot_dendrogram(Z, title, save_as):
    plt.figure(figsize=(10, 7))
    hc.dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig(save_as)
    plt.close()

for metric in distance_metrics:
    for method in linkage_methods:
        dist_matrix = pdist(embeddings, metric=metric)
        Z = hc.linkage(dist_matrix, method=method)
        save_as=f"assignments/2/figures/dendrograms/{metric}_{method}.png"
        plot_dendrogram(Z, f'Dendrogram ({metric} distance, {method} linkage)', save_as)
        print(f"Linkage matrix shape for {metric} distance and {method} linkage: {Z.shape}")
        print("First few rows of the linkage matrix:")
        print(Z[:5])
        print("\n")

best_linkage = 'ward'

kbest1 = 4
kbest2 = 5 

Z_best = hc.linkage(embeddings, method=best_linkage, metric='euclidean')
clusters_kbest1 = hc.fcluster(Z_best, t=kbest1, criterion='maxclust')
clusters_kbest2 = hc.fcluster(Z_best, t=kbest2, criterion='maxclust')
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters_kbest1, cmap='viridis')
plt.title(f'Hierarchical Clustering (k={kbest1})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(122)
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters_kbest2, cmap='viridis')
plt.title(f'Hierarchical Clustering (k={kbest2})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Cluster assignments for kbest1:")
print(clusters_kbest1)
print("\nCluster assignments for kbest2:")
print(clusters_kbest2)