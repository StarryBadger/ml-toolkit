import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.k_means.k_means import KMeans
from sklearn.decomposition import PCA


def load_embeddings(file_path):
    df = pd.read_feather(file_path)
    embeddings = np.vstack(df.iloc[:, 1].values)
    return embeddings

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    embeddings = df[['x', 'y']].values
    return embeddings

def perform_kmeans_clustering(embeddings, k):
    kmeans = KMeans(k=k)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.predict(embeddings)
    cost = kmeans.getCost(embeddings)
    return cluster_labels, cost

def elbow_method(embeddings, max_k=25):
    wcss = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        _, cost = perform_kmeans_clustering(embeddings, k)
        wcss.append(cost)
    
    plot_elbow(k_values, wcss)

def plot_elbow(k_values, wcss):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

def plot_clusters(embeddings, cluster_labels):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='plasma', marker='o')
    plt.title("Clusters Visualization in 2D")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

def main():
    file_path = "data/external/word-embeddings.feather"
    embeddings = load_embeddings(file_path)

    # file_path = "data/external/2d_clustering_kaggle.csv"
    # embeddings = load_csv_data(file_path)

    elbow_method(embeddings, max_k=25)

    optimal_k = 8
    cluster_labels, cost = perform_kmeans_clustering(embeddings, optimal_k)

    print(f"Optimal number of clusters (k): {optimal_k}")
    print(f"Cluster labels: {cluster_labels}")
    print(f"WCSS cost for optimal k: {cost}")
    plot_clusters(embeddings, cluster_labels)

if __name__ == "__main__":
    main()
