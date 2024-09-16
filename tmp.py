import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM
from models.pca.pca import PCA
from sklearn.mixture import GaussianMixture

# from sklearn.decomposition import PCA as SklearnPCA
from mpl_toolkits.mplot3d import Axes3D

K_KMEANS_1 = 8
K_GMM_1 = 1
K_2 = 4


def load_embeddings(file_path):
    """https://pandas.pydata.org/docs/reference/api/pandas.read_feather.html"""
    df = pd.read_feather(file_path)
    embeddings = np.vstack(df.iloc[:, 1].values)
    return embeddings

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    embeddings = df[["x", "y"]].values
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

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig("assignments/2/figures/elbow_kmeans.png")

def k_means_tasks(embeddings):
    elbow_method(embeddings, max_k=25)
    cluster_labels, cost = perform_kmeans_clustering(embeddings, K_KMEANS_1)

    print(f"Optimal number of clusters (k_kmeans1): {K_KMEANS_1}")
    print(f"Cluster labels: {cluster_labels}")
    print(f"WCSS cost for optimal k: {cost}")
    # plot_clusters(embeddings, cluster_labels)


def perform_gmm_clustering(embeddings, k):
    print("Custom GMM:")
    gmm = GMM(k=k)
    gmm.fit(embeddings)
    params = gmm.getParams()
    membership = gmm.getMembership(embeddings)
    likelihood = gmm.getLikelihood(embeddings)
    print("Custom GMM Parameters:", params)
    print("Custom GMM Membership:", membership)
    print("Custom GMM Log Likelihood:", likelihood)
    
def perform_gmm_clustering_sklearn(embeddings, k):
    print("\nSklearn GMM:")
    sklearn_gmm = GaussianMixture(n_components=k, covariance_type='full')
    sklearn_gmm.fit(embeddings)

    soft_membership = sklearn_gmm.predict_proba(embeddings) 
    # hard_membership = sklearn_gmm.predict(embeddings)
    likelihood = sklearn_gmm.score_samples(embeddings).sum() 
    
    print("Sklearn GMM Soft Membership:", soft_membership)
    # print("Sklearn GMM Hard Membership:", hard_membership)
    print("Sklearn GMM Log Likelihood:", likelihood)
  
def plot_aic_bic_for_k(embeddings):
    k_list = [1, 5, 10, 20, 40, 80] 
    aic_values = []
    bic_values = []
    for k in k_list:
        #? My Implementaton
        gmm = GMM(k=k)
        gmm.fit(embeddings)
        aic_values.append(gmm.aic())
        bic_values.append(gmm.bic())
        #? SKlearn
        # sklearn_gmm = GaussianMixture(n_components=k, covariance_type="full")
        # sklearn_gmm.fit(embeddings)
        # aic_values.append(sklearn_gmm.aic(embeddings))
        # bic_values.append(sklearn_gmm.bic(embeddings))

    plt.figure(figsize=(8, 6))
    plt.plot(k_list, aic_values, label='AIC', marker='o')
    plt.plot(k_list, bic_values, label='BIC', marker='s')
    plt.title('AIC and BIC vs Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('AIC / BIC Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("assignments/2/figures/aic_bic_gmm.png")
        
    optimal_k_bic = np.argmin(bic_values) + 1
    optimal_k_aic = np.argmin(aic_values) + 1
    print(f"\nOptimal number of clusters based on BIC: {optimal_k_bic}")
    print(f"Optimal number of clusters based on AIC: {optimal_k_aic}")
    return optimal_k_bic, optimal_k_aic

def gmm_tasks(embeddings):
    perform_gmm_clustering(embeddings, K_KMEANS_1)
    perform_gmm_clustering_sklearn(embeddings, K_KMEANS_1)
    plot_aic_bic_for_k(embeddings)
    perform_gmm_clustering(embeddings, K_GMM_1)

def fit_transform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    transformed_data = pca.transform()
    pca.checkPCA()
    return transformed_data

def visualize_2D(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], color="green", label="PCA 2D Data")
    plt.title("PCA Transformed Data (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

def visualize_3D(data):
    data = np.real(data)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="blue", label="PCA 3D Data")
    ax.set_title("PCA Transformed Data (3D)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    plt.show()

def pca_tasks(embeddings):
    pca_2D_data = fit_transform_pca(embeddings, 2)
    visualize_2D(pca_2D_data)
    pca_3D_data = fit_transform_pca(embeddings, 3)
    visualize_3D(pca_3D_data)

def pca_n_clustering(embeddings):
    def plot_clusters(embeddings, cluster_labels):
    pca = PCA(n_components=20)
    pca.fit(embeddings)
    reduced_embeddings = pca.transform()
    pca.checkPCA()
    plt.figure(figsize=(10, 6))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=cluster_labels,
        cmap="plasma",
        marker="o",
    )
    plt.title("Clusters Visualization in 2D")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    # plt.show()

def main():

    file_path = "data/interim/2/word-embeddings.feather"
    file_path_kaggle = "data/interim/2/2d_clustering_kaggle.csv"
    embeddings = load_embeddings(file_path)
    # embeddings = load_csv_data(file_path_kaggle) #? uncomment to test on 2D dataset
    
    # k_means_tasks(embeddings)
    # gmm_tasks(embeddings)
    # pca_tasks(embeddings)
    perform_kmeans_clustering(embeddings, K_2)



if __name__ == "__main__":
    main()
