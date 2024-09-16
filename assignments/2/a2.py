import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM
from models.pca.pca import PCA
from models.knn.knn import KNN
from sklearn.mixture import GaussianMixture
from performance_measures.classification_metrics import Metrics


# from sklearn.decomposition import PCA as SklearnPCA
from mpl_toolkits.mplot3d import Axes3D

K_KMEANS_1 = 5
K_GMM_1 = 1
K_2 = 4
K_KMEANS_3 = 11
K_GMM_3 = 1

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

def elbow_method(embeddings, max_k=25, save_path="assignments/2/figures/elbow_kmeans_5.png"):
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
    plt.savefig(save_path)

def k_means_tasks(embeddings):
    elbow_method(embeddings, max_k=25)
    cluster_labels, cost = perform_kmeans_clustering(embeddings, K_KMEANS_1)

    print(f"Optimal number of clusters (k_kmeans1): {K_KMEANS_1}")
    print(f"Cluster labels: {cluster_labels}")
    print(f"WCSS cost for optimal k: {cost}")
    
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
  
def plot_aic_bic_for_k(embeddings, save_path="assignments/2/figures/aic_bic_gmm.png"):
    k_list = [i for i in range(1,21)] 
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
    plt.savefig(save_path)
        
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
    
def perform_kmeans_clustering_K2(embeddings):
    cluster_labels, cost = perform_kmeans_clustering(embeddings, K_2)
    print(f"Cluster labels (k={K_2}): {cluster_labels}")
    print(f"WCSS cost for k={K_2}: {cost}")
    return cluster_labels

def generate_scree_plot(embeddings):
    pca = PCA(n_components=embeddings.shape[1])
    pca.fit(embeddings)

    explained_variance_ratios = []
    total_variance = np.sum(pca.eigenvalues)
    
    for i in range(1, embeddings.shape[1] + 1):
        explained_variance_ratio = np.sum(pca.eigenvalues[:i]) / total_variance
        explained_variance_ratios.append(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, marker='o', alpha=0.5)
    plt.title('Scree Plot: Cumulative Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.savefig("assignments/2/figures/scree_plot.png")
    optimal_dims = np.argmax(np.array(explained_variance_ratios) >= 0.90) + 1
    print(f"Optimal number of dimensions based on 85% explained variance: {optimal_dims}")

    explained_variance = pca.eigenvalues / np.sum(pca.eigenvalues)
    # explained_variance=explained_variance[:100]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', alpha=0.5)
    plt.title('Scree Plot: Variance Explained by Each Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.savefig("assignments/2/figures/scree_plot_individual.png")
    return optimal_dims

def perform_kmeans_on_reduced_data(embeddings, optimal_dims):
    print(f"Applying PCA to reduce dataset to {optimal_dims} dimensions.")
    reduced_data = fit_transform_pca(embeddings, optimal_dims)
    print("Determining the optimal number of clusters for reduced dataset using Elbow Method.")
    elbow_method(reduced_data, save_path="assignments/2/figures/elbow_kmeans_optimal_clusters.png")
    print(f"Performing K-Means clustering with k={K_KMEANS_3} on the reduced dataset.")
    cluster_labels, cost = perform_kmeans_clustering(reduced_data, K_KMEANS_3)
    
    print(f"Cluster labels for reduced dataset (k={K_KMEANS_3}): {cluster_labels}")
    print(f"WCSS cost for reduced dataset (k={K_KMEANS_3}): {cost}")
    return reduced_data, cluster_labels

def scree_and_reduced_kmeans_tasks(embeddings):
    perform_kmeans_clustering_K2(embeddings)
    optimal_dims = generate_scree_plot(embeddings)
    reduced_data, reduced_cluster_labels = perform_kmeans_on_reduced_data(embeddings, optimal_dims)
    return reduced_data, reduced_cluster_labels

def determine_optimal_kgmm3(embeddings):
    global K_GMM_3
    optimal_k_bic, optimal_k_aic = plot_aic_bic_for_k(embeddings)
    K_GMM_3 = optimal_k_bic
    print(f"Optimal number of clusters based on BIC: {K_GMM_3}")

def pca_gmm_tasks(embeddings):
    optimal_dims = generate_scree_plot(embeddings)
    reduced_data = fit_transform_pca(embeddings, optimal_dims)
    perform_gmm_clustering(reduced_data, K_GMM_3)

def split(X, train_ratio=0.8, val_ratio=0.1):
    np.random.seed(1)
    indices = np.random.permutation(len(X))
    X_shuffled = X.iloc[indices]

    train_size = int(train_ratio * len(X_shuffled))
    val_size = int(val_ratio * len(X_shuffled))

    X_train = X_shuffled[:train_size]
    X_val = X_shuffled[train_size : train_size + val_size]
    X_test = X_shuffled[train_size + val_size :]
    return X_train, X_val, X_test

def knn_pca_task():
    file_path_spotify=  "data/interim/2/spotify_normalized_numerical.csv"
    df=pd.read_csv(file_path_spotify)
    genres = df['track_genre'].values
    features = df.drop(columns=['track_genre']).values
    generate_scree_plot(features)
    reduced_features=fit_transform_pca(features, 8)
    features_df = pd.DataFrame(reduced_features)
    result_df = features_df.copy()
    result_df['track_genre'] = genres

    train, val, _ = split(df)
    classifier = KNN(k=64, distance_metric="manhattan")
    classifier.fit(train.drop(columns=['track_genre']).values, train['track_genre'].values)
    start_time = time.time()
    y_pred = classifier.predict(val.drop(columns=['track_genre']).values)
    time_taken = time.time() - start_time
    Metrics(y_true=val['track_genre'].values, y_pred=y_pred, task="classification").print_metrics()
    print(f"{time_taken=}")

    train, val, _ = split(result_df)
    classifier = KNN(k=64, distance_metric="manhattan")
    classifier.fit(train.drop(columns=['track_genre']).values, train['track_genre'].values)
    start_time = time.time()
    y_pred = classifier.predict(val.drop(columns=['track_genre']).values)
    time_taken = time.time() - start_time
    Metrics(y_true=val['track_genre'].values, y_pred=y_pred, task="classification").print_metrics()
    print(f"{time_taken=}")

def main():

    file_path = "data/interim/2/word-embeddings.feather"
    file_path_kaggle = "data/interim/2/2d_clustering_kaggle.csv"
    embeddings = load_embeddings(file_path)
    # embeddings = load_csv_data(file_path_kaggle) #? uncomment to test on 2D dataset
    
    # k_means_tasks(embeddings)
    gmm_tasks(embeddings)
    # pca_tasks(embeddings)

    # perform_kmeans_clustering_K2(embeddings)
    # scree_and_reduced_kmeans_tasks(embeddings)

    # perform_gmm_clustering(embeddings, K_2)
    # determine_optimal_kgmm3(embeddings)
    # pca_gmm_tasks(embeddings)

    # knn_pca_task()

    

if __name__ == "__main__":
    main()
