import numpy as np


class PCA:
    def __init__(self, n_components):
        pass
        self.n_components = n_components

    def fit(self, X):
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        return eigenvectors[:,self.n_components]

    def transform(self):
        pass

    def checkPCA(self):
        pass
