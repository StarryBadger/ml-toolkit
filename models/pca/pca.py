import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.principal_components = None
        self.X = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X):
        self.X = X
        self.mean = np.mean(self.X, axis=0)
        X_centered = self.X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.principal_components = eigenvectors[:, :self.n_components]

    def transform(self):
        self.X_transformed = np.dot(self.X - self.mean, self.principal_components)
        return np.real(self.X_transformed)

    def get_principal_axes(self):
        return self.principal_components, self.mean

    def checkPCA(self):
        X_reduced = self.transform()
        X_reconstructed = np.dot(X_reduced, self.principal_components.T) + self.mean
        reconstruction_error = np.mean(np.square(self.X - X_reconstructed))
        total_variance = np.sum(self.eigenvalues)
        explained_variance_ratio = np.sum(self.eigenvalues[:self.n_components]) / total_variance
        
        print(f"Reconstruction Error: {reconstruction_error}")
        print(f"Explained Variance Ratio: {explained_variance_ratio:.4f}")
        
        return reconstruction_error<0.15