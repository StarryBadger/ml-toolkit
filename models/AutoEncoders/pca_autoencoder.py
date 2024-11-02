import numpy as np
import matplotlib.pyplot as plt
class PcaAutoencoder:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.eigenvectors = eigenvectors[:, :self.n_components]

    def encode(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)

    def forward(self, X):
        X_encoded = self.encode(X)
        return np.dot(X_encoded, self.eigenvectors.T) + self.mean

    def reconstruction_error(self, X):
        X_reconstructed = self.forward(X)
        mse_loss = np.mean(np.square(X - X_reconstructed))
        return mse_loss

def determine_optimal_components(X_train, max_components=50):
    errors = []
    for n in range(1, max_components + 1):
        pca_autoencoder = PcaAutoencoder(n_components=n)
        pca_autoencoder.fit(X_train)
        error = pca_autoencoder.reconstruction_error(X_train)
        errors.append(error)

    plt.plot(range(1, max_components + 1), errors, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs Number of Components')
    plt.show()

    optimal_components = np.argmin(np.diff(errors)) + 1
    return optimal_components


