import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k, iteration_lim=300, tolerance=1e-4):
        self.k = k
        self.iteration_lim = iteration_lim
        self.tolerance = tolerance
        self.centroids = None

    def fit(self, X):
        # 100, 102, 105
        # np.random.seed(1024) # k = 8
        np.random.seed(43)  # k = 8
        # np.random.seed(33) # k = 5
        # np.random.seed(35) # k = 4
        # np.random.seed(36) # k = 5

        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.iteration_lim):
            distances = self._compute_distances(X)
            clusters = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [X[clusters == j].mean(axis=0) for j in range(self.k)]
            )

            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def getCost(self, X):
        distances = self._compute_distances(X)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances**2)

    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
