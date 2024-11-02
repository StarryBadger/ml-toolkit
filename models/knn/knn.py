import numpy as np

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.

    Parameters:
    -----------
    k : int
        The number of nearest neighbors to consider.
    distance_metric : str, optional (default="euclidean")
        The distance metric to use. Supported values are "euclidean", "cosine", and "manhattan".
    
    Attributes:
    -----------
    X_train : np.ndarray
        Training data features.
    y_train : np.ndarray
        Training data labels.
    norm_X_train : np.ndarray, optional
        Norms of training data features, used in cosine distance calculation.
    calculate_distances : function
        The function used to calculate distances based on the chosen metric.
    """
    def __init__(self, k: int, distance_metric: str = "euclidean"):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k

        if distance_metric.lower() not in ["euclidean", "cosine", "manhattan"]:
            raise ValueError("Distance metric not supported")
        self.distance_metric = distance_metric.lower()

        if self.distance_metric == "euclidean":
            self.calculate_distances = self._euclidean_distances
        elif self.distance_metric == "manhattan":
            self.calculate_distances = self._manhattan_distances
        elif self.distance_metric == "cosine":
            self.calculate_distances = self._cosine_distances
            self.norm_X_train = None

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the KNN model using the provided training data.

        Parameters:
        -----------
        X : np.ndarray
            Training data features.
        y : np.ndarray
            Training data labels.
        """
        self.X_train = X
        self.y_train = y
        if self.distance_metric== "cosine":
            self.norm_X_train_T = (self.X_train / np.linalg.norm(self.X_train, axis=1, keepdims=True)).T

    def predict(self, X_test):
        """
        Predict the labels for the provided test data.

        Parameters:
        -----------
        X_test : np.ndarray
            Test data features.

        Returns:
        --------
        np.ndarray
            Predicted labels for the test data.
        """
        distances = self.calculate_distances(X_test)
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_indices]
        predictions = np.array([np.bincount(labels).argmax() for labels in k_nearest_labels])
        return predictions

    def _euclidean_distances(self, X_test):
        # Calculate the squared differences and sum them along the feature axis
        return np.sqrt(np.sum((X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))

    def _manhattan_distances(self, X_test):
        # Calculate the absolute differences and sum them along the feature axis
        return np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)



    def _cosine_distances(self, x):
        # dot_product = np.dot(self.X_train, x)
        # norm_x = np.linalg.norm(x)
        # cosine_similarity = dot_product / (self.norm_X_train * norm_x)
        # # ? https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        # return 1 - cosine_similarity
        # X_train_norm = self.X_train / np.linalg.norm(self.X_train, axis=1, keepdims=True)
        X_test_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        # Calculate cosine distance as 1 - cosine similarity
        cosine_similarity = np.dot(X_test_norm, self.norm_X_train_T)
        return 1 - cosine_similarity  # Cosine distance
