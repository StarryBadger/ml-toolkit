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
            self.norm_X_train = np.linalg.norm(self.X_train, axis=1)

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
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = self.calculate_distances(x)
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]
        # ? https://stackoverflow.com/questions/16330831/most-efficient-way-to-find-mode-in-numpy-array
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def _euclidean_distances(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _manhattan_distances(self, x):
        return np.sum(np.abs(self.X_train - x), axis=1)

    def _cosine_distances(self, x):
        dot_product = np.dot(self.X_train, x)
        norm_x = np.linalg.norm(x)
        cosine_similarity = dot_product / (self.norm_X_train * norm_x)
        # ? https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        return 1 - cosine_similarity
