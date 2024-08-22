import numpy as np
class KNN:
    def __init__(self, k: int, distance_metrics: str = "euclidean"):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k

        if distance_metrics.lower() not in ["euclidean", "cosine", "manhattan"]:
            raise ValueError("Distance metric not supported")
        self.distance_metrics = distance_metrics.lower()

        if self.distance_metrics == "euclidean":
            self.calculate_distance = self._euclidean_distance
        elif self.distance_metrics == "manhattan":
            self.calculate_distance = self._manhattan_distance
        elif self.distance_metrics == "cosine":
            self.calculate_distance = self._cosine_distance

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = self.calculate_distance(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        #? https://stackoverflow.com/questions/16330831/most-efficient-way-to-find-mode-in-numpy-array
        unique_labels,counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def _euclidean_distance(self, x):
        return np.sqrt(np.sum((self.X_train - x)**2, axis=1))

    def _manhattan_distance(self, x):
        return np.sum(np.abs(self.X_train - x), axis=1)

    def _cosine_distance(self, x):
        dot_product = np.dot(self, x)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(x)
        cosine_similarity = dot_product / (norm_x * norm_y)
        # ? https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        return 1 - cosine_similarity

    # def scoring(self, actual_labels, pred_labels):
    #     f1 = f1_score(actual_labels, pred_labels, zero_division=0, average="weighted")
    #     accuracy = accuracy_score(actual_labels, pred_labels)
    #     precision = precision_score(
    #         actual_labels, pred_labels, zero_division=0, average="weighted"
    #     )
    #     recall = recall_score(
    #         actual_labels, pred_labels, zero_division=0, average="weighted"
    #     )

        # Return a dictionary of scores rounded off to 4 decimal places
        # return {'f1': round(f1, 4), 'accuracy': round(accuracy, 4), 'precision': round(precision, 4), 'recall': round
