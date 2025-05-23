import numpy as np


class Metrics:
    def __init__(self, y_true, y_pred, task):
        if task.lower() == "regression":
            self.y_true = y_true
            self.y_pred = y_pred
            self.print_metrics = self.print_regression_metrics
        elif task.lower() == "classification":
            self.y_true = y_true
            self.y_pred = y_pred
            self.classes = np.unique(np.concatenate((y_true, y_pred)))
            self.print_metrics = self.print_classification_metrics
        else:
            raise ValueError(f"Not metric support for {task}")

    def confusion_matrix(self):
        true_indices = np.searchsorted(self.classes, self.y_true)
        pred_indices = np.searchsorted(self.classes, self.y_pred)
        confusion = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        np.add.at(confusion, (true_indices, pred_indices), 1)
        return confusion
    # To handle np.sum(confusion, axis=n) = 0 for n=0 or 1, we add 1e-6 to each element. 
    def accuracy(self, one_hot = False):
        if not one_hot:
            return np.sum(self.y_true == self.y_pred) / len(self.y_true)
        return np.sum(np.all(self.y_true == self.y_pred, axis=1)) / self.y_true.shape[0]


    def precision_score(self, average="macro"):
        confusion = self.confusion_matrix()
        precisions = np.diagonal(confusion) / (np.sum(confusion, axis=0)+1e-6)
        if average == "macro":
            return np.mean(precisions)
        elif average == "micro":
            return np.sum(np.diagonal(confusion)) / np.sum(confusion)

    def recall_score(self, average="macro"):
        confusion = self.confusion_matrix()
        recalls = np.diagonal(confusion) / (np.sum(confusion, axis=1)+1e-6)
        if average == "macro":
            return np.mean(recalls)
        elif average == "micro":
            return np.sum(np.diagonal(confusion)) / np.sum(confusion)

    def f1_score(self, average="macro"):
        precision = self.precision_score(average)
        recall = self.recall_score(average)
        return 2 * (precision * recall) / (precision + recall)
    

    def hamming_loss(self):
        return np.mean(self.y_true != self.y_pred)

    def hamming_accuracy(self):
        return 1 - self.hamming_loss()

    def print_classification_metrics(self):
        print("Classification Task Scores")
        print("---------------------------")
        print(f"Accuracy: {self.accuracy():.4f}")
        for x in ["macro", "micro"]:
            print(f"  Precision ({x}): {self.precision_score(average=x):.4f}")
            print(f"  Recall ({x}): {self.recall_score(average=x):.4f}")
            print(f"  F1-Score ({x}): {self.f1_score(average=x):.4f}")
        print("---------------------------")

    def mse(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    
    def rmse(self):
        print(self.mse())
        print(np.sqrt(self.mse()))

        return np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
    
    def mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))

    def standard_deviation(self):
        return np.sqrt(np.var(self.y_true - self.y_pred))

    def variance(self):
        return np.var(self.y_true - self.y_pred)

    def r2_score(self):
        residuals = self.y_true - self.y_pred
        total_sum_of_squares = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        residual_sum_of_squares = np.sum(residuals ** 2)
        return 1 - (residual_sum_of_squares / total_sum_of_squares)
    
    def print_regression_metrics(self):
        print("Regression Task Scores")
        print("---------------------------")
        print(f"Mean Squared Error (MSE): {self.mse():.4f}")
        print(f"Standard Deviation: {self.standard_deviation():.4f}")
        print(f"Variance: {self.variance():.4f}")
        print("---------------------------")

    
