import numpy as np
class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = np.unique(np.concatenate((y_true, y_pred)))

    def confusion_matrix(self):
            true_indices = np.searchsorted(self.classes, self.y_true)
            pred_indices = np.searchsorted(self.classes, self.y_pred)
            confusion = np.zeros((len(self.classes), len(self.classes)), dtype=int)
            np.add.at(confusion, (true_indices, pred_indices), 1)
            return confusion

    def accuracy(self):
        return np.sum(self.y_true == self.y_pred) / len(self.y_true)

    def precision_score(self, average='macro'):
        confusion = self.confusion_matrix()
        precisions = np.diagonal(confusion) / np.sum(confusion, axis=0)
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            return np.sum(np.diagonal(confusion)) / np.sum(confusion)

    def recall_score(self, average='macro'):
        confusion = self.confusion_matrix()
        recalls = np.diagonal(confusion) / np.sum(confusion, axis=1)
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            return np.sum(np.diagonal(confusion)) / np.sum(confusion)

    def f1_score(self, average='macro'):
        precision = self.precision_score(average)
        recall = self.recall_score(average)
        return 2 * (precision * recall) / (precision + recall)
        
    