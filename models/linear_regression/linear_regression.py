import numpy as np

class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.05, iterations=1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _polynomial_features(self, X):
        return np.column_stack([X**i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iterations):
            y_pred = self._predict(X)
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _predict(self, X):
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias

    def predict(self, X):
        return self._predict(X)
    
    def score(self, X, y):
        """Calculate the Mean Squared Error."""
        predictions = self.predict(X)
        mse = np.mean((predictions - y)**2)
        return mse