import numpy as np
import matplotlib.pyplot as plt

class MLPLogisticRegression:

    def __init__(self, input_size, learning_rate=0.01, loss='bce', optimizer='sgd', print_every=10, patience=5):
        assert loss.lower() in ['bce', 'mse'], "Loss function must be either 'bce' or 'mse'"
        assert optimizer.lower() in ['sgd', 'bgd', 'mbgd'], "Optimizer must be either 'sgd', 'bgd', or 'mbgd'"
        assert input_size > 0, "Input size must be greater than 0"
        assert learning_rate > 0, "Learning rate must be greater than 0"
        
        self.input_size = input_size
        self.output_size = 1 
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.patience = patience
        
        self.weights = np.random.randn(input_size, 1) 
        self.bias = np.zeros((1, 1)) 
        self.print_every = print_every
        self.train_losses = []
        self.val_losses = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_pred, y):
        m = y.shape[0]
        if self.loss == 'bce':
            return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        elif self.loss == 'mse':
            return np.mean((y_pred - y) ** 2)

    def _forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def _backward(self, X, y, y_pred):
        m = X.shape[0]
        if self.loss == 'bce':
            dz = y_pred - y
        elif self.loss == 'mse':
            dz = (y_pred - y) * y_pred * (1 - y_pred) 
        
        dW = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)
        return dW, db

    def update_parameters(self, dW, db):
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db

    def fit(self, X_train, y_train, X_val, y_val, max_epochs=50):
        y_train = np.expand_dims(y_train, axis=1)
        y_val = np.expand_dims(y_val, axis=1)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            y_pred_train = self._forward(X_train)
            train_loss = self._compute_loss(y_pred_train, y_train)
            self.train_losses.append(train_loss)

            dW, db = self._backward(X_train, y_train, y_pred_train)
            self.update_parameters(dW, db)
            y_pred_val = self._forward(X_val)
            val_loss = self._compute_loss(y_pred_val, y_val)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % self.print_every == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

        return self.train_losses, self.val_losses  

    def predict(self, X):
        y_pred = self._forward(X)
        return (y_pred > 0.5).astype(int)
