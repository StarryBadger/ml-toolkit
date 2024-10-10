import numpy as np

class MLPLogisticRegression:

    def __init__(self, input_size, learning_rate=0.01, loss='bce', optimizer='sgd', print_every=10):
        assert loss.lower() in ['bce', 'mse'], "Loss function must be either 'bce' or 'mse'"
        assert optimizer.lower() in ['sgd', 'bgd', 'mbgd'], "Optimizer must be either 'sgd', 'bgd', or 'mbgd'"
        assert input_size > 0, "Input size must be greater than 0"
        assert learning_rate > 0, "Learning rate must be greater than 0"
        
        self.input_size = input_size
        self.output_size = 1  # Logistic regression has 1 output
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        
        self.weights = np.random.randn(input_size, 1)  # Weights for logistic regression
        self.bias = np.zeros((1, 1))  # Bias term
        self.print_every = print_every
        self.train_losses = []

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

    def fit(self, X_train, y_train, max_epochs=10):
        y_train = np.expand_dims(y_train, axis=1)
        for epoch in range(max_epochs):
            y_pred = self._forward(X_train)
            loss = self._compute_loss(y_pred, y_train)
            self.train_losses.append(loss)

            dW, db = self._backward(X_train, y_train, y_pred)
            self.update_parameters(dW, db)
            if epoch % self.print_every == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self._forward(X)
        return (y_pred > 0.5).astype(int) 


