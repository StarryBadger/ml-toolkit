import numpy as np

class MultiLabelMLP:

    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, activation='sigmoid', optimizer='sgd', print_every=10):
        assert activation.lower() in ['sigmoid', 'relu', 'tanh'], "Activation must be 'sigmoid', 'relu', or 'tanh'"
        assert optimizer.lower() in ['sgd', 'bgd', 'mbgd'], "Optimizer must be 'sgd', 'bgd', or 'mbgd'"
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer

        # Initialize weights and biases
        self.weights, self.biases = self._initialize_weights_and_biases()
        self.print_every = print_every

    def _initialize_weights_and_biases(self):
        weights = []
        biases = []
        layers = [self.input_size] + self.hidden_layers + [self.output_size]

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1])
            b = np.zeros((1, layers[i + 1]))
            weights.append(w)
            biases.append(b)

        return weights, biases
    
    def _activate(self, X, activation):
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-X))
        elif activation == "tanh":
            return np.tanh(X)
        elif activation == "relu":
            return np.maximum(0, X)
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self, X, activation):
        if activation == "sigmoid":
            return X * (1 - X)
        elif activation == "tanh":
            return 1 - X ** 2
        elif activation == "relu":
            return (X > 0).astype(float)
        else:
            raise ValueError("Unsupported activation function")

    def forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            Z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                A = self._activate(Z, 'sigmoid')  # Use sigmoid in the output layer for multi-label
            else:
                A = self._activate(Z, self.activation)
            self.layer_outputs.append(A)
        return self.layer_outputs[-1]

    def backpropagation(self, X, y, A):
        m = X.shape[0]
        self.gradients = []
        num_layers = len(self.weights)

        # Compute delta for the output layer (Binary cross-entropy gradient)
        delta = A - y  
        dW = (1 / m) * np.dot(self.layer_outputs[-2].T, delta)
        db = (1 / m) * np.sum(delta, axis=0, keepdims=True)
        self.gradients.append((dW, db))

        # Backpropagate through the hidden layers
        for i in reversed(range(num_layers - 1)):
            dA = np.dot(delta, self.weights[i + 1].T)
            delta = dA * self._activation_derivative(self.layer_outputs[i + 1], self.activation)
            dW = (1 / m) * np.dot(self.layer_outputs[i].T, delta)
            db = (1 / m) * np.sum(delta, axis=0, keepdims=True)
            self.gradients.append((dW, db))

        self.gradients = self.gradients[::-1]  # Reverse gradients to match the layers

    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients[i][0]
            self.biases[i] -= self.learning_rate * self.gradients[i][1]

    def compute_loss(self, A, y):
        m = y.shape[0]
        # Binary cross-entropy loss
        loss = -np.mean(np.sum(y * np.log(A + 1e-8) + (1 - y) * np.log(1 - A + 1e-8), axis=1))
        return loss

    def fit(self, X_train, y_train, X_validation=None, y_validation=None, max_epochs=10, batch_size=32, early_stopping=False, patience=100):
        best_loss = float('inf')
        patience_counter = 0
        self.train_losses = []
        self.validation_losses = []

        for epoch in range(max_epochs):
            # Forward propagation
            A_train = self.forward_propagation(X_train)
            train_loss = self.compute_loss(A_train, y_train)

            # Backpropagation and update parameters
            self.backpropagation(X_train, y_train, A_train)
            self.update_parameters()

            self.train_losses.append(train_loss)

            if X_validation is not None and y_validation is not None:
                A_val = self.forward_propagation(X_validation)
                val_loss = self.compute_loss(A_val, y_validation)
                self.validation_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if early_stopping and patience_counter > patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            if epoch % self.print_every == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss if X_validation is not None else 'N/A'}")

    def predict(self, X, threshold=0.5):
        probabilities = self.forward_propagation(X)
        return (probabilities >= threshold).astype(int)

