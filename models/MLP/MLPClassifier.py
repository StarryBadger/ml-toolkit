import numpy as np
import wandb
from performance_measures.classification_metrics import Metrics


class MLPClassifier:

    def __init__(self, input_size, hidden_layers, num_classes=6, learning_rate=0.01, activation='sigmoid', optimizer='sgd', wandb_log=False, print_every=10):
        assert activation.lower() in ['sigmoid', 'relu', 'tanh', 'linear', 'signum'], "Activation function must be either 'sigmoid', 'relu' or 'tanh' (or 'linear' for testing)"
        assert optimizer.lower() in ['sgd', 'bgd', 'mbgd'], "Optimizer must be either 'sgd', 'bgd' or 'mbgd'"
        assert input_size > 0, "Input size must be greater than 0"
        assert num_classes > 0, "Output size must be greater than 0"
        assert learning_rate > 0, "Learning rate must be greater than 0"
        assert type(hidden_layers) == list and len(hidden_layers) > 0, "Hidden layers must be a list of size greater than 0"
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = num_classes
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        
        self.weights, self.biases = self._initialize_weights_and_biases()
        self.train_losses = []
        self.validation_losses = []
        self.label_map = None
        self.inverse_label_map = None

        self.print_every = print_every
        self. wandb_log = wandb_log

    def _initialize_weights_and_biases(self):
        num_layers = len(self.hidden_layers)
        weights = []
        biases = []

        if num_layers == 0:
            w = np.random.randn(self.input_size, self.output_size)
            b = np.zeros((1, self.output_size))
            weights.append(w)
            biases.append(b)
            return weights, biases
        
        for i in range(num_layers + 1):
            if i == 0:
                w = np.random.randn(self.input_size, self.hidden_layers[0])
            elif i == num_layers:
                w = np.random.randn(self.hidden_layers[-1], self.output_size)
            else:
                w = np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i])
            
            b = np.zeros((1, w.shape[1]))
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
        elif activation == "linear":
            return X
        elif activation == "softmax":
            exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        elif activation == "signum":
            return np.where(X > 0, 1, np.where(X < 0, -1, 0))
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self, X, activation):
        if activation == "sigmoid":
            return X * (1 - X)
        elif activation == "tanh":
            return 1 - np.power(X, 2)
        elif activation == "relu":
            return (X > 0).astype(float)
        elif activation == "linear":
            return np.ones_like(X)
        elif activation == "signum":
            return np.zeros_like(X)        
        else:
            raise ValueError("Unsupported activation function")

    def forward_propagation(self, X):

        self.layer_outputs = [X]

        for i in range(len(self.weights)):

            if i == len(self.weights) - 1:
                Z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
                A = self._activate(Z, "softmax")

            else:
                Z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
                A = self._activate(Z, self.activation)

            self.layer_outputs.append(A)

        return self.layer_outputs[-1]

    def backpropagation(self, X, y, A):

        m = X.shape[0]
        self.gradients = []
        num_layers = len(self.weights)
        delta = A - y
        dW = (1 / m) * np.dot(self.layer_outputs[-2].T, delta)
        db = (1 / m) * np.sum(delta, axis=0, keepdims=True)
        self.gradients.append((dW, db))

        for i in reversed(range(num_layers - 1)):
            dA = np.dot(delta, self.weights[i + 1].T)
            delta = dA * self._activation_derivative(
                self.layer_outputs[i + 1], self.activation
            )
            dW = (1 / m) * np.dot(self.layer_outputs[i].T, delta)
            db = (1 / m) * np.sum(delta, axis=0, keepdims=True)
            self.gradients.append((dW, db))

        self.gradients = self.gradients[::-1]

    def update_parameters(self):
        for i in range(len(self.weights)):
            if self.optimizer in ["bgd", "sgd", "mbgd"]:
                self.weights[i] -= self.learning_rate * self.gradients[i][0]
                self.biases[i] -= self.learning_rate * self.gradients[i][1]
            else:
                raise ValueError("Unsupported optimizer")

    def fit(self, X_train, y_train, X_validation=None, y_validation=None, max_epochs=10, batch_size=32, early_stopping=False, patience=100):
        unique_labels = np.unique(y_train)
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.inverse_label_map = {i: label for label, i in self.label_map.items()}

        y_onehot_train = self._one_hot_encode(y_train, self.output_size)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            if self.optimizer == "bgd":
                y_pred_train = self.forward_propagation(X_train)
                self.backpropagation(X_train, y_onehot_train, y_pred_train)
                self.update_parameters()
            elif self.optimizer == "sgd":
                for i in range(X_train.shape[0]): 
                    X_sample = X_train[i:i+1] 
                    y_sample = y_onehot_train[i:i+1] 
                    y_pred_sample = self.forward_propagation(X_sample)
                    self.backpropagation(X_sample, y_sample, y_pred_sample)
                    self.update_parameters()
            elif self.optimizer == "mbgd":
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                for start_idx in range(0, X_train.shape[0], batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    X_batch = X_train[batch_indices]
                    y_batch = y_onehot_train[batch_indices]

                    y_pred_batch = self.forward_propagation(X_batch)
                    self.backpropagation(X_batch, y_batch, y_pred_batch)
                    self.update_parameters()

            current_loss = self._compute_loss(X_train, y_onehot_train)
            self.train_losses.append(current_loss)

            if X_validation is not None and y_validation is not None:
                validation_loss = self._compute_loss(X_validation, self._one_hot_encode(y_validation, self.output_size))
                self.validation_losses.append(validation_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping and patience_counter > patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            y_pred_train = self.predict(X_train)
            y_pred_validation = self.predict(X_validation)

            train_metrics = Metrics(y_train, y_pred_train, task="classification")
            validation_metrics = Metrics(y_validation, y_pred_validation, task="classification")

            train_accuracy = train_metrics.accuracy()
            train_precision = train_metrics.precision_score()
            train_recall = train_metrics.recall_score()
            train_f1 = train_metrics.f1_score()

            validation_accuracy = validation_metrics.accuracy()
            validation_precision = validation_metrics.precision_score()
            validation_recall = validation_metrics.recall_score()
            validation_f1 = validation_metrics.f1_score()

            if self.wandb_log:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": current_loss,
                    "validation_loss": validation_loss,
                    "train_accuracy": train_accuracy,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "validation_accuracy": validation_accuracy,
                    "validation_precision": validation_precision,
                    "validation_recall": validation_recall,
                    "validation_f1": validation_f1,
                })

            if epoch % self.print_every == 0 or epoch == max_epochs-1:
                print(
                    f"Epoch {epoch}, Train Loss: {current_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}"
                )

        return self.train_losses

    def predict(self, X):
        probabilities = self.forward_propagation(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return np.array([self.inverse_label_map[i] for i in predicted_indices])

    def _one_hot_encode(self, Y, num_classes):
        return np.eye(num_classes)[Y]

    def _compute_loss(self, X, y):
        y_pred = self.forward_propagation(X)
        return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))

    def gradient_checking(self, X, y, epsilon=1e-7):

        y_onehot = self._one_hot_encode(y, self.output_size)

        y_pred = self.forward_propagation(X)
        self.backpropagation(X, y_onehot, y_pred)

        parameters = self.weights + self.biases
        gradients_analytical = [grad[0] for grad in self.gradients] + [
            grad[1] for grad in self.gradients
        ]

        gradients_numerical = []

        for param in range(len(parameters)):
            param_shape = parameters[param].shape
            grad_numerical = np.zeros_like(parameters[param])

            for i in range(param_shape[0]):
                for j in range(param_shape[1]):
                    theta_plus = np.copy(parameters[param])
                    theta_minus = np.copy(parameters[param])
                    theta_plus[i, j] += epsilon
                    theta_minus[i, j] -= epsilon

                    if param < len(self.weights):
                        self.weights[param] = theta_plus
                    else:
                        self.biases[param - len(self.weights)] = theta_plus
                    loss_plus = self._compute_loss(X, y_onehot)

                    if param < len(self.weights):
                        self.weights[param] = theta_minus
                    else:
                        self.biases[param - len(self.weights)] = theta_minus
                    loss_minus = self._compute_loss(X, y_onehot)

                    grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

            gradients_numerical.append(grad_numerical)

        for grad_num, grad_anl in zip(gradients_numerical, gradients_analytical):
            relative_difference = np.linalg.norm(grad_anl - grad_num) / (
                np.linalg.norm(grad_anl) + np.linalg.norm(grad_num) + 1e-8
            )
            if relative_difference > 1e-6:
                print(
                    f"Gradient check failed with relative difference: {relative_difference}"
                )
                return False

        print("Gradient check passed!")
        return True
