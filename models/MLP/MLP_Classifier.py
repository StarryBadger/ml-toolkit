import numpy as np
import wandb
class MLP_Classifier:
    def __init__(self, input_size, hidden_layers, num_classes=6, learning_rate=0.01, activation='sigmoid', optimizer='sgd', wandb_log=False, print_every=10):
        assert activation.lower() in ['sigmoid', 'relu', 'tanh', 'linear'], "Activation function must be either 'sigmoid', 'relu' or 'tanh' (or 'linear' for testing)"
        assert optimizer.lower() in ['sgd', 'bgd', 'mbgd'], "Optimizer must be either 'sgd', 'bgd' or 'mbgd'"
        assert input_size > 0, "Input size must be greater than 0"
        assert num_classes > 0, "Output size must be greater than 0"
        assert learning_rate > 0, "Learning rate must be greater than 0"
        assert type(hidden_layers) == list and len(hidden_layers) > 0, "Hidden layers must be a list of size greater than 0"

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = num_classes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation_func = self._get_activation_func(activation)
        self.optimizer_func = self._get_optimizer_func(optimizer)
        self.weights, self.biases = self._initialize_weights_and_biases()

        self.wandb_log = wandb_log
        self.print_every = print_every
    
    def _get_activation_func(self, activation):
        if activation == 'sigmoid':
            return self._sigmoid
        elif activation == 'tanh':
            return self._tanh
        elif activation == 'relu':
            return self._relu
        elif activation == 'linear':
            return self._linear
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _relu(self, x):
        return np.maximum(0, x)
    
    def _linear(self, x):
        return x
    
    def _activation_derivative(self, Z):
        if self.activation_func == self._sigmoid:
            return self._sigmoid_derivative(Z)
        elif self.activation_func == self._tanh:
            return self._tanh_derivative(Z)
        elif self.activation_func == self._relu:
            return self._relu_derivative(Z)
        elif self.activation_func == self._linear:
            return self._linear_derivative(Z)
        else:
            raise ValueError(f"Activation function '{self.activation_func}' not supported.")
    
    def _sigmoid_derivative(self, Z):
        return self._sigmoid(Z) * (1 - self._sigmoid(Z))
    
    def _tanh_derivative(self, Z):
        return 1 - np.square(self._tanh(Z))
    
    def _relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)
    
    def _linear_derivative(self, Z):
        return np.ones_like(Z)
    
    def _get_optimizer_func(self, optimizer):
        if optimizer == 'sgd':
            return self._sgd
        elif optimizer == 'bgd':
            return self._bgd
        elif optimizer == 'mbgd':
            return self._mbgd
        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported.")
    
    def _sgd(self, grads):
        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * grads['dW'][i])
            self.biases[i] -= (self.learning_rate * grads['db'][i])
    
    def _bgd(self, grads):
        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * grads['dW'][i] / self.input_size)
            self.biases[i] -= (self.learning_rate * grads['db'][i] / self.input_size)

    def _mbgd(self, grads):
        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * grads['dW'][i] / grads['dW'][i].shape[1])
            self.biases[i] -= (self.learning_rate * grads['db'][i] / grads['db'][i].shape[1])
    
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
    
    def _forward_propagation(self, X):
        num_layers = len(self.weights)
        A = X
        caches = []
        
        for i in range(num_layers):
            W = self.weights[i]
            b = self.biases[i]
            Z = np.dot(A, W) + b
            
            caches.append((A, W, b, Z))

            A = self.activation_func(Z)
        
        return A, caches

    def _backward_propagation(self, A, Y, caches):
        num_layers = len(self.weights)
        grads = {'dW': [], 'db': []}

        delta = A-Y
        for i in reversed(range(num_layers)):
            A, W, _, Z = caches[i]
            dZ = np.multiply(delta, self._activation_derivative(Z))
            if dZ.ndim == 1:
                dZ = dZ.reshape((dZ.shape[0], 1))
            dW = np.dot(A.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            delta = np.dot(dZ, W.T)
            if len(dW.shape) == 1:
                dW = dW.reshape(-1, 1)

            grads['dW'].append(dW)
            grads['db'].append(db)
        
        grads['dW'].reverse()
        grads['db'].reverse()

        return grads
    
    def predict(self, X):
        A, _ = self._forward_propagation(X)
        A = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)
        return np.argmax(A, axis=1)
    
    def calculate_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def _one_hot_encode(self, Y, num_classes):
        return np.eye(num_classes)[Y]

    def _calculate_cost(self, A, Y):
        m = Y.shape[0]
        epsilon = 1e-15  # Small value to avoid log(0)
        
        # print(f"Shape of A: {A.shape}")
        # print(f"Shape of Y: {Y.shape}")
        
        # Ensure A has the same number of samples as Y
        if A.shape[0] != m:
            A = A[:m]
        
        # Apply softmax to get probabilities
        A = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)
        
        cross_entropy = -np.sum(Y * np.log(A + epsilon)) / m
        return cross_entropy

    def fit(self, X, Y, max_epochs=10, batch_size=32, X_validation=None, y_validation=None, early_stopping=False, patience=100):
        num_samples = X.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        costs = []
        
        print(f"Shape of X: {X.shape}")
        print(f"Shape of Y: {Y.shape}")
        print(f"Unique values in Y: {np.unique(Y)}")
        
        Y_one_hot = self._one_hot_encode(Y, self.output_size)
        
        print(f"Shape of Y_one_hot: {Y_one_hot.shape}")
        
        for i in range(max_epochs):
            if self.optimizer == "bgd":
                batch_size = num_samples
                num_batches = 1
            elif self.optimizer == "sgd":
                batch_size = 1
                num_batches = num_samples
            elif self.optimizer == "mbgd":
                num_batches = num_samples // batch_size
            else:
                raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

            for j in range(num_batches):
                start = j * batch_size
                end = start + batch_size
                
                A, caches = self._forward_propagation(X[start:end])
                grads = self._backward_propagation(A, Y_one_hot[start:end], caches)
                self.optimizer_func(grads)
            
            A, _ = self._forward_propagation(X)
            cost = self._calculate_cost(A, Y_one_hot)
            costs.append(cost)

            data_to_log = {
                "epoch": i + 1,
                "train_loss": cost
            }

            if X_validation is not None and y_validation is not None:
                A_val, _ = self._forward_propagation(X_validation)
                y_validation_one_hot = self._one_hot_encode(y_validation, self.output_size)
                val_loss = self._calculate_cost(A_val, y_validation_one_hot)
                data_to_log["val_loss"] = val_loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if early_stopping and patience_counter > patience:
                    print(f"Early stopping at epoch {i+1}")
                    break

            if self.wandb_log:
                wandb.log(data_to_log)
            
            if self.print_every and (i+1) % self.print_every == 0:
                print(f"Cost after {i+1} epochs: {cost}")
        
        return costs
    
    
    
    # def _calculate_cost(self, A, Y):
    #     print("This is A")
    #     print(A)
    #     print("This is Y")
    #     print(Y)
    #     cost = np.mean(np.not_equal(A, Y))
    #     return cost

    # def predict(self, X):
    #     A, _ = self._forward_propagation(X)
    #     A = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)

    #     A = np.argmax(A,axis=1)
    #     return A

    # def _one_hot_encode(self, Y, num_classes):
    #     print(Y.size)
    #     one_hot = np.zeros((Y.size, num_classes))
    #     print(one_hot)
    #     one_hot[np.arange(Y.size), Y] = 1
    #     return one_hot

    
    # def gradient_check(self, X, Y, epsilon=1e-7):
    #     A, caches = self._forward_propagation(X)
    #     grads = self._backward_propagation(A, Y, caches)

    #     for i in range(len(self.weights)):
    #         W = self.weights[i]
    #         dW_approx = np.zeros_like(W)
    #         for j in range(W.shape[0]):
    #             for k in range(W.shape[1]):
    #                 W_plus = W.copy()
    #                 W_plus[j, k] += epsilon
    #                 W_minus = W.copy()
    #                 W_minus[j, k] -= epsilon

    #                 self.weights[i] = W_plus
    #                 A_plus, _ = self._forward_propagation(X)
    #                 cost_plus = self._calculate_cost(A_plus, Y)

    #                 self.weights[i] = W_minus
    #                 A_minus, _ = self._forward_propagation(X)
    #                 cost_minus = self._calculate_cost(A_minus, Y)

    #                 dW_approx[j, k] = (cost_plus - cost_minus) / (2 * epsilon)

    #         difference = np.linalg.norm(grads['dW'][i] - dW_approx) / (np.linalg.norm(grads['dW'][i]) + np.linalg.norm(dW_approx))
    #         if difference > epsilon:
    #             print(f"Gradient check failed for layer {i} with difference {difference}")
    #         else:
    #             print(f"Gradient check passed for layer {i}")
    #     self.weights = caches

    
    # def fit(self, X, Y, max_epochs=10, batch_size=32, X_validation=None, y_validation=None, early_stopping=False, patience=1000):
    #     num_samples = X.shape[0]
    #     best_loss = float('inf')
    #     patience_counter = 0
    #     costs = []
    #     shift = np.min(Y)
    #     Y= Y- shift
    #     y_validation = shift
    #     y_new = self._one_hot_encode(Y, self.output_size)
        
    #     for i in range(max_epochs):
    #         if self.optimizer == "bgd":
    #             batch_size = num_samples
    #             num_batches = 1
    #         elif self.optimizer == "sgd":
    #             batch_size = 1
    #             num_batches = num_samples
    #         elif self.optimizer == "mbgd":
    #             num_batches = num_samples // batch_size
    #         else:
    #             raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

    #         for j in range(num_batches):
    #             start = j * batch_size
    #             end = start + batch_size
                
    #             A, caches = self._forward_propagation(X[start:end])
    #             grads = self._backward_propagation(A, y_new[start:end], caches)
    #             self.optimizer_func(grads)
            
    #         A = self.predict(X)
    #         cost = self._calculate_cost(A, Y)
    #         costs.append(cost)

    #         data_to_log = {
    #             "epoch": i + 1,
    #             "train_loss": cost
    #         }

    #         if X_validation is not None and y_validation is not None:
    #             A = self.predict(X_validation)
    #             val_loss = self._calculate_cost(A, y_validation)
    #             data_to_log["val_loss"] = val_loss
    #             if val_loss < best_loss:
    #                 best_loss = val_loss
    #                 patience_counter = 0
    #             else:
    #                 patience_counter += 1

    #             if early_stopping and patience_counter > patience:
    #                 print(f"Early stopping at epoch {i+1}")
    #                 break


    #         if self.wandb_log:
    #             wandb.log(data_to_log)
            
    #         if self.print_every and (i+1) % self.print_every == 0:
    #             print(f"Cost after {i+1} epochs: {cost}")

            
        
    #     return costs
