import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

class PolynomialRegression:
    """
    A class to perform polynomial regression with options for L1 and L2 regularization.
    It also supports saving/loading models and generating GIFs of the fitting process.

    Attributes:
        degree (int): The degree of the polynomial to be fitted.
        learning_rate (float): The learning rate for gradient descent.
        iterations (int): The number of iterations for gradient descent.
        weights (np.ndarray): The weights of the polynomial model.
        bias (float): The bias term of the polynomial model.
        regularization (str or None): The type of regularization ('l1' or 'l2') if any.
        lamda (float): The regularization parameter.
    """
    def __init__(self, degree, learning_rate=0.03, iterations=2000, regularization = None, lamda = 0):
        """
        Initializes the PolynomialRegression model.

        Args:
            degree (int): The degree of the polynomial.
            learning_rate (float): The learning rate for gradient descent. Default is 0.03.
            iterations (int): The number of iterations for gradient descent. Default is 2000.
            regularization (str or None): The type of regularization ('l1' or 'l2'). Default is None.
            lamda (float): The regularization parameter. Default is 0.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.lamda = lamda


    def _polynomial_features(self, X):
        """
        Generates polynomial features for input data X.

        Args:
            X (np.ndarray): The input feature vector.

        Returns:
            np.ndarray: A matrix with polynomial features up to the specified degree.
        """
        return np.column_stack([X**i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        """
        Fits the polynomial regression model to the input data X and target y.

        Args:
            X (np.ndarray): The input feature vector.
            y (np.ndarray): The target vector.
        """
        X_poly = self._polynomial_features(X)
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iterations):
            y_pred = self._predict(X)
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            if self.regularization == 'l1':
                dw += self.lamda * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += self.lamda * self.weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # private predict
    def _predict(self, X):
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias

    def predict(self, X):
        """
        Public method to make predictions using the polynomial model.

        Args:
            X (np.ndarray): The input feature vector.

        Returns:
            np.ndarray: The predicted values.
        """
        return self._predict(X)
    
    def save_model(self, file_path):
        """
        Saves the model parameters (degree, weights, bias) to a CSV file.

        Args:
            file_path (str): The file path to save the model.
        """
        model_params = {
            'degree': [self.degree],
            'weights': [self.weights],
            'bias': [self.bias]
        }
        df = pd.DataFrame(model_params)
        df.to_csv(file_path, index=False)

    def load_model(self, file_path):
        """
        Loads the model parameters (degree, weights, bias) from a CSV file.

        Args:
            file_path (str): The file path to load the model from.
        """
        df = pd.read_csv(file_path)
        self.degree = int(df['degree'].iloc[0])
        self.weights = np.array(eval(df['weights'].iloc[0]))
        self.bias = df['bias'].iloc[0]

    def fit_with_gif(self, X, y):
        """
        Fits the polynomial regression model and generates a GIF of the fitting process.

        Args:
            X (np.ndarray): The input feature vector.
            y (np.ndarray): The target vector.

        Saves:
            GIF: A GIF of the fitting process saved at the specified path.
        """
        X_poly = self._polynomial_features(X)
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        frames = []
        for iteration in range(0,self.iterations+1):

            y_pred = self._predict(X)
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            if self.regularization == 'l1':
                dw += self.lamda * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += self.lamda * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if iteration % 50 == 0 or iteration == self.iterations - 1:
                mse = np.mean((y - y_pred) ** 2)
                std_dev = np.sqrt(np.var(y - y_pred))
                variance = np.var(y - y_pred)
                
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                sorted_indices = np.argsort(X)
                X_sorted = X[sorted_indices]
                y_pred_sorted = y_pred[sorted_indices]
                axs[0, 0].scatter(X, y, color='blue', label='Original Data', alpha=0.2)
                axs[0, 0].plot(X_sorted, y_pred_sorted, color='red', label=f'Fitted Line (k={self.degree})')
                axs[0, 0].legend()
                axs[0, 0].set_title('Polynomial Fit')

                # MSE
                axs[0, 1].bar(['MSE'], [mse], color='green')
                axs[0, 1].set_ylim(0, max(mse, 0.4))
                axs[0, 1].set_title('Mean Squared Error')

                # Standard Deviation
                axs[1, 0].bar(['Std Dev'], [std_dev], color='orange')
                axs[1, 0].set_ylim(0, max(std_dev, 0.4))
                axs[1, 0].set_title('Standard Deviation')

                # Variance
                axs[1, 1].bar(['Variance'], [variance], color='purple')
                axs[1, 1].set_ylim(0, max(variance, 0.4))
                axs[1, 1].set_title('Variance')
                plt.savefig(f'assignments/1/figures/gifs/temp_images/frame_{iteration}.png')
                plt.close()
                frames.append(imageio.imread(f'assignments/1/figures/gifs/temp_images/frame_{iteration}.png'))
        imageio.mimsave(f'assignments/1/figures/gifs/{self.degree}_polynomial_fitting.gif', frames, fps=5)
        print(f"GIF saved at assignments/1/figures/{self.degree}_polynomial_fitting.gif")


    