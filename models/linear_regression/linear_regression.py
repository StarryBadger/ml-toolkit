import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.03, iterations=2000, regularization = None, lamda = 0):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.lamda = lamda


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
        return self._predict(X)
    
    def save_model(self, file_path):
        model_params = {
            'degree': [self.degree],
            'weights': [self.weights],
            'bias': [self.bias]
        }
        df = pd.DataFrame(model_params)
        df.to_csv(file_path, index=False)

    def load_model(self, file_path):
        df = pd.read_csv(file_path)
        self.degree = int(df['degree'].iloc[0])
        self.weights = np.array(eval(df['weights'].iloc[0]))
        self.bias = df['bias'].iloc[0]

    def fit_with_gif(self, X, y):
        X_poly = self._polynomial_features(X)
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        frames = []
        for iteration in range(1,self.iterations+1):

            y_pred = self._predict(X)
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            if self.regularization == 'l1':
                dw += self.lamda * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += self.lamda * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if iteration % 100 == 0 or iteration == self.iterations - 1:
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

                # Plot MSE
                axs[0, 1].bar(['MSE'], [mse], color='green')
                axs[0, 1].set_ylim(0, max(mse, 0.4))
                axs[0, 1].set_title('Mean Squared Error')

                # Plot Standard Deviation
                axs[1, 0].bar(['Std Dev'], [std_dev], color='orange')
                axs[1, 0].set_ylim(0, max(std_dev, 0.4))
                axs[1, 0].set_title('Standard Deviation')

                # Plot Variance
                axs[1, 1].bar(['Variance'], [variance], color='purple')
                axs[1, 1].set_ylim(0, max(variance, 0.4))
                axs[1, 1].set_title('Variance')
                plt.savefig(f'assignments/1/figures/gifs/temp_images/frame_{iteration}.png')
                plt.close()
                frames.append(imageio.imread(f'assignments/1/figures/gifs/temp_images/frame_{iteration}.png'))
        imageio.mimsave(f'assignments/1/figures/gifs/{self.degree}_polynomial_fitting.gif', frames, fps=5)
        # print(f"GIF saved at assignments/1/figures/{self.degree}_polynomial_fitting.gif")


    