# BedrockML: A Toolkit of Scratch-Built Machine Learning Algorithms

Welcome to BedrockML! This repository is a comprehensive collection of machine learning algorithms implemented from scratch using Python. The primary goal of this project is to provide a deep, practical understanding of the inner workings of these algorithms, moving beyond the black-box usage of common libraries. Each algorithm is accompanied by demonstrations on various datasets.

## Key Features

*   **From-Scratch Implementations:** Core logic of numerous ML algorithms built using fundamental libraries like NumPy, with PyTorch used for more complex neural network architectures where appropriate.
*   **Broad Coverage:** Includes algorithms for classification, regression, clustering, dimensionality reduction, density estimation, and sequential data modeling.
*   **Modular Design:** Each algorithm is encapsulated in its own class, for reusability and understanding.
*   **Performance Metrics:** A dedicated module for evaluating model performance for both classification and regression tasks.
*   **Practical Demonstrations:** The `demonstrations/` directory showcases the application of these algorithms on diverse datasets and tasks, illustrating their capabilities and usage.

## Directory Structure

```
.
|-- README.md
|-- demonstrations               
|   |-- 1_knn_linear_regression
|   |   |-- README.md
|   |   |-- 1.py
|   |-- 2_clustering_pca
|   |   |-- ...
|   |-- 3_mlp_autoencoders
|   |   |-- ...
|   |-- 4_cnn_advanced_autoencoders
|   |   |-- ...
|   `-- 5_kde_hmm_rnn_ocr
|       |-- ...
|-- data
|   |-- README.md
|   |-- external            # Raw, original datasets
|   |   |-- ...
|   `-- interim             # Processed or intermediate datasets
|       |-- 1
|       |-- ...
|-- models
|   |-- README.md
|   |-- AutoEncoders
|   |   |-- AutoEncoders.py         # MLP-based AutoEncoder
|   |   |-- cnn_autoencoder.py      # CNN-based AutoEncoder
|   |   `-- pca_autoencoder.py      # PCA-based AutoEncoder
|   |-- MLP
|   |   |-- MLP.py                  # Unified MLP for classification/regression
|   |   |-- MLPClassifier.py
|   |   |-- MLPLogistic.py
|   |   |-- MLPRegression.py
|   |   `-- MultiLabelMLP.py
|   |-- cnn
|   |   |-- cnn.py                  # CNN for classification/regression
|   |   `-- multilabel_cnn.py       # CNN for multi-label classification
|   |-- gmm
|   |   `-- gmm.py                  # Gaussian Mixture Models
|   |-- k_means
|   |   `-- k_means.py              # K-Means Clustering
|   |-- kde
|   |   `-- kde.py                  # Kernel Density Estimation
|   |-- knn
|   |   `-- knn.py                  # K-Nearest Neighbors
|   |-- linear_regression
|   |   `-- linear_regression.py    # Polynomial Regression
|   |-- ocr
|   |   `-- ocr.py                  # CNN+RNN OCR Model
|   |-- pca
|   |   `-- pca.py                  # Principal Component Analysis
|   `-- rnn
|       |-- best_model.pth
|       |-- generalization_plot.png
|       `-- rnn.py                  # RNN for bit counting
|-- performance_measures
|   |-- README.md
|   `-- classification_metrics.py # Custom metrics calculator
```

## Implemented Algorithms (Models)

This section details the core algorithms implemented in the `models/` directory.

### 1. K-Nearest Neighbors (KNN)
*   **File:** `models/knn/knn.py`
*   **Description:** A non-parametric, instance-based learning algorithm for classification. It classifies a new data point based on the majority class of its 'k' nearest neighbors.
*   **Class:** `KNN`
*   **Core Methods:**
    *   `__init__(self, k: int, distance_metric: str = "euclidean")`: Initializes the KNN classifier.
    *   `fit(self, X, y)`: Stores the training data `X` and labels `y`.
    *   `predict(self, X_test)`: Predicts labels for the test data `X_test`.
*   **Key Parameters:**
    *   `k`: Number of neighbors to consider.
    *   `distance_metric`: Metric for distance calculation (e.g., "euclidean", "manhattan", "cosine").
*   **Features:** Vectorized distance calculations for efficiency.

### 2. Polynomial Linear Regression
*   **File:** `models/linear_regression/linear_regression.py`
*   **Description:** Implements polynomial regression, fitting a polynomial equation to the data. Supports L1 (Lasso) and L2 (Ridge) regularization.
*   **Class:** `PolynomialRegression`
*   **Core Methods:**
    *   `__init__(self, degree, learning_rate=0.03, iterations=2000, regularization=None, lamda=0)`: Initializes the model.
    *   `fit(self, X, y)`: Trains the model using gradient descent.
    *   `predict(self, X)`: Makes predictions on new data.
    *   `save_model(self, file_path)` / `load_model(self, file_path)`: Saves/loads model parameters.
    *   `fit_with_gif(self, X, y)`: Fits the model and generates a GIF visualizing the fitting process.
*   **Key Parameters:**
    *   `degree`: Degree of the polynomial.
    *   `learning_rate`, `iterations`: For gradient descent.
    *   `regularization`: Type of regularization ('l1' or 'l2').
    *   `lamda`: Regularization strength.

### 3. K-Means Clustering
*   **File:** `models/k_means/k_means.py`
*   **Description:** An unsupervised clustering algorithm that partitions data into 'k' clusters by minimizing the within-cluster sum of squares (WCSS).
*   **Class:** `KMeans`
*   **Core Methods:**
    *   `__init__(self, k, iteration_lim=300, tolerance=1e-4)`: Initializes the K-Means model.
    *   `fit(self, X)`: Computes cluster centroids.
    *   `predict(self, X)`: Assigns data points to the nearest cluster.
    *   `getCost(self, X)`: Calculates the WCSS (inertia).
*   **Key Parameters:**
    *   `k`: Number of clusters.
    *   `iteration_lim`, `tolerance`: Convergence criteria.

### 4. Gaussian Mixture Models (GMM)
*   **File:** `models/gmm/gmm.py`
*   **Description:** A probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions with unknown parameters. Uses the Expectation-Maximization (EM) algorithm for fitting.
*   **Class:** `GMM`
*   **Core Methods:**
    *   `__init__(self, k, iteration_lim=100, tolerance=1e-3, reg_covar=1e-6)`: Initializes the GMM.
    *   `fit(self, X)`: Estimates model parameters using the EM algorithm.
    *   `getParams()`: Returns weights, means, and covariances.
    *   `getMembership(self, X)`: Returns posterior probabilities of cluster membership.
    *   `getLikelihood(self, X)`: Computes the log-likelihood of the data.
    *   `getHardAssignments(self, X)`: Assigns data points to the most likely cluster.
    *   `aic()` / `bic()`: Computes Akaike Information Criterion and Bayesian Information Criterion.
*   **Key Parameters:**
    *   `k`: Number of Gaussian components.
    *   `reg_covar`: Regularization added to the diagonal of covariance matrices for stability.

### 5. Principal Component Analysis (PCA)
*   **File:** `models/pca/pca.py`
*   **Description:** A dimensionality reduction technique that transforms data into a new set of orthogonal variables (principal components) ordered by their variance.
*   **Class:** `PCA`
*   **Core Methods:**
    *   `__init__(self, n_components)`: Initializes PCA.
    *   `fit(self, X)`: Computes principal components from the training data.
    *   `transform(self)`: Applies dimensionality reduction to the data.
    *   `checkPCA()`: Verifies the PCA implementation by checking reconstruction error and explained variance.
*   **Key Parameters:**
    *   `n_components`: Number of principal components to keep.

### 6. Multi-Layer Perceptron (MLP)
*   **Files:**
    *   `models/MLP/MLP.py`: A unified MLP class capable of both classification and regression.
    *   `models/MLP/MLPClassifier.py`: Specialized MLP for classification tasks.
    *   `models/MLP/MLPRegression.py`: Specialized MLP for regression tasks.
    *   `models/MLP/MLPLogistic.py`: MLP for logistic regression (binary classification).
    *   `models/MLP/MultiLabelMLP.py`: MLP for multi-label classification.
*   **Description:** Implements feedforward neural networks (MLPs) from scratch. Supports various activation functions, optimizers, and options for early stopping and Weights & Biases logging.
*   **General Class Structure (e.g., `MLPClassifier`, `MLPRegression`):**
    *   `__init__(...)`: Initializes network architecture, learning rate, activation, optimizer, etc.
    *   `fit(self, X_train, y_train, ...)`: Trains the network using backpropagation and gradient descent.
    *   `predict(self, X)`: Makes predictions.
    *   `forward_propagation(self, X)`: Computes the output of the network.
    *   `backpropagation(self, X, y, y_pred)`: Computes gradients.
    *   `update_parameters()`: Updates weights and biases.
    *   `_compute_loss(...)`: Calculates the loss (e.g., Cross-Entropy for classification, MSE for regression).
    *   `gradient_checking(...)`: (Optional) Verifies gradient computations.
*   **Key Parameters:**
    *   `input_size`, `hidden_layers`, `output_size` (or `num_classes`).
    *   `learning_rate`, `activation`, `optimizer`.
    *   `task` (for unified `MLP.py`): 'classification' or 'regression'.
    *   `wandb_log`: Boolean to enable/disable Weights & Biases logging.

### 7. AutoEncoders
*   **Files:**
    *   `models/AutoEncoders/AutoEncoders.py`: MLP-based AutoEncoder.
    *   `models/AutoEncoders/cnn_autoencoder.py`: Convolutional AutoEncoder (PyTorch-based).
    *   `models/AutoEncoders/pca_autoencoder.py`: AutoEncoder using PCA principles.
*   **Description:**
    *   **MLP AutoEncoder (`AutoEncoder` class):** Uses two MLPRegression instances (one for encoder, one for decoder) to learn a compressed representation of data.
        *   `__init__(...)`: Initializes encoder and decoder MLPs.
        *   `fit(self, X_train, ...)`: Trains the autoencoder.
        *   `get_latent(self, X)`: Returns the encoded (latent) representation.
        *   `reconstruct(self, X)`: Returns the reconstructed data from input X.
    *   **CNN AutoEncoder (`CNNAutoencoder` class):** Uses convolutional and transpose convolutional layers (PyTorch) for encoding and decoding, suitable for image data.
        *   `__init__(...)`: Defines encoder and decoder architectures.
        *   `encode(self, x)`, `decode(self, z)`, `forward(self, x)`.
        *   `fit(self, train_loader, ...)`: Trains the CNN autoencoder.
        *   `plot_losses()`: Plots training/validation loss curves.
    *   **PCA AutoEncoder (`PcaAutoencoder` class):** Implements autoencoding functionality based on PCA principles.
        *   `fit(self, X)`, `encode(self, X)`, `forward(self, X)` (reconstructs).

### 8. Convolutional Neural Networks (CNN)
*   **Files:**
    *   `models/cnn/cnn.py`: General CNN for image classification or regression (PyTorch-based).
    *   `models/cnn/multilabel_cnn.py`: CNN for multi-label image classification (PyTorch-based).
*   **Description:** Implements CNNs using PyTorch for image-based tasks.
*   **General Class Structure (e.g., `CNN`):**
    *   `__init__(self, task, num_classes, ...)`: Defines CNN architecture (conv layers, pooling, fully connected layers).
    *   `forward(self, x, ...)`: Defines the forward pass.
    *   `fit(self, train_loader, val_loader, ...)`: Trains the CNN.
    *   `evaluate(self, loader, criterion)`: Evaluates the model.
    *   `predict(self, loader)`: Makes predictions.
    *   `plot_loss(self, history)`: Plots training/validation loss.
*   **Key Parameters:**
    *   `task`: 'classification' or 'regression'.
    *   `num_classes`: For classification.
    *   `num_conv_layers`, `dropout_rate`, `optimizer_choice`, `activation_function`.


### 9. Kernel Density Estimation (KDE)
*   **File:** `models/kde/kde.py`
*   **Description:** A non-parametric way to estimate the probability density function of a random variable.
*   **Class:** `KDE`
*   **Core Methods:**
    *   `__init__(self, kernel='gaussian', bandwidth=1.0)`: Initializes KDE.
    *   `fit(self, X)`: Stores the data.
    *   `predict(self, x)`: Estimates density at point(s) `x`.
    *   `visualize()`: Plots the 2D density estimate.
*   **Key Parameters:**
    *   `kernel`: Type of kernel ('gaussian', 'box', 'triangular').
    *   `bandwidth`: Bandwidth of the kernel.

### 10. Recurrent Neural Networks (RNN) & OCR Model
*   **RNN for Bit Counting:**
    *   **File:** `models/rnn/rnn.py`
    *   **Description:** A simple RNN (PyTorch-based) designed to count the number of '1's in a binary input sequence.
    *   **Class:** `RNNBitCounter`
    *   `__init__(...)`: Defines the RNN architecture (RNN layer + fully connected layer).
    *   `forward(self, x)`: Processes the input sequence.
*   **OCR Model (CNN + RNN):**
    *   **File:** `models/ocr/ocr.py`
    *   **Description:** A model combining CNNs for feature extraction from images and RNNs for sequence prediction, aimed at Optical Character Recognition (PyTorch-based).
    *   **Class:** `OCRModel`
    *   `__init__(...)`: Defines the CNN encoder and RNN decoder.
    *   `forward(self, x)`: Processes an image and outputs character sequence predictions.

## Performance Measures

*   **File:** `performance_measures/classification_metrics.py`
*   **Description:** Provides a `Metrics` class to calculate various performance metrics for both classification and regression tasks.
*   **Class:** `Metrics(y_true, y_pred, task)`
*   **Classification Metrics:**
    *   `confusion_matrix()`
    *   `accuracy(one_hot=False)`
    *   `precision_score(average="macro"|"micro")`
    *   `recall_score(average="macro"|"micro")`
    *   `f1_score(average="macro"|"micro")`
    *   `hamming_loss()`
    *   `hamming_accuracy()`
*   **Regression Metrics:**
    *   `mse()` (Mean Squared Error)
    *   `rmse()` (Root Mean Squared Error)
    *   `mae()` (Mean Absolute Error)
    *   `standard_deviation()` (of errors)
    *   `variance()` (of errors)
    *   `r2_score()` (R-squared)
*   **Utility:** `print_metrics()` method to display relevant scores based on the task.

## Demonstrations & Use Cases (`demonstrations/` directory)

This directory contains scripts and notebooks demonstrating the usage of the implemented models on various datasets and tasks. Each sub-directory corresponds to a set of related problems or algorithms.

*   **`demonstrations/1_knn_linear_regression/`**
    *   **Description:** Demonstrates K-Nearest Neighbors for music genre prediction (Spotify dataset) including Exploratory Data Analysis (EDA), hyperparameter tuning, and performance optimization. Also showcases Polynomial Linear Regression for fitting curves to data, including degree 1 and higher-degree polynomials, animated fitting process, and regularization techniques.
    *   **Models Used:** `KNN`, `PolynomialRegression`.
    *   **Key Scripts:** `1.py`, `data_preprocessing_knn.py`, `data_preprocessing_linreg.py`.
    *   **Datasets:** `spotify.csv`, `linreg.csv`, `regularisation.csv`, `spotify-2/`.

*   **`demonstrations/2_clustering_pca/`**
    *   **Description:** Focuses on unsupervised learning. Implements K-Means and Gaussian Mixture Models (GMM) for clustering word embeddings. Demonstrates Principal Component Analysis (PCA) for dimensionality reduction and visualization. Compares clustering results and applies KNN (from example 1) on PCA-reduced Spotify data. Hierarchical clustering (using `scipy`) is also explored.
    *   **Models Used:** `KMeans`, `GMM`, `PCA`, `KNN`.
    *   **Key Scripts:** `2.py`, `hierarchial.py`.
    *   **Datasets:** `word-embeddings.feather`, `spotify.csv`.

*   **`demonstrations/3_mlp_autoencoders/`**
    *   **Description:** Introduces Multi-Layer Perceptrons (MLPs) and basic AutoEncoders. Includes MLP implementations for wine quality classification, Boston housing price regression, multi-label classification on an advertisement dataset, and a comparison of MSE vs. Binary Cross-Entropy loss for diabetes classification. An MLP-based AutoEncoder is used for dimensionality reduction on the Spotify dataset, followed by KNN/MLP classification.
    *   **Models Used:** `MLPClassifier`, `MLPRegression`, `MLP` (unified), `MultiLabelMLP`, `MLPLogistic`, `AutoEncoder` (MLP-based), `KNN`.
    *   **Key Scripts:** `3.py`.
    *   **Datasets:** `WineQT.csv`, `HousingData.csv` (Boston Housing), `advertisement.csv`, `diabetes.csv`, `spotify.csv`.

*   **`demonstrations/4_cnn_advanced_autoencoders/`**
    *   **Description:** Delves into Convolutional Neural Networks (CNNs) and more advanced AutoEncoder comparisons. Implements a CNN for predicting the number of digits in images from a modified Multi-MNIST dataset (both classification and regression approaches). Includes feature map visualization. A multi-label CNN is also built to predict all digits present in an image. Finally, a comparative analysis of CNN, MLP, and PCA-based autoencoders is performed on the Fashion MNIST dataset, with KNN used to classify the resulting latent features.
    *   **Models Used:** `CNN`, `MultiLabelCNN`, `CNNAutoencoder`, `AutoEncoder` (MLP-based from example 3), `PcaAutoencoder`, `KNN`.
    *   **Key Scripts:** `CNN.ipynb`, `AutoEncoder.ipynb`.
    *   **Datasets:** `double_mnist/`, Fashion MNIST (via Kaggle).

*   **`demonstrations/5_kde_hmm_rnn_ocr/`**
    *   **Description:** Explores density estimation and sequential models. Implements Kernel Density Estimation (KDE) and compares it with GMMs on synthetic data. Hidden Markov Models (HMMs, using `hmmlearn`) are used for speech digit recognition from the Free Spoken Digit Dataset. A Recurrent Neural Network (RNN) is trained to count bits in binary sequences. Lastly, a combined CNN-RNN model is developed for Optical Character Recognition (OCR) on images of words.
    *   **Models Used:** `KDE`, `GMM`, `RNNBitCounter`, `OCRModel`. (HMMs use `hmmlearn`).
    *   **Key Scripts:** `5.py`, `ocr.ipynb`.
    *   **Datasets:** Synthetic data for KDE, Free Spoken Digit Dataset (`fsdd/`), synthetic bit streams, `nltk.corpus.words` for OCR.

## Data

*   **`data/external/`**: Contains the raw, original datasets used in the demonstrations.
*   **`data/interim/`**: Intended for storing any transformed, cleaned, or intermediate datasets generated during preprocessing steps within the demonstrations. Each subfolder `1` through `5` corresponds to the respective example set.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ml-toolkit.git
    cd ml-toolkit
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    A `requirements.txt` file is present at the root of the project, consolidating all necessary packages. 
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

1.  **Import a model:**
    ```python
    from models.knn.knn import KNN
    from models.linear_regression.linear_regression import PolynomialRegression
    # ... and so on for other models
    ```
2.  **Instantiate the model:**
    ```python
    knn_classifier = KNN(k=5, distance_metric="euclidean")
    poly_reg = PolynomialRegression(degree=2)
    ```
3.  **Prepare your data:** Load and preprocess your data (often `X_train`, `y_train`).
4.  **Fit the model:**
    ```python
    knn_classifier.fit(X_train, y_train)
    poly_reg.fit(X_train_reg, y_train_reg)
    ```
5.  **Make predictions:**
    ```python
    predictions = knn_classifier.predict(X_test)
    reg_predictions = poly_reg.predict(X_test_reg)
    ```
6.  **Evaluate performance:**
    ```python
    from performance_measures.classification_metrics import Metrics
    
    knn_metrics = Metrics(y_test, predictions, task="classification")
    knn_metrics.print_metrics()
    
    reg_metrics = Metrics(y_test_reg, reg_predictions, task="regression")
    reg_metrics.print_metrics()
    ```

For detailed usage patterns and advanced configurations, please refer to the scripts and notebooks within the `demonstrations/` directory.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to a consistent style and includes documentation where necessary.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (You'll need to add a LICENSE file, MIT is a good default for open source projects).