import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.knn.knn import KNN
from models.linear_regression.linear_regression import PolynomialRegression
from performance_measures.classification_metrics import Metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def save_model_parameters(coef, filename):
    df = pd.DataFrame(coef)
    df.to_csv(filename, index=False, header=False)

def load_model_parameters(filename):
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()

def visualization_spotify():
    df = pd.read_csv("data/interim/1/spotify/step1_drop_unnamed.csv")
    numerical_features = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    categorical_features = ["explicit", "mode", "time_signature", "track_genre"]

    # Histograms
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle("Distribution of Numerical Features", fontsize=16)
    for i, feature in enumerate(numerical_features):
        row = i // 3
        col = i % 3
        axes[row, col].hist(df[feature], bins=30, edgecolor="black")
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("Frequency")
    plt.savefig("assignments/1/figures/histograms.png")

    # Bar plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Distribution of Categorical Features", fontsize=16)
    for i, feature in enumerate(categorical_features):
        value_counts = df[feature].value_counts()
        axes[i].bar(value_counts.index.astype(str), value_counts.values)
        axes[i].set_title(feature)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=45)
    plt.savefig("assignments/1/figures/barplots.png")

    # Violin plots
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle("Violin Plots of Numerical Features", fontsize=16)
    for i, feature in enumerate(numerical_features):
        row = i // 3
        col = i % 3
        axes[row, col].violinplot(df[feature])
        axes[row, col].set_title(feature)
        axes[row, col].set_xticks([])

    plt.savefig("assignments/1/figures/violin_plots.png")

    # Pair plots
    scatter = pd.plotting.scatter_matrix(
        df[numerical_features], figsize=(40, 40), diagonal="hist", alpha=0.15
    )
    for ax in scatter.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    plt.suptitle("Scatter Plot Matrix of Numerical Features", fontsize=25)

    plt.savefig("assignments/1/figures/pair_plots.png")

    # Correlation heatmap
    # ? https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap
    correlation_matrix = df[numerical_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(correlation_matrix, cmap="coolwarm")
    ax.set_xticks(range(len(numerical_features)))
    ax.set_yticks(range(len(numerical_features)))
    ax.set_xticklabels(numerical_features, rotation=45, ha="right")
    ax.set_yticklabels(numerical_features)
    plt.colorbar(im)
    plt.title("Correlation Heatmap of Numerical Features")
    for i in range(len(numerical_features)):
        for j in range(len(numerical_features)):
            text = ax.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.savefig("assignments/1/figures/correlation_heatmap.png")

    # Box plots
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle("Box Plots of Numerical Features", fontsize=16)
    for i, feature in enumerate(numerical_features):
        row = i // 3
        col = i % 3
        axes[row, col].boxplot(df[feature])
        axes[row, col].set_title(feature)
        axes[row, col].set_xticks([])

    plt.savefig("assignments/1/figures/box_plots.png")

    genre_mapping = {genre: idx for idx, genre in enumerate(df["track_genre"].unique())}
    df["genre_mapped"] = df["track_genre"].map(genre_mapping)

    # Scatter plots
    scatter_plots = [
        ("danceability", "energy"),
        ("danceability", "valence"),
        ("loudness", "energy"),
        ("acousticness", "instrumentalness"),
        ("tempo", "energy"),
    ]

    for x_feature, y_feature in scatter_plots:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df[x_feature], df[y_feature], c=df["genre_mapped"], alpha=0.1, cmap="magma"
        )
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(
            f"{x_feature.capitalize()} vs {y_feature.capitalize()} (Colored by Genre)"
        )
        plt.colorbar(scatter, label="Track Genre")
        plt.savefig(f"assignments/1/figures/scatter_{x_feature}_{y_feature}.png")

def knn_on_spotify(dataset_dir) -> None:
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = (
            globals()[f"data_{x}"].drop(columns=["track_genre", "tempo"]).to_numpy()
        )
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()
    classifier = KNN(k=85, distance_metric="manhattan")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_validate)
    Metrics(y_true=y_validate, y_pred=y_pred, task="classification").print_metrics()

def hyperparam_tuning_knn():
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{"data/interim/1/spotify/split"}/{x}.csv")
        globals()[f"X_{x}"] = (
            globals()[f"data_{x}"].drop(columns=["track_genre"]).to_numpy()
        )
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()

    # GP for values for k from 1 to root N
    k_values = [1, 2, 4, 8, 16, 32, 64,85, 128, 256]
    distance_metrics = ["euclidean", "manhattan", "cosine"]
    results = []
    for metric in distance_metrics:
        for k in k_values:        
            classifier = KNN(k=k, distance_metric=metric)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_validate)
            accuracy = Metrics(y_true=y_validate, y_pred=y_pred, task="classification").accuracy()
            results.append((k, metric, accuracy))

    results.sort(key=lambda x: x[2], reverse=True)
    print("Top 10 {k, distance_metric} pairs:")
    for i, (k, metric, accuracy) in enumerate(results[:10]):
        print(f"{i+1}. k={k}, metric={metric}, accuracy={accuracy:.4f}")

def k_vs_accuracy():
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{"data/interim/1/spotify/split"}/{x}.csv")
        globals()[f"X_{x}"] = (
            globals()[f"data_{x}"].drop(columns=["track_genre"]).to_numpy()
        )
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()

    k_values = [x for x in range(1,32,5)]
    metric = "manhattan"
    results = []
    for k in k_values:        
        classifier = KNN(k=k, distance_metric=metric)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_validate)
        accuracy = Metrics(y_true=y_validate, y_pred=y_pred, task="classification").accuracy()
        results.append((k, accuracy))
    
    k_vals, accuracies = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('K vs Accuracy')
    plt.grid(True)
    plt.xticks(k_vals)
    plt.savefig(f"assignments/1/figures/k_vs_accuracy.png")

def inference_time_plot():
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{"data/interim/1/spotify/split"}/{x}.csv")
        globals()[f"X_{x}"] = (
            globals()[f"data_{x}"].drop(columns=["track_genre"]).to_numpy()
        )
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()

    classifier = KNN(k=64, distance_metric="cosine")
    classifier.fit(X_train, y_train)
    start_time = time.time()
    y_pred = classifier.predict(X_validate)
    end_time = time.time()
    inference_time_optimal = end_time - start_time

    classifier = KNN(k=85, distance_metric="manhattan")
    classifier.fit(X_train, y_train)
    start_time = time.time()
    y_pred = classifier.predict(X_validate)
    end_time = time.time()
    inference_time_best = end_time - start_time

    sklearn_knn = KNeighborsClassifier() # using default
    sklearn_knn.fit(X_train, y_train)
    start_time = time.time()
    y_pred_sklearn = sklearn_knn.predict(X_validate)
    end_time = time.time()
    sklearn_inference_time = end_time - start_time

    classifier = KNN(k=85, distance_metric="manhattan")
    classifier.fit(X_train, y_train)
    start_time = time.time()
    y_pred = classifier.predict(X_validate)
    end_time = time.time()
    inference_time_best = end_time - start_time

    labels = ['sklearn KNN', 'Best KNN', 'Optimal KNN']
    times = [sklearn_inference_time, inference_time_best, inference_time_optimal]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, times, color=['blue', 'green', 'red'])
    plt.title('Inference Time Comparison')
    plt.xlabel('Model')
    plt.ylabel('Inference Time (seconds)')
    plt.savefig(f"assignments/1/figures/inference_time_plot.png")
    plt.close()

def inference_time_vs_train_size_plot():
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"data/interim/1/spotify/split/{x}.csv")
        globals()[f"X_{x}"] = (
            globals()[f"data_{x}"].drop(columns=["track_genre"]).to_numpy()
        )
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()

    dataset_sizes = np.linspace(0.2, 1.0, 4)
    sklearn_times = []
    best_knn_times = []
    optimal_knn_times = []

    for size in dataset_sizes:
        subset_size = int(size * len(X_train))
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]

        # Optimal KNN
        classifier_optimal = KNN(k=32, distance_metric="cosine")
        classifier_optimal.fit(X_train_subset, y_train_subset)
        start_time = time.time()
        y_pred_optimal = classifier_optimal.predict(X_validate)
        end_time = time.time()
        optimal_knn_times.append(end_time - start_time)

        # Best KNN
        classifier_best = KNN(k=85, distance_metric="manhattan")
        classifier_best.fit(X_train_subset, y_train_subset)
        start_time = time.time()
        y_pred_best = classifier_best.predict(X_validate)
        end_time = time.time()
        best_knn_times.append(end_time - start_time)

        # Sklearn KNN
        sklearn_knn = KNeighborsClassifier()  # using default
        sklearn_knn.fit(X_train_subset, y_train_subset)
        start_time = time.time()
        y_pred_sklearn = sklearn_knn.predict(X_validate)
        end_time = time.time()
        sklearn_times.append(end_time - start_time)
       

    plt.figure(figsize=(10, 8))
    plt.plot(dataset_sizes, sklearn_times, label='Sklearn KNN', marker='o', color='blue')
    plt.plot(dataset_sizes, best_knn_times, label='Best KNN', marker='o', color='green')
    plt.plot(dataset_sizes, optimal_knn_times, label='Optimal KNN', marker='o', color='red')

    plt.title('Inference Time vs Train Dataset Size')
    plt.xlabel('Fraction of Training Data Used')
    plt.ylabel('Inference Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"assignments/1/figures/inference_time_vs_train_size_plot.png")
    plt.close()

def visualize_linreg() -> None:
    dataset_dir = "data/interim/1/linreg"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"]["x"].to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["y"].to_numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Train', alpha=0.3)
    plt.scatter(X_validate, y_validate, color='green', label='Validate', alpha=0.5)
    plt.scatter(X_test, y_test, color='red', label='Test', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Train, Validation, and Test Splits')
    plt.legend()
    plt.savefig(f"assignments/1/figures/first_dataset_separate_visualisation.png")
    plt.close()

def regression() -> None:
    dataset_dir = "data/interim/1/linreg"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"]["x"].to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["y"].to_numpy()

    model = PolynomialRegression(degree=1, learning_rate=0.03)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validate)
    Metrics(y_true=y_validate, y_pred=y_pred, task="regression").print_metrics()
    w=model.weights[0]
    b=model.bias
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Train', alpha=0.3)
    x_line = np.linspace(-1.2, 1.2, 4)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Line: y = {w}x + {b}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Train, Validation, and Test Splits')
    plt.legend()
    plt.savefig(f"assignments/1/figures/linreg.png")
    plt.close()

    


def hyperparam_tuning_regression() -> None:
    dataset_dir = "data/interim/1/linreg"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"]["x"].to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["y"].to_numpy()

    degrees = [x for x in range (1,35)]
    best_k = None
    best_mse = float('inf')

    for degree in degrees:
        model = PolynomialRegression(degree=degree)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print(f"Train with k={degree}")
        Metrics(y_true=y_train, y_pred=y_pred_train, task="regression").print_metrics()
        metrics_val = Metrics(y_true=y_test, y_pred=y_pred_test, task="regression")
        print(f"Test with k={degree}")
        metrics_val.print_metrics()
        mse=metrics_val.mse()

        if mse < best_mse:
            best_mse = mse
            best_k = degree
            best_model = model

    print(f"Best degree (k): {best_k} with MSE: {best_mse}")
    best_model.save_model('assignments/1/best_model_params.csv')

def animation()-> None:
    dataset_dir = "data/interim/1/linreg"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"]["x"].to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["y"].to_numpy()
    for k in [2,5,10,15,23,32]:
        model = PolynomialRegression(degree=k, learning_rate=0.03)
        model.fit_with_gif(X_train, y_train)

def experiment_with_regularization():
    dataset_dir = "data/interim/1/regularisation"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"]["x"].to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["y"].to_numpy()

    degrees = list(range(1, 21))

    for reg in [None, 'l1', 'l2']:
        reg_label = 'None' if reg is None else reg
        fig, axes = plt.subplots(len(degrees), 1, figsize=(10, 5 * len(degrees)), sharex=True)
        for idx, degree in enumerate(degrees):
            ax = axes[idx] 
            model = PolynomialRegression(degree=degree, regularization=reg, lamda=0.01)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            ax.scatter(X_test, y_test, color='blue', label='True Values')

            sorted_indices = np.argsort(X_test)
            X_sorted = X_test[sorted_indices]
            y_pred_sorted = y_pred[sorted_indices]

            ax.plot(X_sorted, y_pred_sorted, color='red', label=f'Predicted (Degree {degree})')
            
            metrics = Metrics(y_true=y_test, y_pred=y_pred, task="regression")
            print(f"{reg_label} regularization on k={degree}")
            metrics.print_metrics()
            ax.set_title(f'Degree {degree} with {reg} Regularization')
            ax.set_ylabel("y")
            ax.legend()

        fig.text(0.5, 0.04, 'X_test', ha='center')
        plt.savefig(f"assignments/1/figures/polynomial_regression_{reg_label}.png")

if __name__ == "__main__":
    start_time = time.time()
    # visualization_spotify()
    knn_on_spotify("data/interim/1/spotify/split")
    # hyperparam_tuning_knn()
    # k_vs_accuracy()
    # inference_time_plot()
    # inference_time_vs_train_size_plot()
    knn_on_spotify("data/interim/1/spotify-2/final")
    
    # visualize_linreg()
    regression()
    # hyperparam_tuning_regression()
    # animation()
    # experiment_with_regularization()
    time_taken = time.time() - start_time
    print(f"{time_taken=}")
