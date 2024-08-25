import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.knn.knn import KNN
from models.linear_regression.linear_regression import PolynomialRegression
from performance_measures.classification_metrics import Metrics

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
            globals()[f"data_{x}"].drop(columns=["track_genre"]).to_numpy()
        )
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()
    classifier = KNN(k=85, distance_metric="cosine")
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


if __name__ == "__main__":
    start_time = time.time()
    # visualization_spotify()
    # knn_on_spotify("data/interim/1/spotify/split")
    # hyperparam_tuning_knn()
    # knn_on_spotify("data/interim/1/spotify-2/final")
    # regression()
    # hyperparam_tuning_regression()
    animation()
    time_taken = time.time() - start_time
    print(f"{time_taken=}")
