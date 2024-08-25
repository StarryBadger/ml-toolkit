import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.knn.knn_suboptimal import KNN
from models.linear_regression.linear_regression import PolynomialRegression
from performance_measures.classification_metrics import Metrics

def visualization_spotify():
    df = pd.read_csv('data/interim/1/spotify/step1_drop_unnamed.csv')
    numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    categorical_features = ['explicit', 'mode', 'time_signature']

    # Histograms
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('Distribution of Numerical Features', fontsize=16)
    for i, feature in enumerate(numerical_features):
        row = i // 3
        col = i % 3
        axes[row, col].hist(df[feature], bins=30, edgecolor='black')
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Frequency')
    plt.savefig('assignments/1/figures/histograms.png')

    # Bar plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Distribution of Categorical Features', fontsize=16)
    for i, feature in enumerate(categorical_features):
        value_counts = df[feature].value_counts()
        axes[i].bar(value_counts.index.astype(str), value_counts.values)
        axes[i].set_title(feature)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    plt.savefig('assignments/1/figures/barplots.png')
    
    # Violin plots
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('Violin Plots of Numerical Features', fontsize=16)
    for i, feature in enumerate(numerical_features):
        row = i // 3
        col = i % 3
        axes[row, col].violinplot(df[feature])
        axes[row, col].set_title(feature)
        axes[row, col].set_xticks([])
    plt.tight_layout()
    plt.savefig('assignments/1/figures/violin_plots.png')
    plt.close()

    # Pair plots
    pd.plotting.scatter_matrix(df[numerical_features], figsize=(80, 80), diagonal='hist', alpha=0.08)
    plt.suptitle('Scatter Plot Matrix of Numerical Features', fontsize=30)
    plt.tight_layout()
    plt.savefig('assignments/1/figures/pair_plots.png')
    plt.close()

    # Correlation heatmap
    # ? https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap
    correlation_matrix = df[numerical_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(correlation_matrix, cmap='coolwarm')
    ax.set_xticks(range(len(numerical_features)))
    ax.set_yticks(range(len(numerical_features)))
    ax.set_xticklabels(numerical_features, rotation=45, ha='right')
    ax.set_yticklabels(numerical_features)
    plt.colorbar(im)
    plt.title('Correlation Heatmap of Numerical Features')
    for i in range(len(numerical_features)):
        for j in range(len(numerical_features)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig('assignments/1/figures/correlation_heatmap.png')
    plt.close()

    # Box plots
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('Box Plots of Numerical Features', fontsize=16)
    for i, feature in enumerate(numerical_features):
        row = i // 3
        col = i % 3
        axes[row, col].boxplot(df[feature])
        axes[row, col].set_title(feature)
        axes[row, col].set_xticks([])
    plt.tight_layout()
    plt.savefig('assignments/1/figures/box_plots.png')
    plt.close()

def knn_on_spotify() -> None:
    dataset_dir="data/interim/1/spotify/split"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"].drop(columns=["track_genre"]).to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()

    classifier = KNN(k=30, distance_metric="manhattan")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    Metrics(y_true=y_test, y_pred=y_pred, task='regression').print_metrics()

def regression() -> None:
    dataset_dir="data/interim/1/linreg"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"]['x'].to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]['y'].to_numpy()

    model = PolynomialRegression(degree=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validate)
    Metrics(y_true=y_validate, y_pred=y_pred, task='regression').print_metrics()

if __name__ == "__main__":
    start_time = time.time()
    visualization_spotify()
    # knn_on_spotify()
    # regression()
    time_taken = time.time() - start_time
    print(f"{time_taken=}")
