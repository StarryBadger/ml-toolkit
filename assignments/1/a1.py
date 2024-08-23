import time
import pandas as pd
from models.knn.knn_suboptimal import KNN
from performance_measures.classification_metrics import Metrics


def main() -> None:
    dataset_dir="data/interim/spotify/split"
    for x in {"train", "test", "validate"}:
        globals()[f"data_{x}"] = pd.read_csv(f"{dataset_dir}/{x}.csv")
        globals()[f"X_{x}"] = globals()[f"data_{x}"].drop(columns=["track_genre"]).to_numpy()
        globals()[f"y_{x}"] = globals()[f"data_{x}"]["track_genre"].to_numpy()

    classifier = KNN(k=30, distance_metric="manhattan")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics = Metrics(y_true=y_test, y_pred=y_pred)
    print(f"Accuracy: {metrics.accuracy():.2f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_taken = time.time() - start_time
    print(f"{time_taken=}")
