import numpy as np
import pandas as pd
import time
from models.knn.knn_suboptimal import KNN
from performance_measures.classification_metrics import Metrics

data_train = pd.read_csv("data/interim/spotify/split/train.csv")
data_test = pd.read_csv("data/interim/spotify/split/test.csv")
data_val = pd.read_csv("data/interim/spotify/split/val.csv")

X_train = data_train.drop(columns=["track_genre"]).to_numpy()
X_test = data_test.drop(columns=["track_genre"]).to_numpy()

y_train = data_train["track_genre"].to_numpy()
y_test = data_test["track_genre"].to_numpy()

start_time = time.time()
classifier = KNN(k=30, distance_metric="manhattan")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
metrics = Metrics(y_true=y_test, y_pred=y_pred)
print(f"Accuracy: {metrics.accuracy():.2f}")
print(f"Time: {time.time() - start_time}")
