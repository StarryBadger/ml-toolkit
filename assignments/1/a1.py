import numpy as np
import pandas as pd
from models.knn.knn_suboptimal import KNN
from performance_measures.classification_metrics import Metrics

data = pd.read_csv('data/external/spotify.csv')
X = data.drop(columns=['track_id', 'artists', 'album_name', 'track_name', 'track_genre', 'explicit'])
# ? https://www.geeksforgeeks.org/how-to-drop-unnamed-column-in-pandas-dataframe/
X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
# ? https://www.w3schools.com/python/pandas/ref_df_drop_duplicates.asp
# X=X.drop_duplicates()
y = data['track_genre']
# print(X)
X_np = X.to_numpy()
y_np = y.to_numpy()

np.random.seed(10)
indices = np.random.permutation(len(X_np))
X_shuffled = X_np[indices]
y_shuffled = y_np[indices]
train_size = int(0.8 * len(X_shuffled))
val_size = int(0.1 * len(X_shuffled))

X_train, y_train = X_shuffled[:train_size], y_shuffled[:train_size]
X_val, y_val = X_shuffled[train_size:train_size + val_size], y_shuffled[train_size:train_size + val_size]
X_test, y_test = X_shuffled[train_size + val_size:], y_shuffled[train_size + val_size:]

X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std
import time
start_time=time.perf_counter()
classifier = KNN(k=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
metrics=Metrics(y_true=y_test, y_pred=y_pred)
print(f'Accuracy: {metrics.accuracy():.2f}')
print(time.perf_counter()-start_time)
