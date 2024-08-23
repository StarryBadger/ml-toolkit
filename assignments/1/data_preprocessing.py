import pandas as pd
import numpy as np

from pathlib import Path


def load_data(filepath):
    return pd.read_csv(filepath)

# ? https://www.geeksforgeeks.org/how-to-drop-unnamed-column-in-pandas-dataframe/
def drop_unnamed_columns(df):
    df = df.drop(df.columns[df.columns.str.contains("unnamed", case=False)], axis=1)
    df.to_csv("data/interim/spotify/step1_drop_unnamed.csv", index=False)
    return df


def drop_non_numerical(df):
    df = df.drop(
        columns=["track_id", "artists", "album_name", "track_name", "explicit"]
    )
    df.to_csv("data/interim/spotify/step2_drop_non_numerical.csv", index=False)
    return df

# ? https://www.w3schools.com/python/pandas/ref_df_drop_duplicates.asp
def drop_duplicates(df):
    df = df.drop_duplicates()
    df.to_csv("data/interim/spotify/step3_drop_duplicates.csv", index=False)
    return df


def split(X, train_ratio=0.8, val_ratio=0.1):
    np.random.seed(10)
    indices = np.random.permutation(len(X))
    X_shuffled = X.iloc[indices]

    train_size = int(train_ratio * len(X_shuffled))
    val_size = int(val_ratio * len(X_shuffled))

    X_train = X_shuffled[:train_size]
    X_val = X_shuffled[train_size : train_size + val_size]
    X_test = X_shuffled[train_size + val_size :]
    split_folder="data/interim/spotify/split"
    Path(split_folder).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(f"{split_folder}/train.csv", index=False)
    X_val.to_csv(f"{split_folder}/val.csv", index=False)
    X_test.to_csv(f"{split_folder}/test.csv", index=False)

    return X_train, X_val, X_test


def normalize(df, scaler="standard"):
    X = df.drop(columns=["track_genre"])
    df_values = X.to_numpy()
    y = df[["track_genre"]]

    if scaler == "standard":
        df_mean = np.mean(df_values, axis=0)
        df_std = np.std(df_values, axis=0)
        df_values = (df_values - df_mean) / df_std
    elif scaler == "minmax":
        df_min = np.min(df_values, axis=0)
        df_max = np.max(df_values, axis=0)
        df_values = (df_values - df_min) / (df_max - df_min)
    else:
        raise ValueError("Invalid scaler")
    df_normalized = pd.DataFrame(df_values, columns=X.columns, index=X.index)
    df_normalized = df_normalized.join(y)
    df_normalized.to_csv("data/interim/spotify/step4_normalized.csv", index=False)
    return df_normalized


data = load_data("data/external/spotify.csv")
Path("data/interim/spotify").mkdir(parents=True, exist_ok=True)
data = drop_unnamed_columns(data)
data = drop_non_numerical(data)
data = drop_duplicates(data)
data = normalize(data)
data_train, data_val, data_test = split(data)
