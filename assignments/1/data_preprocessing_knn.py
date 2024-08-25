import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath):
    return pd.read_csv(filepath)


# ? https://www.geeksforgeeks.org/how-to-drop-unnamed-column-in-pandas-dataframe/
def drop_unnamed_columns(df, path):
    df = df.drop(df.columns[df.columns.str.contains("unnamed", case=False)], axis=1)
    df.to_csv(path, index=False)
    return df


def drop_non_numerical(df, path):
    df = df.drop(
        columns=[
            "track_id",
            "artists",
            "album_name",
            "track_name",
            "explicit",
            "mode",
            "time_signature",
        ]
    )
    df.to_csv(path, index=False)
    return df


# ? https://www.w3schools.com/python/pandas/ref_df_drop_duplicates.asp
def drop_duplicates(df, path):
    df = df.drop_duplicates()
    df.to_csv(path, index=False)
    return df


def split(X, path, train_ratio=0.8, val_ratio=0.1):
    np.random.seed(10)
    indices = np.random.permutation(len(X))
    X_shuffled = X.iloc[indices]

    train_size = int(train_ratio * len(X_shuffled))
    val_size = int(val_ratio * len(X_shuffled))

    X_train = X_shuffled[:train_size]
    X_val = X_shuffled[train_size : train_size + val_size]
    X_test = X_shuffled[train_size + val_size :]
    split_folder = path
    Path(split_folder).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(f"{split_folder}/train.csv", index=False)
    X_val.to_csv(f"{split_folder}/validate.csv", index=False)
    X_test.to_csv(f"{split_folder}/test.csv", index=False)

    return X_train, X_val, X_test


def normalize(df, path, scaler="standard"):
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
    df_normalized.to_csv(path, index=False)
    return df_normalized

def normalize_splits(df_train, df_val, df_test, path, scaler="standard"):
    X_train = df_train.drop(columns=["track_genre"]).to_numpy()
    X_val = df_val.drop(columns=["track_genre"]).to_numpy()
    X_test = df_test.drop(columns=["track_genre"]).to_numpy()
    
    y_train = df_train[["track_genre"]]
    y_val = df_val[["track_genre"]]
    y_test = df_test[["track_genre"]]
    
    if scaler == "standard":
        # Calculate mean and std from the training data
        df_mean = np.mean(X_train, axis=0)
        df_std = np.std(X_train, axis=0)
        
        X_train_normalized = (X_train - df_mean) / df_std
        X_val_normalized = (X_val - df_mean) / df_std
        X_test_normalized = (X_test - df_mean) / df_std
    
    elif scaler == "minmax":
        df_min = np.min(X_train, axis=0)
        df_max = np.max(X_train, axis=0)
        
        X_train_normalized = (X_train - df_min) / (df_max - df_min)
        X_val_normalized = (X_val - df_min) / (df_max - df_min)
        X_test_normalized = (X_test - df_min) / (df_max - df_min)
    
    else:
        raise ValueError("Invalid scaler")
    
    # Convert numpy arrays back to DataFrames and add the track_genre column back
    df_train_normalized = pd.DataFrame(X_train_normalized, columns=df_train.columns[:-1], index=df_train.index)
    df_train_normalized = df_train_normalized.join(y_train)
    
    df_val_normalized = pd.DataFrame(X_val_normalized, columns=df_val.columns[:-1], index=df_val.index)
    df_val_normalized = df_val_normalized.join(y_val)
    
    df_test_normalized = pd.DataFrame(X_test_normalized, columns=df_test.columns[:-1], index=df_test.index)
    df_test_normalized = df_test_normalized.join(y_test)
    
    # Save the normalized DataFrames to CSV files
    df_train_normalized.to_csv(f'{path}/df_train_normalized.csv', index=False)
    df_val_normalized.to_csv(f'{path}/df_val_normalized.csv', index=False)
    df_test_normalized.to_csv(f'{path}/df_test_normalized.csv', index=False)
    
    return df_train_normalized, df_val_normalized, df_test_normalized




# spotify 1
data = load_data("data/external/spotify.csv")
Path("data/interim/1/spotify").mkdir(parents=True, exist_ok=True)
data = drop_unnamed_columns(
    data, path="data/interim/1/spotify/step1_drop_unnamed_columns.csv"
)
data = drop_non_numerical(
    data, path="data/interim/1/spotify/step2_drop_non_numerical.csv"
)
data = drop_duplicates(data, "data/interim/1/spotify/step3_drop_duplicates.csv")
data = normalize(data, path="data/interim/1/spotify/step4_normalized.csv")
data_train, data_val, data_test = split(data, path="data/interim/1/spotify/split")

# spotify 2
train = load_data("data/external/spotify-2/train.csv")
test = load_data("data/external/spotify-2/test.csv")
val = load_data("data/external/spotify-2/validate.csv")

Path("data/interim/1/spotify-2/step1_drop_unnamed_columns").mkdir(parents=True, exist_ok=True)
train = drop_unnamed_columns(
    train, path="data/interim/1/spotify-2/step1_drop_unnamed_columns/train.csv"
)
val = drop_unnamed_columns(
    val, path="data/interim/1/spotify-2/step1_drop_unnamed_columns/validate.csv"
)
test = drop_unnamed_columns(
    test, path="data/interim/1/spotify-2/step1_drop_unnamed_columns/test.csv"
)

Path("data/interim/1/spotify-2/step2_drop_non_numerical").mkdir(parents=True, exist_ok=True)
train = drop_non_numerical(
    train, path="data/interim/1/spotify-2/step2_drop_non_numerical/train.csv"
)
val = drop_non_numerical(
    val, path="data/interim/1/spotify-2/step2_drop_non_numerical/validate.csv"
)
test = drop_non_numerical(
    test, path="data/interim/1/spotify-2/step2_drop_non_numerical/test.csv"
)

# we assume no common data in test, train and val
Path("data/interim/1/spotify-2/step3_drop_duplicates").mkdir(parents=True, exist_ok=True)
train = drop_duplicates(
    train, path="data/interim/1/spotify-2/step3_drop_duplicates/train.csv"
)
val = drop_duplicates(
    val, path="data/interim/1/spotify-2/step3_drop_duplicates/validate.csv"
)
test = drop_duplicates(
    test, path="data/interim/1/spotify-2/step3_drop_duplicates/test.csv"
)

# normalize on train
Path("data/interim/1/spotify-2/final").mkdir(parents=True, exist_ok=True)
# Path("data/interim/1/spotify-2/final").mkdir(parents=True, exist_ok=True)
normalize_splits(train,val,test,"data/interim/1/spotify-2/final")

