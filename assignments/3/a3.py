import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
from models.MLP.MLP_Classifier import MLP_Classifier

def describe_dataset(df):
    numerical_cols = df.to_numpy()
    results = []
    means = np.mean(numerical_cols, axis=0)
    stds = np.std(numerical_cols, axis=0)
    mins = np.min(numerical_cols, axis=0)
    maxs = np.max(numerical_cols, axis=0)
    results = np.column_stack((df.columns, means, stds, mins, maxs))
    results_df = pd.DataFrame(results, columns=['Attribute', 'Mean', 'Standard Deviation', 'Min', 'Max'])
    md_table = results_df.to_markdown(index=False)
    print(md_table)
    results_df.to_csv('data/interim/3/WineQT/WineQT_description.csv', index=False)
    return results_df

def plot_quality_distribution(df):
    plt.figure(figsize=(10, 6))
    df['quality'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution of Wine Quality')
    plt.xlabel('Quality')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.savefig('assignments/3/figures/quality_distribution.png')
    plt.close()

    features = df.columns.tolist()
    features.remove('quality')
    num_features = len(features)
    cols = 3
    rows = (num_features // cols) + (num_features % cols > 0)
    plt.figure(figsize=(15, rows * 4))
    for i, feature in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        plt.hist(df[feature], bins=30, color='lightblue', edgecolor='black')
        plt.title(feature)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('assignments/3/figures/distributions.png')
    plt.close()

def normalize_and_standardize(df):

    df.fillna(df.mean(), inplace=True)

    min_max_scaler = MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(df.drop(columns=['quality']))

    standard_scaler = StandardScaler()
    standardized_data = standard_scaler.fit_transform(df.drop(columns=['quality']))

    normalized_df = pd.DataFrame(normalized_data, columns=df.columns[0:-1])
    standardized_df = pd.DataFrame(standardized_data, columns=df.columns[0:-1])

    normalized_df['quality'] = df['quality']
    standardized_df['quality'] = df['quality']

    normalized_df.to_csv('data/interim/3/WineQT/WineQT_normalized.csv', index=False)
    standardized_df.to_csv('data/interim/3/WineQT/WineQT_standardized.csv', index=False)
    return normalized_df, standardized_df

def split_dataset(dataset, output_dir='data/interim/3/WineQT/split/', train_size=0.8, val_size=0.1):
    
    X = dataset.drop(columns=['quality']).to_numpy()
    y = dataset['quality'].to_numpy()
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    train_split = int(train_size * X_shuffled.shape[0])
    val_split = int(val_size * X_shuffled.shape[0])

    X_train = X_shuffled[:train_split]
    y_train = y_shuffled[:train_split]

    X_validation = X_shuffled[train_split:train_split + val_split]
    y_validation = y_shuffled[train_split:train_split + val_split]

    X_test = X_shuffled[train_split + val_split:]
    y_test = y_shuffled[train_split + val_split:]

    feature_columns = dataset.columns[:-1]  
    train_split_df = pd.DataFrame(X_train, columns=feature_columns)
    train_split_df['quality'] = y_train

    validation_split_df = pd.DataFrame(X_validation, columns=feature_columns)
    validation_split_df['quality'] = y_validation

    test_split_df = pd.DataFrame(X_test, columns=feature_columns)
    test_split_df['quality'] = y_test

    train_split_df.to_csv(output_dir + 'train.csv', index=False)
    validation_split_df.to_csv(output_dir + 'validation.csv', index=False)
    test_split_df.to_csv(output_dir + 'test.csv', index=False)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_validation.shape, y_validation.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def data_preprocessing():
    file_path = 'data/interim/3/WineQT/WineQT.csv'
    dataset = pd.read_csv(file_path)
    dataset.drop(['Id'], axis=1, inplace=True)

    description = describe_dataset(dataset)
    plot_quality_distribution(dataset)
    normalized_data, standardized_data = normalize_and_standardize(dataset)

    file_path = 'data/interim/3/WineQT/WineQT_normalized.csv'
    dataset = pd.read_csv(file_path)
    return split_dataset(dataset)

def wandb_init(lr, max_epochs, optimizer, activation, hidden_layers, batch_size):
    config = {
        "lr": lr, 
        "model_type": "MLP_Classifier",
        "optimizer": optimizer, # SGC/BGD/MBGD
        "criterion": "mse",
        "num_epochs": max_epochs,
        "batch_size": batch_size,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "wandb_run_name": "shaunak1" ,
    }

    wandb.init(project = "SMAI_A3",
            config = config  
            )
    wandb.run.name = f"{config['optimizer']}_{config['activation']}_{len(config['hidden_layers'])}_{config['lr']}_{config['batch_size']}_{config['num_epochs']}"
    print(wandb.run.name)


if __name__ == "__main__":
    X_train, y_train, X_validation, y_validation, X_test, y_test = data_preprocessing()

    lr = 0.001
    max_epochs = 10000
    optimizer = 'bgd'
    activation = 'sigmoid'
    hidden_layers = [8,8,]
    batch_size = 32

    wandb_init(lr, max_epochs, optimizer, activation, hidden_layers, batch_size)
    model = MLP_Classifier(X_train.shape[1], hidden_layers, 11, learning_rate=lr, activation=activation, optimizer=optimizer, print_every=100, wandb_log=True)
    costs = model.fit(X_train, y_train, max_epochs=max_epochs, batch_size=batch_size, X_validation=X_validation, y_validation=y_validation, early_stopping=True)
    wandb.finish()
    # plt.figure(figsize=(8, 6))
    # plt.plot(costs)
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.show()

