import numpy as np
import wandb
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from performance_measures.classification_metrics import Metrics
from models.MLP.MLPClassifier import MLPClassifier

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

    normalized_df['quality'] = df['quality'] -3
    standardized_df['quality'] = df['quality'] -3

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

    # print("Training set shape:", X_train.shape, y_train.shape)
    # print("Validation set shape:", X_validation.shape, y_validation.shape)
    # print("Test set shape:", X_test.shape, y_test.shape)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def data_preprocessing():
    file_path = 'data/interim/3/WineQT/WineQT.csv'
    dataset = pd.read_csv(file_path)
    dataset.drop(['Id'], axis=1, inplace=True)

    description = describe_dataset(dataset)
    plot_quality_distribution(dataset)
    normalized_data, standardized_data = normalize_and_standardize(dataset)

def load_wineqt():
    file_path = 'data/interim/3/WineQT/WineQT_normalized.csv'
    dataset = pd.read_csv(file_path)
    return split_dataset(dataset)


def train_and_log(project="SMAI_A3", config=None):
    # Initialize the run for sweeps
    with wandb.init(config=config):
        config = wandb.config
        config_dict = dict(config)
        wandb.run.name = f"{config_dict['optimizer']}_{config_dict['activation']}_{len(config_dict['hidden_layers'])}_{config_dict['lr']}_{config_dict['batch_size']}_{config_dict['max_epochs']}"
        
        # Initialize the MLP classifier with the W&B configuration
        model = MLPClassifier(
            input_size=X_train.shape[1],
            hidden_layers=config.hidden_layers,
            num_classes=6,
            learning_rate=config.lr,
            activation=config.activation,
            optimizer=config.optimizer,
            wandb_log=True
        )

        costs = model.fit(
            X_train, y_train, 
            max_epochs=config.max_epochs, 
            batch_size=config.batch_size, 
            X_validation=X_validation, 
            y_validation=y_validation, 
            early_stopping=True, 
            patience=5
        )

        y_pred_validation = model.predict(X_validation)
        validation_metrics = Metrics(y_validation, y_pred_validation, task="classification")

        validation_accuracy = validation_metrics.accuracy()
        precision = validation_metrics.precision_score()
        recall = validation_metrics.recall_score()
        f1_score = validation_metrics.f1_score()

        wandb.log({
            'accuracy_val_final': validation_accuracy,
            'precision_val_final': precision,
            'recall_val_final': recall,
            'f1_score_val_final': f1_score
        })

        global best_model_params, best_validation_accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_params = dict(config)

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'accuracy_val_final',
    },
    'parameters': {
        'lr': {'values': [0.002, 0.01, 0.05]}, 
        'max_epochs': {'values': [200, 800]},
        'optimizer': {'values': ['sgd', 'bgd', 'mbgd']}, 
        'activation': {'values': ['sigmoid','tanh','relu','signum']},
        'hidden_layers': {'values': [[8], [16], [8, 8], [8, 16], [16, 8], [16, 16], [8, 8, 8]]}, 
        'batch_size': {'values': [16, 32]} 
    }
}

if __name__ == "__main__":
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_wineqt()

    best_model_params = None
    best_validation_accuracy = 0

    sweep_id = wandb.sweep(sweep_config, project="SMAI_A3")
    wandb.agent(sweep_id, function=train_and_log, count=256)

    with open('best_model_config.json', 'w') as f:
        json.dump(best_model_params, f)

    wandb.finish()


