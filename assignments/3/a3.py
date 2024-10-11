import time
import copy
import json
import wandb
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.knn.knn import KNN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from performance_measures.classification_metrics import Metrics
from models.MLP.MLPClassifier import MLPClassifier
from models.MLP.MultiLabelMLP import MultiLabelMLP
from models.MLP.MLPRegression import MLPRegression
from models.MLP.MLPLogistic import MLPLogisticRegression
from models.AutoEncoders.AutoEncoders import AutoEncoder


class TestMLPGradientChecking(unittest.TestCase):

    def setUp(self):
        self.X_train_classification = X_train_classification
        self.y_train_classification = y_train_classification

        self.X_train_regression = X_train_regression
        self.y_train_regression = y_train_regression

    def test_classification_gradient_checking_with_sigmoid(self):
        model = MLPClassifier(
            input_size=self.X_train_classification.shape[1],
            hidden_layers=[10, 5],
            num_classes=6,
            learning_rate=0.01,
            activation="sigmoid",
            optimizer="sgd",
        )
        model.gradient_checking(
            self.X_train_classification[:20], self.y_train_classification[:20]
        )

    def test_classification_gradient_checking_with_relu(self):
        model = MLPClassifier(
            input_size=self.X_train_classification.shape[1],
            hidden_layers=[10],
            num_classes=6,
            learning_rate=0.001,
            activation="relu",
            optimizer="bgd",
        )
        model.gradient_checking(
            self.X_train_classification[100:110], self.y_train_classification[100:110]
        )

    def test_classification_gradient_checking_with_tanh(self):
        model = MLPClassifier(
            input_size=self.X_train_classification.shape[1],
            hidden_layers=[15, 5, 3],
            num_classes=6,
            learning_rate=0.005,
            activation="tanh",
            optimizer="mbgd",
        )
        model.gradient_checking(
            self.X_train_classification[200:300], self.y_train_classification[200:300]
        )


def describe_dataset(df, file_path="data/interim/3/WineQT/WineQT_description.csv"):
    numerical_cols = df.to_numpy()
    results = []
    means = np.nanmean(numerical_cols, axis=0)
    stds = np.nanstd(numerical_cols, axis=0)
    mins = np.nanmin(numerical_cols, axis=0)
    maxs = np.nanmax(numerical_cols, axis=0)
    results = np.column_stack((df.columns, means, stds, mins, maxs))
    results_df = pd.DataFrame(
        results, columns=["Attribute", "Mean", "Standard Deviation", "Min", "Max"]
    )
    md_table = results_df.to_markdown(index=False)
    print(md_table)
    results_df.to_csv(file_path, index=False)
    return results_df


def plot_quality_distribution(
    df, regression=False, save_path="assignments/3/figures/distributions.png"
):
    if not regression:
        plt.figure(figsize=(10, 6))
        df["quality"].value_counts().sort_index().plot(kind="bar")
        plt.title("Distribution of Wine Quality")
        plt.xlabel("Quality")
        plt.ylabel("Frequency")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        plt.savefig("assignments/3/figures/quality_distribution.png")
        plt.close()

    features = df.columns.tolist()
    if not regression:
        features.remove("quality")
    num_features = len(features)
    cols = 3
    rows = (num_features // cols) + (num_features % cols > 0)
    plt.figure(figsize=(15, rows * 4))
    for i, feature in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        plt.hist(df[feature], bins=30, color="lightblue", edgecolor="black")
        plt.title(feature)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def normalize_and_standardize(df, regression=False):
    if not regression:
        df.fillna(df.mean(), inplace=True)

        min_max_scaler = MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(df.drop(columns=["quality"]))

        standard_scaler = StandardScaler()
        standardized_data = standard_scaler.fit_transform(df.drop(columns=["quality"]))

        normalized_df = pd.DataFrame(normalized_data, columns=df.columns[0:-1])
        standardized_df = pd.DataFrame(standardized_data, columns=df.columns[0:-1])

        normalized_df["quality"] = df["quality"] - 3
        standardized_df["quality"] = df["quality"] - 3

        normalized_df.to_csv("data/interim/3/WineQT/WineQT_normalized.csv", index=False)
        standardized_df.to_csv(
            "data/interim/3/WineQT/WineQT_standardized.csv", index=False
        )
        return normalized_df, standardized_df
    else:
        df.fillna(df.mean(), inplace=True)

        min_max_scaler = MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(df)

        standard_scaler = StandardScaler()
        standardized_data = standard_scaler.fit_transform(df)

        normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
        standardized_df = pd.DataFrame(standardized_data, columns=df.columns)

        normalized_df.to_csv(
            "data/interim/3/HousingData/HousingData_normalized.csv", index=False
        )
        standardized_df.to_csv(
            "data/interim/3/HousingData/HousingData_standardized.csv", index=False
        )
        return normalized_df, standardized_df


def split_dataset_wineqt(
    dataset, output_dir="data/interim/3/WineQT/split/", train_size=0.8, val_size=0.1
):

    X = dataset.drop(columns=["quality"]).to_numpy()
    y = dataset["quality"].to_numpy()

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    train_split = int(train_size * X_shuffled.shape[0])
    val_split = int(val_size * X_shuffled.shape[0])

    X_train = X_shuffled[:train_split]
    y_train = y_shuffled[:train_split]

    X_validation = X_shuffled[train_split : train_split + val_split]
    y_validation = y_shuffled[train_split : train_split + val_split]

    X_test = X_shuffled[train_split + val_split :]
    y_test = y_shuffled[train_split + val_split :]

    feature_columns = dataset.columns[:-1]
    train_split_df = pd.DataFrame(X_train, columns=feature_columns)
    train_split_df["quality"] = y_train

    validation_split_df = pd.DataFrame(X_validation, columns=feature_columns)
    validation_split_df["quality"] = y_validation

    test_split_df = pd.DataFrame(X_test, columns=feature_columns)
    test_split_df["quality"] = y_test

    train_split_df.to_csv(output_dir + "train.csv", index=False)
    validation_split_df.to_csv(output_dir + "validation.csv", index=False)
    test_split_df.to_csv(output_dir + "test.csv", index=False)

    # print("Training set shape:", X_train.shape, y_train.shape)
    # print("Validation set shape:", X_validation.shape, y_validation.shape)
    # print("Test set shape:", X_test.shape, y_test.shape)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def split_dataset_housing(dataset, train_size=0.7, val_size=0.15):
    shuffled_indices = np.random.permutation(len(dataset))
    dataset_shuffled = dataset.iloc[shuffled_indices]
    train_end = int(train_size * len(dataset))
    val_end = train_end + int(val_size * len(dataset))

    train_set = dataset_shuffled.iloc[:train_end]
    val_set = dataset_shuffled.iloc[train_end:val_end]
    test_set = dataset_shuffled.iloc[val_end:]

    X_train = train_set.drop(columns=["MEDV"]).to_numpy()
    y_train = train_set["MEDV"].to_numpy()

    X_val = val_set.drop(columns=["MEDV"]).to_numpy()
    y_val = val_set["MEDV"].to_numpy()

    X_test = test_set.drop(columns=["MEDV"]).to_numpy()
    y_test = test_set["MEDV"].to_numpy()

    return X_train, y_train, X_val, y_val, X_test, y_test


def wine_preprocessing():
    file_path = "data/interim/3/WineQT/WineQT.csv"
    dataset = pd.read_csv(file_path)
    dataset.drop(["Id"], axis=1, inplace=True)

    description = describe_dataset(dataset)
    plot_quality_distribution(dataset)
    normalized_data, standardized_data = normalize_and_standardize(dataset)


def housing_preprocessing():
    file_path = "data/interim/3/HousingData/HousingData.csv"
    dataset = pd.read_csv(file_path)

    description = describe_dataset(
        dataset, file_path="data/interim/3/HousingData/HousingData_description.csv"
    )
    plot_quality_distribution(
        dataset,
        regression=True,
        save_path="assignments/3/figures/regression_distributions.png",
    )
    normalized_data, standardized_data = normalize_and_standardize(
        dataset, regression=True
    )


def load_wineqt():
    file_path = "data/interim/3/WineQT/WineQT_normalized.csv"
    dataset = pd.read_csv(file_path)
    return split_dataset_wineqt(dataset)


def load_housing():
    file_path = "data/interim/3/HousingData/HousingData_normalized.csv"
    dataset = pd.read_csv(file_path)
    return split_dataset_housing(dataset)


def train_and_log_classification(project="SMAI_A3", config=None):
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
            wandb_log=True,
        )

        costs = model.fit(
            X_train,
            y_train,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            X_validation=X_validation,
            y_validation=y_validation,
            early_stopping=True,
            patience=config.max_epochs // 20,
        )

        y_pred_validation = model.predict(X_validation)
        validation_metrics = Metrics(
            y_validation, y_pred_validation, task="classification"
        )

        validation_accuracy = validation_metrics.accuracy()

        precision = validation_metrics.precision_score()
        recall = validation_metrics.recall_score()
        f1_score = validation_metrics.f1_score()

        wandb.log(
            {
                "accuracy_val_final": validation_accuracy,
                "precision_val_final": precision,
                "recall_val_final": recall,
                "f1_score_val_final": f1_score,
            }
        )

        global best_model_params, best_validation_accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_params = dict(config)


def train_and_log_regression(project="SMAI_A3", config=None):
    with wandb.init(config=config):
        config = wandb.config
        config_dict = dict(config)
        wandb.run.name = f"{config_dict['optimizer']}_{config_dict['activation']}_{len(config_dict['hidden_layers'])}_{config_dict['lr']}_{config_dict['batch_size']}_{config_dict['max_epochs']}"

        model = MLPRegression(
            input_size=X_train.shape[1],
            hidden_layers=config.hidden_layers,
            output_size=1,
            learning_rate=config.lr,
            activation=config.activation,
            optimizer=config.optimizer,
            wandb_log=True,
        )

        costs = model.fit(
            X_train,
            y_train,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            X_validation=X_validation,
            y_validation=y_validation,
            early_stopping=True,
            patience=config.max_epochs // 50,
        )

        y_pred_validation = model.predict(X_validation).squeeze()
        validation_metrics = Metrics(y_validation, y_pred_validation, task="regression")

        mse = validation_metrics.mse()
        mae = validation_metrics.mae()
        r2 = validation_metrics.r2_score()
        rmse = validation_metrics.rmse()

        wandb.log(
            {
                "mse_val_final": mse,
                "mae_val_final": mae,
                "r2_val_final": r2,
                "rmse_val_final": rmse,
            }
        )

        global best_model_params_regression, best_validation_mse
        if mse < best_validation_mse:
            best_validation_mse = mse
            best_model_params_regression = dict(config)

def train_and_log_multi_label_classification(project="SMAI_A3", config=None):
    # Initialize the run for sweeps
    with wandb.init(config=config):
        config = wandb.config
        config_dict = dict(config)
        wandb.run.name = f"{config_dict['optimizer']}_{config_dict['activation']}_{len(config_dict['hidden_layers'])}_{config_dict['lr']}_{config_dict['batch_size']}_{config_dict['max_epochs']}"

        # Initialize the MLP classifier with the W&B configuration
        model = MultiLabelMLP(
            input_size=X_train.shape[1],
            hidden_layers=config.hidden_layers,
            output_size=8,
            learning_rate=config.lr,
            activation=config.activation,
            optimizer=config.optimizer,
            wandb_log=True,
        )

        costs = model.fit(
            X_train,
            y_train,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            X_validation=X_validation,
            y_validation=y_validation,
            early_stopping=True,
            patience=config.max_epochs // 20,
        )

        y_pred_validation = model.predict(X_validation)
        validation_metrics = Metrics(
            y_validation, y_pred_validation, task="classification"
        )

        validation_accuracy = validation_metrics.accuracy(one_hot=True)
        precision = validation_metrics.precision_score()
        recall = validation_metrics.recall_score()
        f1_score = validation_metrics.f1_score()
        hamming_loss = validation_metrics.hamming_loss()
        hamming_accuracy = validation_metrics.hamming_accuracy()

        wandb.log(
            {
                "accuracy_val_final": validation_accuracy,
                "precision_val_final": precision,
                "recall_val_final": recall,
                "f1_score_val_final": f1_score,
                "hamming_loss": hamming_loss,
                "hamming_accuracy": hamming_accuracy,
            }
        )

        global best_model_params, best_validation_accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_params = dict(config)



def print_hyperparams_wineqt(file_path="data/interim/3/WineQT/hyperparams_2fd1o2fl.csv"):
    df = pd.read_csv(file_path)

    metrics_df = df[
        [
            "activation",
            "batch_size",
            "hidden_layers",
            "optimizer",
            "lr",
            "max_epochs",
            "accuracy_val_final",
            "epoch",
            "f1_score_val_final",
            "precision_val_final",
            "recall_val_final",
        ]
    ]
    metrics_df.columns = [
        "Activation",
        "Batch Size",
        "Hidden Layers",
        "Optimizer",
        "Learning Rate",
        "Max Epochs",
        "Validation Accuracy",
        "Epoch",
        "F1 Score",
        "Precision",
        "Recall",
    ]

    markdown_table = metrics_df.to_markdown(index=False)
    print(markdown_table)
    metrics_df.to_markdown("temp.md", index=False)

def print_hyperparams_advertisement(file_path="data/interim/3/advertisement/hyperparams_2hry5c8h.csv"):
    df = pd.read_csv(file_path)

    metrics_df = df[
        [
            "activation",
            "batch_size",
            "hidden_layers",
            "optimizer",
            "lr",
            "max_epochs",
            "accuracy_val_final",
            "epoch",
            "f1_score_val_final",
            "precision_val_final",
            "recall_val_final",
            "hamming_accuracy",
            "hamming_loss"

        ]
    ]
    metrics_df.columns = [
        "Activation",
        "Batch Size",
        "Hidden Layers",
        "Optimizer",
        "Learning Rate",
        "Max Epochs",
        "Validation Accuracy",
        "Epoch",
        "F1 Score",
        "Precision",
        "Recall",
        "Hamming Accuracy",
        "Hamming Loss"
    ]

    markdown_table = metrics_df.to_markdown(index=False)
    print(markdown_table)
    metrics_df.to_markdown("temp.md", index=False)

def test_on_best_wineqt():
    with open("data/interim/3/WineQT/best_model_config.json", "r") as file:
        config = json.load(file)
    model = MLPClassifier(
        X_train.shape[1],
        config["hidden_layers"],
        6,
        learning_rate=config["lr"],
        activation=config["activation"],
        optimizer=config["optimizer"],
        print_every=10,
        wandb_log=False,
    )
    costs = model.fit(
        X_train,
        y_train,
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        X_validation=X_validation,
        y_validation=y_validation,
        early_stopping=True,
        patience=config["max_epochs"] // 20,
    )
    # model.gradient_checking(X_train[:5], y_train[:5])
    y_pred_test = model.predict(X_test)
    test_metrics = Metrics(y_test, y_pred_test, task="classification")
    print(y_test, y_pred_test)

    test_accuracy = test_metrics.accuracy()
    precision = test_metrics.precision_score()
    recall = test_metrics.recall_score()
    f1_score = test_metrics.f1_score()

    print(
        f"Accuracy: {test_accuracy}\
          \nPrecision: {precision}\
          \nRecall: {recall}\
          \nF1 Score: {f1_score}"
    )

def analyze_model_impact():
    with open("data/interim/3/WineQT/best_model_config.json", "r") as file:
        best_config = json.load(file)
    def plot_loss_vs_epochs(experiments, title, save_path):
        plt.figure(figsize=(10, 6))
        for label, losses in experiments.items():
            plt.plot(losses, label=label)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.savefig(save_path)
        plt.show()

    # Effect of Non-linearity (Activation Function)
    def analyze_activation_impact():
        activations = ['sigmoid', 'relu', 'tanh', 'linear']
        experiments = {}
        for activation in activations:
            model = MLPClassifier(
                X_train.shape[1],
                best_config["hidden_layers"],
                6,
                learning_rate=best_config["lr"],
                activation=activation,
                optimizer=best_config["optimizer"],
                print_every=10,
                wandb_log=False
            )
            costs = model.fit(
                X_train,
                y_train,
                max_epochs=best_config["max_epochs"],
                batch_size=best_config["batch_size"],
                X_validation=X_validation,
                y_validation=y_validation,
                early_stopping=True,
                patience=best_config["max_epochs"] // 20,
            )
            experiments[activation] = costs
        plot_loss_vs_epochs(experiments, 'Effect of Activation Function on Loss', 'assignments/3/figures/2.5.1.png')

    # Effect of Learning Rate
    def analyze_learning_rate_impact():
        learning_rates = [0.001, 0.005, 0.01, 0.05]
        experiments = {}
        for lr in learning_rates:
            model = MLPClassifier(
                X_train.shape[1],
                best_config["hidden_layers"],
                6,
                learning_rate=lr,
                activation=best_config["activation"],
                optimizer=best_config["optimizer"],
                print_every=10,
                wandb_log=False
            )
            costs = model.fit(
                X_train,
                y_train,
                max_epochs=best_config["max_epochs"],
                batch_size=best_config["batch_size"],
                X_validation=X_validation,
                y_validation=y_validation,
                early_stopping=True,
                patience=best_config["max_epochs"] // 20,
            )
            experiments[f'LR={lr}'] = costs
        plot_loss_vs_epochs(experiments, 'Effect of Learning Rate on Loss', 'assignments/3/figures/2.5.2.png')

    # Effect of Batch Size
    def analyze_batch_size_impact():
        batch_sizes = [16, 32, 64, 128]
        experiments = {}
        for batch_size in batch_sizes:
            model = MLPClassifier(
                X_train.shape[1],
                best_config["hidden_layers"],
                6,
                learning_rate=best_config["lr"],
                activation=best_config["activation"],
                optimizer=best_config["optimizer"],
                print_every=10,
                wandb_log=False
            )
            costs = model.fit(
                X_train,
                y_train,
                max_epochs=best_config["max_epochs"],
                batch_size=batch_size,
                X_validation=X_validation,
                y_validation=y_validation,
                early_stopping=True,
                patience=best_config["max_epochs"] // 20,
            )
            experiments[f'Batch={batch_size}'] = costs
        plot_loss_vs_epochs(experiments, 'Effect of Batch Size on Loss', 'assignments/3/figures/2.5.3.png')

    analyze_activation_impact()
    analyze_learning_rate_impact()
    analyze_batch_size_impact()

def test_on_best_housing():

    with open("data/interim/3/HousingData/best_model_config.json", "r") as file:
        config = json.load(file)
    # print(config)
    # with wandb.init(config=config):
    #     config = wandb.config
    #     config_dict = dict(config)
    #     wandb.run.name = f"{config_dict['optimizer']}_{config_dict['activation']}_{len(config_dict['hidden_layers'])}_{config_dict['lr']}_{config_dict['batch_size']}_{config_dict['max_epochs']}"
    model = MLPRegression(
        X_train.shape[1],
        config["hidden_layers"],
        output_size=1,
        learning_rate=config["lr"],
        activation=config["activation"],
        optimizer=config["optimizer"],
        print_every=10,
        wandb_log=False,
    )
    # model.gradient_checking(X_train[:5], y_train[:5])
    costs = model.fit(
        X_train,
        y_train,
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        X_validation=X_validation,
        y_validation=y_validation,
        early_stopping=True,
        patience=config["max_epochs"] // 50,
    )
    y_pred_test = model.predict(X_test).squeeze()
    test_metrics = Metrics(y_test, y_pred_test, task="regression")

    mse = test_metrics.mse()
    mae = test_metrics.mae()
    r2 = test_metrics.r2_score()
    rmse = test_metrics.rmse()

    print(
        f"MSE: {mse}\
            \nMAE: {mae}\
            \nRMSE: {rmse}\
            \nR2 Score: {r2}"
    )

def test_on_best_advertisement(index_to_label):
    with open("data/interim/3/advertisement/best_model_config.json", "r") as file:
        config = json.load(file)
    model = MultiLabelMLP(
        input_size=X_train.shape[1],
        hidden_layers=config["hidden_layers"],
        output_size=8,
        learning_rate=config["lr"],
        activation=config["activation"],
        optimizer=config["optimizer"],
        wandb_log=False,
        print_every=100
    )
    costs = model.fit(
        X_train,
        y_train,
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        X_validation=X_validation,
        y_validation=y_validation,
        early_stopping=True,
        patience=config["max_epochs"]// 20,
    )
    
    y_pred_test = model.predict(X_test)

    all_labels = []

    for multi_hot_vector in y_pred_test:
        labels = [index_to_label[idx] for idx, value in enumerate(multi_hot_vector) if value == 1]
        all_labels.append(labels)

    print('Predictions',all_labels)

    print('Multi Hot Encoding of predictions', y_pred_test)

    for multi_hot_vector in y_test:
        labels = [index_to_label[idx] for idx, value in enumerate(multi_hot_vector) if value == 1]
        all_labels.append(labels)

    print('Actual', all_labels)

    print('Multi Hot Encoding of actual', y_test)
 
    test_metrics = Metrics(y_test, y_pred_test, task="classification")

    test_accuracy = test_metrics.accuracy(one_hot=True)
    precision = test_metrics.precision_score()
    recall = test_metrics.recall_score()
    f1_score = test_metrics.f1_score()
    hamming_loss = test_metrics.hamming_loss()
    hamming_accuracy = test_metrics.hamming_accuracy()

    print(f'Accuracy: {test_accuracy}\
          \nPrecision: {precision}\
          \nRecall: {recall}\
          \nF1 Score: {f1_score}')

    print(f'Hamming Loss: {hamming_loss}\
        \nHamming Accuracy: {hamming_accuracy}')


def multi_hot_encode(labels):
    unique_labels = set(label for sublist in labels for label in sublist.split())
    multi_hot = {label: 0 for label in unique_labels}
    for label in labels:
        for l in label.split():
            multi_hot[l] = 1
    return pd.Series(multi_hot)


sweep_config_wine = {
    "method": "bayes",
    "metric": {
        "name": "accuracy_val_final",
    },
    "parameters": {
        "lr": {"values": [0.002, 0.01, 0.05]},
        "max_epochs": {"values": [200, 800]},
        "optimizer": {"values": ["sgd", "bgd", "mbgd"]},
        "activation": {"values": ["sigmoid", "tanh", "relu", "signum"]},
        "hidden_layers": {
            "values": [[8], [16], [8, 8], [8, 16], [16, 8], [16, 16], [8, 8, 8]]
        },
        "batch_size": {"values": [16, 32]},
    },
}

sweep_config_housing = {
    "method": "bayes",
    "metric": {
        "name": "mse_val_final",
        "goal": "minimize",
    },
    "parameters": {
        "lr": {"values": [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]},
        "max_epochs": {"values": [500, 2500, 7500]},
        "optimizer": {"values": ["sgd", "bgd", "mbgd"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "hidden_layers": {
            "values": [
                [8],
                [16],
                [8, 8],
                [8, 16],
                [16, 8],
                [16, 16],
                [8, 8, 8],
                [8, 8, 8, 8],
            ]
        },
        "batch_size": {"values": [16, 32]},
    },
}

sweep_config_diabetes = {
    "method": "bayes",
    "metric": {
        "name": "accuracy_val_final",
    },
    "parameters": {
        "lr": {"values": [0.01, 0.05, 0.1, 0.2]},
        "max_epochs": {"values": [800, 1500]},
        "optimizer": {"values": ["sgd", "bgd", "mbgd"]}, 
        "activation": {"values": ["sigmoid", "tanh", "relu"]}, 
        "hidden_layers": {
            "values": [
                [64, 64],
                [128],
                [64, 32],
                [128, 64],
                [64, 64, 32]
            ] 
        },
        "batch_size": {"values": [16, 32, 64]}, 
    },
}


def advertisement_preprocessing(
    dataset_path="data/interim/3/advertisement/advertisement.csv",
):
    dataset = pd.read_csv(dataset_path)
    gender_encoded = pd.get_dummies(dataset["gender"], prefix="gender", dtype=int)
    education_order = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    dataset["education"] = dataset["education"].map(education_order)
    dataset["married"] = dataset["married"].astype(int)
    occupation_encoded = pd.get_dummies(
        dataset["occupation"], prefix="occupation", dtype=int
    )
    most_bought_encoded = pd.get_dummies(
        dataset["most bought item"], prefix="most_bought", dtype=int
    )
    dataset = dataset.drop(columns=["city", "gender", "occupation", "most bought item"])
    encoded_dataset = pd.concat(
        [dataset, gender_encoded, occupation_encoded, most_bought_encoded], axis=1
    )
    scaler = MinMaxScaler()
    numerical_features = ["age", "income", "education", "children", "purchase_amount"]
    encoded_dataset[numerical_features] = scaler.fit_transform(
        encoded_dataset[numerical_features]
    )
    encoded_dataset.to_csv(
        "data/interim/3/advertisement/advertisement_encoded.csv", index=False
    )


def encode_labels(dataset):
    unique_labels = set()
    for label_list in dataset["labels"]:
        unique_labels.update(label_list.split())
    unique_labels = sorted(unique_labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    print(label_to_index)
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    def multi_hot_encode(labels):
        encoded_array = []
        for label_list in labels:
            encoded = [0] * len(unique_labels)
            for label in label_list.split():
                if label in label_to_index:
                    encoded[label_to_index[label]] = 1
            encoded_array.append(encoded)
        return encoded_array

    return multi_hot_encode(dataset["labels"]), index_to_label


def get_advertisement_data():
    dataset = pd.read_csv("data/interim/3/advertisement/advertisement_encoded.csv")
    labels, index_to_label = encode_labels(dataset)
    dataset["labels"] = labels

    shuffled_indices = np.random.permutation(len(dataset))
    shuffled_dataset = dataset.iloc[shuffled_indices].reset_index(drop=True)

    train_size = int(0.8 * len(shuffled_dataset))
    val_size = int(0.1 * len(shuffled_dataset))

    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]

    X_train = train_dataset.drop(columns="labels").values
    y_train = np.array(train_dataset["labels"].tolist())

    X_validation = val_dataset.drop(columns="labels").values
    y_validation = np.array(val_dataset["labels"].tolist())

    X_test = test_dataset.drop(columns="labels").values
    y_test = np.array(test_dataset["labels"].tolist())

    return X_train, y_train, X_validation, y_validation, X_test, y_test, index_to_label

import os

def process_diabetes_data():
    input_file = "data/interim/3/diabetes/diabetes.csv"
    output_dir = "data/interim/3/diabetes/"
    split_dir = f"{output_dir}split"
    os.makedirs(split_dir, exist_ok=True)
    data = pd.read_csv(input_file)
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    means = features.mean()
    stds = features.std()
    standardized_features = (features - means) / stds
    standardized_data = pd.concat([standardized_features, target], axis=1)
    standardized_data.to_csv(f"{output_dir}_standardized_diabetes.csv")

    shuffled_data = standardized_data.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    train_size = int(0.7 * len(shuffled_data))
    validation_size = int(0.15 * len(shuffled_data))

    X_train = shuffled_data.iloc[:train_size, :-1]
    y_train = shuffled_data.iloc[:train_size, -1]

    X_validation = shuffled_data.iloc[train_size : train_size + validation_size, :-1]
    y_validation = shuffled_data.iloc[train_size : train_size + validation_size, -1]

    X_test = shuffled_data.iloc[train_size + validation_size :, :-1]
    y_test = shuffled_data.iloc[train_size + validation_size :, -1]

    X_train.to_csv(os.path.join(split_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(split_dir, "y_train.csv"), index=False)
    X_validation.to_csv(os.path.join(split_dir, "X_validation.csv"), index=False)
    y_validation.to_csv(os.path.join(split_dir, "y_validation.csv"), index=False)
    X_test.to_csv(os.path.join(split_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(split_dir, "y_test.csv"), index=False)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def split(X, train_ratio=0.8, val_ratio=0.1):
    np.random.seed(1)
    indices = np.random.permutation(len(X))
    X_shuffled = X.iloc[indices]

    train_size = int(train_ratio * len(X_shuffled))
    val_size = int(val_ratio * len(X_shuffled))

    X_train = X_shuffled[:train_size]
    X_val = X_shuffled[train_size : train_size + val_size]
    X_test = X_shuffled[train_size + val_size :]
    return X_train, X_val, X_test


def autoencoder_knn_task():
    file_path_spotify = "data/interim/2/spotify_normalized_numerical.csv"
    df = pd.read_csv(file_path_spotify)
    genres = df["track_genre"].values
    features = df.drop(columns=["track_genre"]).values
    input_size = features.shape[1]
    latent_size = 9
    encoder_layers = [10]
    decoder_layers = [10]

    autoencoder = AutoEncoder(
        input_size=input_size,
        latent_size=latent_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )
    autoencoder.fit(features, max_epochs=10)
    reduced_features = autoencoder.get_latent(features)

    features_df = pd.DataFrame(reduced_features)
    result_df = features_df.copy()
    result_df["track_genre"] = genres
    train, val, _ = split(df)
    classifier = KNN(k=64, distance_metric="manhattan")
    classifier.fit(
        train.drop(columns=["track_genre"]).values, train["track_genre"].values
    )
    start_time = time.time()
    y_pred = classifier.predict(val.drop(columns=["track_genre"]).values)
    time_taken1 = time.time() - start_time
    Metrics(
        y_true=val["track_genre"].values, y_pred=y_pred, task="classification"
    ).print_metrics()
    print(f"{time_taken1=}")
    train, val, _ = split(result_df)
    classifier = KNN(k=64, distance_metric="manhattan")
    classifier.fit(
        train.drop(columns=["track_genre"]).values, train["track_genre"].values
    )
    start_time = time.time()
    y_pred = classifier.predict(val.drop(columns=["track_genre"]).values)
    time_taken2 = time.time() - start_time
    Metrics(
        y_true=val["track_genre"].values, y_pred=y_pred, task="classification"
    ).print_metrics()
    print(f"{time_taken2=}")
    models = [f"Original (12 dim)", f"AutoEncoder ({latent_size} dim)"]
    times = [time_taken1, time_taken2]
    plt.figure(figsize=(8, 5))
    plt.bar(models, times, color=["skyblue", "lightgreen"])
    plt.ylabel("Inference Time (seconds)")
    plt.title("KNN Inference Time Comparison (Original vs AutoEncoder)")
    plt.savefig("assignments/3/figures/knn_og_vs_autoencoder_bar_2.png")


def prepare_and_train_mlp(file_path, config):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Sort unique genres and create a mapping
    unique_genres = sorted(df['track_genre'].unique())
    genre_to_number = {genre: idx for idx, genre in enumerate(unique_genres)}
    df['track_genre'] = df['track_genre'].map(genre_to_number)

    # Split the data into features and target variable
    X = df.drop(columns=["track_genre"]).values
    y = df['track_genre'].values

    # Use your custom split function
    train, val, test = split(df)

    # Prepare training and validation sets
    X_train = train.drop(columns=["track_genre"]).values
    y_train = train["track_genre"].values
    X_validation = val.drop(columns=["track_genre"]).values
    y_validation = val["track_genre"].values
    X_test = test.drop(columns=["track_genre"]).values
    y_test = test["track_genre"].values

    # Initialize and fit the MLP Classifier
    model = MLPClassifier(
        input_size=X_train.shape[1],
        hidden_layers=config["hidden_layers"],
        num_classes=len(unique_genres),  # Number of unique classes
        learning_rate=config["lr"],
        activation=config["activation"],
        optimizer=config["optimizer"],
        print_every=1,
        wandb_log=False,
    )
    
    costs = model.fit(
        X_train,
        y_train,
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        X_validation=X_validation,
        y_validation=y_validation,
        early_stopping=True,
        patience=config["max_epochs"] // 20,
    )
    
    # Predict on the test set
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    test_metrics = Metrics(y_test, y_pred_test, task="classification")

    # Print the metrics
    print(f"Test Accuracy: {test_metrics.accuracy()}")
    print(f"Precision: {test_metrics.precision_score()}")
    print(f"Recall: {test_metrics.recall_score()}")
    print(f"F1 Score: {test_metrics.f1_score()}")

if __name__ == "__main__":
    np.random.seed(6)
    # wine_preprocessing()
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_wineqt()
    X_train_classification, y_train_classification = copy.deepcopy((X_train, y_train))

    # best_model_params = None
    # best_validation_accuracy = 0
    # sweep_id = wandb.sweep(sweep_config_wine, project="SMAI_A3")
    # wandb.agent(sweep_id, function=train_and_log_classification, count=512)
    # with open("data/interim/3/WineQT/best_model_config.json", "w") as f:
    #     json.dump(best_model_params, f)
    # wandb.finish()

    # print_hyperparams_wineqt()

    # test_on_best_wineqt()

    # analyze_model_impact()


    # advertisement_preprocessing()
    X_train, y_train, X_validation, y_validation, X_test, y_test, index_to_label = (get_advertisement_data())
    # best_model_params = None
    # best_validation_accuracy = 0

    # sweep_id = wandb.sweep(sweep_config_diabetes, project="SMAI_A3")
    # wandb.agent(sweep_id, function=train_and_log_multi_label_classification, count=64)

    # # Save the best model parameters to a JSON file
    # with open("data/interim/3/diabetes/best_model_config.json", "w") as f:
    #     json.dump(best_model_params, f)

    # wandb.finish()

    test_on_best_advertisement(index_to_label)

    # print_hyperparams_advertisement()

    # housing_preprocessing()
    # np.random.seed(13)
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_housing()
    X_train_regression, y_train_regression = copy.deepcopy((X_train, y_train))
    # best_model_params_regression = None
    # best_validation_mse = float("inf")
    # sweep_id_regression = wandb.sweep(sweep_config_housing, project="SMAI_A3")
    # wandb.agent(sweep_id_regression, function=train_and_log_regression, count=512)

    # with open("data/interim/3/HousingData/best_model_config.json", "w") as f:
    #     json.dump(best_model_params_regression, f)

    # wandb.finish()

    # test_on_best_housing()

    # X_train, y_train, X_validation, y_validation, X_test, y_test = process_diabetes_data()

    # model_bce = MLPLogisticRegression(input_size=X_train.shape[1], learning_rate=0.1, loss='bce')
    # model_bce.fit(X_train, y_train, max_epochs=100)

    # model_mse = MLPLogisticRegression(input_size=X_train.shape[1], learning_rate=0.1, loss='mse')
    # model_mse.fit(X_train, y_train, max_epochs=100)

    # predictions_bce = model_bce.predict(X_train)
    # predictions_mse = model_mse.predict(X_train)

    # # print("BCE Predictions:", predictions_bce.flatten())
    # print("BCE Accuracy",np.mean(predictions_bce.flatten() == y_train))
    # # print("MSE Predictions:", predictions_mse.flatten())
    # print("MSE Accuracy", np.mean(predictions_mse.flatten() == y_train))

    # autoencoder_knn_task()

    # config = {
    #     "hidden_layers": [32, 64],
    #     "lr": 0.01,
    #     "activation": "relu",
    #     "optimizer": "mbgd",
    #     "max_epochs": 100,
    #     "batch_size": 16,12
    # }
    # prepare_and_train_mlp("data/interim/2/spotify_normalized_numerical.csv", config)

    # unittest.main()
