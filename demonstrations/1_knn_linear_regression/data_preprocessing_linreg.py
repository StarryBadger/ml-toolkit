import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def report_metrics(data, set_name):
    x_values = data[:, 0]
    y_values = data[:, 1]
    print(f"{set_name} Set Metrics:") 
    print(f"x - Mean: {np.mean(x_values)}, Variance: {np.var(x_values)}, Std Dev: {np.std(x_values)}, Min: {np.min(x_values)}, Max: {np.max(x_values)}")
    print(f"y - Mean: {np.mean(y_values)}, Variance: {np.var(y_values)}, Std Dev: {np.std(y_values)}, Min: {np.min(y_values)}, Max: {np.max(y_values)}\n")

def preproceess(path, save_folder, name):
    print(name)
    data=pd.read_csv(path).to_numpy()
    np.random.shuffle(data)

    train_size = int(0.8 * len(data))
    valid_size = int(0.1 * len(data))

    train_data = data[:train_size]
    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:]

    cols=['x', 'y']
    train_df = pd.DataFrame(train_data, columns=cols)
    valid_df = pd.DataFrame(valid_data, columns=cols)
    test_df = pd.DataFrame(test_data, columns=cols)

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    X=train_df['x'].to_numpy()
    y=train_df['y'].to_numpy()
    plt.scatter(X, y, color='blue', label='Original Data', alpha= 0.5)
    plt.savefig(f'assignments/1/figures/{name}.png')
    plt.close()


    train_df.to_csv(f'{save_folder}/train.csv', index=False)
    valid_df.to_csv(f'{save_folder}/validate.csv', index=False)
    test_df.to_csv(f'{save_folder}/test.csv', index=False)

    report_metrics(train_data, "Train")
    report_metrics(valid_data, "Validation")
    report_metrics(test_data, "Test")

preproceess("data/external/linreg.csv","data/interim/1/linreg", 'first_dataset')
preproceess("data/external/regularisation.csv","data/interim/1/regularisation", 'second_dataset')
