import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np

def describe_dataset(file_path):
    df = pd.read_csv(file_path)
    numerical_cols = df.drop(columns=["Id"]).to_numpy()
    results = []
    means = np.mean(numerical_cols, axis=0)
    stds = np.std(numerical_cols, axis=0)
    mins = np.min(numerical_cols, axis=0)
    maxs = np.max(numerical_cols, axis=0)
    results = np.column_stack((df.columns[:-1], means, stds, mins, maxs))
    results_df = pd.DataFrame(results, columns=['Attribute', 'Mean', 'Standard Deviation', 'Min', 'Max'])
    md_table = results_df.to_markdown(index=False)
    print(md_table)
    results_df.to_csv('data/interim/3/WineQT/WineQT_description.csv', index=False)
    return results_df

def plot_quality_distribution(file_path):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 6))
    df['quality'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution of Wine Quality')
    plt.xlabel('Quality')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.savefig('assignments/3/figures/quality_distribution.png')
    plt.close()


def normalize_and_standardize(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)

    min_max_scaler = MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(df.drop(columns=['Id', 'quality']))

    standard_scaler = StandardScaler()
    standardized_data = standard_scaler.fit_transform(df.drop(columns=['Id', 'quality']))

    normalized_df = pd.DataFrame(normalized_data, columns=df.columns[0:-2])  # Excluding 'Id' and 'quality'
    standardized_df = pd.DataFrame(standardized_data, columns=df.columns[0:-2])

    normalized_df['Id'] = df['Id']
    standardized_df['Id'] = df['Id']
    normalized_df['quality'] = df['quality']
    standardized_df['quality'] = df['quality']

    normalized_df.to_csv('data/interim/3/WineQT/WineQT_normalized.csv', index=False)
    standardized_df.to_csv('data/interim/3/WineQT/WineQT_standardized.csv', index=False)
    return normalized_df, standardized_df

if __name__ == "__main__":
    file_path = 'data/interim/3/WineQT/WineQT.csv'
    description = describe_dataset(file_path)
    plot_quality_distribution(file_path)
    normalized_data, standardized_data = normalize_and_standardize(file_path)

