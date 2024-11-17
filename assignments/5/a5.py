import os
import pandas as pd
import librosa
import numpy as np
np.random.seed(42) 
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from models.kde.kde import KDE
from models.gmm.gmm import GMM

DATA_PATH = "data/interim/5/fsdd/recordings/"

def load_audio_files(path=DATA_PATH):
    audio_data = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            label = int(filename.split("_")[0]) 
            filepath = os.path.join(path, filename)
            signal, sr = librosa.load(filepath, sr=None)
            audio_data.append(signal)
            labels.append(label)
    return audio_data, labels

def load_my_audio_files(path="data/interim/5/fsdd/my_voice"):
    audio_data = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            label = int(filename[0])  
            filepath = os.path.join(path, filename)
            signal, sr = librosa.load(filepath, sr=None)
            audio_data.append(signal)
            labels.append(label)
    
    return audio_data, labels


def extract_mfcc_features(audio_data, sr=4000, n_mfcc=13, n_fft=512, hop_length=64):
    mfcc_features = []
    for signal in audio_data:
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_features.append(mfcc.T)  
    return mfcc_features

def visualize_mfcc(mfcc_features, digit=0):
    plt.figure(figsize=(10, 4))
    sns.heatmap(mfcc_features[digit].T, cmap='viridis')
    plt.title(f'MFCC for Digit {digit}')
    plt.xlabel('Time Frames')
    plt.ylabel('MFCC Coefficients')
    plt.savefig('assignments/5/figures/spectrogram.png')


def train_hmm_models(mfcc_features, labels, n_states=5):
    models = {}
    for digit in range(10):
        digit_features = [mfcc for mfcc, label in zip(mfcc_features, labels) if label == digit]
        X = np.concatenate(digit_features)
        lengths = [len(mfcc) for mfcc in digit_features]
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100)
        model.fit(X, lengths)
        models[digit] = model
    return models

def predict_digit(models, mfcc):
    max_score = float('-inf')
    best_digit = None
    for digit, model in models.items():
        score = model.score(mfcc)
        if score > max_score:
            max_score = score
            best_digit = digit
    return best_digit

def evaluate_model(models, mfcc_features, labels):
    predictions = [predict_digit(models, mfcc) for mfcc in mfcc_features]
    accuracy = accuracy_score(labels, predictions)
    return accuracy

def generate_disc_data(n_samples, center, radius, noise_factor):
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.sqrt(np.random.uniform(0, 1, n_samples)) * radius
    r += np.random.normal(0, radius*noise_factor, n_samples)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    points = np.column_stack((x, y))
    return points[:n_samples]

def kde_generate_and_save_dataset():
    large_disc = generate_disc_data(
        n_samples=3000,
        center=(0, 0),
        radius=2.0,
        noise_factor=0.08
    )
    small_disc = generate_disc_data(
        n_samples=500,
        center=(1, 1),
        radius=0.3,
        noise_factor=0.15
    )
    X = np.vstack([large_disc, small_disc])
    df = pd.DataFrame(X, columns=['X', 'Y'])
    df.to_csv('data/interim/5/kde_dataset.csv', index=False)
    print(f"Total number of points: {len(X)}")
    print(f"Points in large disc: {len(large_disc)}")
    print(f"Points in small disc: {len(small_disc)}")
    return X

def kde_load_and_plot_dataset():
    df = pd.read_csv('data/interim/5/kde_dataset.csv')
    X = df.values
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.grid(True)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title('Generated Dataset with Two Overlapping Filled Discs')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.savefig('assignments/5/figures/kde_dataset.png')
    plt.close()


def kde_fit_and_visualize():
    df = pd.read_csv('data/interim/5/kde_dataset.csv')
    X = df.values
    kde = KDE(kernel='box', bandwidth=0.6)
    kde.fit(X)
    point = np.array([0, 0])
    print(f"Density at {point}: {kde.predict(point)}")
    kde.visualize()

def apply_gmm_and_plot(k=2):
    df = pd.read_csv('data/interim/5/kde_dataset.csv')
    X = df.values
    gmm = GMM(k=k)  
    gmm.fit(X) 
    assignments = gmm.getHardAssignments(X)
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=assignments, s=5)
    plt.title(f'GMM Clustering with {k} Clusters')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.savefig('assignments/5/figures/gmm.png')

def main():
    # kde_load_and_plot_dataset()
    # apply_gmm_and_plot(2)
    kde_fit_and_visualize()
    
    # audio_data, labels = load_audio_files()
    
    # mfcc_features = extract_mfcc_features(audio_data)
    # visualize_mfcc(mfcc_features, digit=0)
    # mfcc_train, mfcc_test, labels_train, labels_test = train_test_split(mfcc_features, labels, test_size=0.2, random_state=42)
    # models = train_hmm_models(mfcc_train, labels_train)
    # test_accuracy = evaluate_model(models, mfcc_test, labels_test)
    # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    # my_audio_data, my_labels = load_my_audio_files(path="data/interim/5/fsdd/my_voice")
    # my_mfcc_features = extract_mfcc_features(my_audio_data)
    # my_test_accuracy = evaluate_model(models, my_mfcc_features, my_labels)
    # print(f"Test Accuracy on My Voice: {my_test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

