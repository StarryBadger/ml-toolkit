import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
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
from models.rnn.rnn import RNNBitCounter

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


def extract_mfcc_features(audio_data, sr=4000, n_mfcc=13, n_fft=512, hop_length=256):
    mfcc_features = []
    for signal in audio_data:
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_features.append(mfcc.T)  
    return mfcc_features

def visualize_mfcc(mfcc_features, labels, digit=2, save_path='assignments/5/figures/spectrogram'):
    digit_samples = [mfcc_features[i] for i in range(len(labels)) if labels[i] == digit]
    if len(digit_samples) == 0:
        print(f"No samples found for digit {digit}.")
        return
    plt.figure(figsize=(10, 4))
    sns.heatmap(digit_samples[0].T, cmap='viridis', cbar=True)
    plt.title(f'MFCC Spectrogram for Digit {digit}')
    plt.xlabel('Time Frames')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.savefig(f"{save_path}_{digit}.png")



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
    kde = KDE(kernel='triangular', bandwidth=0.5)
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

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in sequences], 
                                     batch_first=True, 
                                     padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_sequences, labels

def generate_bit_sequences(num_samples=100000, max_length=16):
    sequences = []
    labels = []
    for _ in range(num_samples):
        length = np.random.randint(1, max_length + 1)
        sequence = np.random.randint(0, 2, size=(length,)).tolist()
        count = sum(sequence)
        sequences.append(sequence)
        labels.append(count)
    return sequences, labels

def split_dataset(sequences, labels, train_ratio=0.8, val_ratio=0.1):
    total = len(sequences)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)

    train_seq, train_labels = sequences[:train_size], labels[:train_size]
    val_seq, val_labels = sequences[train_size:train_size + val_size], labels[train_size:train_size + val_size]
    test_seq, test_labels = sequences[train_size + val_size:], labels[train_size + val_size:]

    return (train_seq, train_labels), (val_seq, val_labels), (test_seq, test_labels)

class BitSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for seqs, counts in dataloader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, counts)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for seqs, counts in dataloader:
            outputs = model(seqs)
            loss = criterion(outputs, counts)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_random_baseline(loader):
    criterion = nn.L1Loss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for seqs, counts in loader:
            random_preds = torch.tensor([np.random.randint(0, len(seq)) for seq in seqs], dtype=torch.float32)
            loss = criterion(random_preds, counts)
            total_loss += loss.item() * len(counts)
            total_samples += len(counts)
    return total_loss / total_samples



def main():
    kde_load_and_plot_dataset()
    apply_gmm_and_plot(2)
    kde_fit_and_visualize()
    
    audio_data, labels = load_audio_files()
    mfcc_features = extract_mfcc_features(audio_data)
    visualize_mfcc(mfcc_features, labels, digit=3)
    mfcc_train, mfcc_test, labels_train, labels_test = train_test_split(mfcc_features, labels, test_size=0.2, random_state=42)
    models = train_hmm_models(mfcc_train, labels_train)
    test_accuracy = evaluate_model(models, mfcc_test, labels_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    my_audio_data, my_labels = load_my_audio_files(path="data/interim/5/fsdd/my_voice")
    my_mfcc_features = extract_mfcc_features(my_audio_data)
    my_test_accuracy = evaluate_model(models, my_mfcc_features, my_labels)
    print(f"Test Accuracy on My Voice: {my_test_accuracy * 100:.2f}%")

    sequences, labels = generate_bit_sequences()
    train_data, val_data, test_data = split_dataset(sequences, labels)
    hidden_size = 32
    num_layers = 1
    learning_rate = 0.0001
    batch_size = 32
    epochs = 5

    train_loader = DataLoader(BitSequenceDataset(*train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(BitSequenceDataset(*val_data), batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(BitSequenceDataset(*test_data), batch_size=batch_size, collate_fn=collate_fn)

    model = RNNBitCounter(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.L1Loss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_model(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    lengths = list(range(1, 33))
    rnn_mae_per_length = []
    random_mae_per_length = []

    for length in lengths:
        sequences, labels = generate_bit_sequences(num_samples=1000, max_length=length)
        loader = DataLoader(BitSequenceDataset(sequences, labels), batch_size=batch_size, collate_fn=collate_fn)
        
        rnn_mae = evaluate_model(model, loader, criterion)
        rnn_mae_per_length.append(rnn_mae)
        
        random_mae = evaluate_random_baseline(loader)
        random_mae_per_length.append(random_mae)

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, rnn_mae_per_length, marker='o', label="RNN MAE")
    plt.xlabel("Sequence Length")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE vs sequence length")
    plt.legend()
    plt.grid()
    plt.ylim(0, 1)
    plt.savefig('./assignments/5/figures/mae_vs_sequence.png')


    plt.figure(figsize=(10, 6))
    plt.plot(lengths, rnn_mae_per_length, marker='o', label="RNN MAE")
    plt.plot(lengths, random_mae_per_length, marker='s', linestyle="--", label="Random Baseline MAE")
    plt.xlabel("Sequence Length")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE Comparison: RNN vs Random Baseline")
    plt.legend()
    plt.grid()
    plt.savefig('assignments/5/figures/mae_comparison_rnn_vs_random.png')
    plt.close()

    for length, rnn_mae, random_mae in zip(lengths, rnn_mae_per_length, random_mae_per_length):
        print(f"Length: {length}, RNN Mean Absolute Error: {rnn_mae:.4f}, Random Baseline Mean Absolute Error: {random_mae:.4f}")


if __name__ == "__main__":
    main()

