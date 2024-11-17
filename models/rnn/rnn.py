import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Dataset Class
class BitSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Custom collate function to handle padding and return lengths
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sequences, labels, lengths

# Dataset generation
def generate_dataset(num_sequences=100_000, max_length=16):
    sequences = []
    labels = []
    for _ in range(num_sequences):
        length = np.random.randint(1, max_length + 1)
        sequence = np.random.randint(0, 2, length).astype(np.float32)
        label = np.sum(sequence)
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels

def prepare_dataloaders(sequences, labels, batch_size=64, splits=(0.8, 0.1, 0.1)):
    dataset = BitSequenceDataset(sequences, labels)
    train_size = int(len(dataset) * splits[0])
    val_size = int(len(dataset) * splits[1])
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

# RNN Class
class CountingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(CountingRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output: single count value

    def forward(self, x, lengths):
        # Initialize hidden state
        h_0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_x, h_0)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc(out[torch.arange(out.size(0)), lengths - 1])  # Take last valid timestep
        return out.squeeze(-1)

# Loss computation with mask
def compute_loss_with_mask(outputs, labels, lengths, criterion):
    mask = torch.arange(outputs.size(0), device=outputs.device) < lengths
    outputs = outputs.masked_select(mask)
    labels = labels.masked_select(mask)
    return criterion(outputs, labels)

# Training and Evaluation Functions
def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.00008, device='cpu'):
    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(device).unsqueeze(-1)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = compute_loss_with_mask(outputs, labels, lengths, criterion)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    total_loss = 0
    criterion = nn.L1Loss()  # MAE

    with torch.no_grad():
        for sequences, labels, lengths in data_loader:
            sequences = sequences.to(device).unsqueeze(-1)
            labels = labels.to(device)
            outputs = model(sequences, lengths)
            loss = compute_loss_with_mask(outputs, labels, lengths, criterion)
            total_loss += loss.item() * sequences.size(0)

    return total_loss / len(data_loader.dataset)

# Generalization Testing
def generalization_test(model, max_length=32, device='cpu'):
    model.to(device)
    lengths = list(range(1, max_length + 1))
    maes = []

    for length in lengths:
        sequences, labels = generate_dataset(num_sequences=1000, max_length=length)
        dataset = BitSequenceDataset(sequences, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        mae = evaluate_model(model, loader, device)
        maes.append(mae)

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(lengths, maes, marker='o', label='Generalization MAE')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Absolute Error')
    plt.title('Generalization Across Sequence Lengths')
    plt.legend()
    plt.grid(True)
    plt.savefig('generalization_plot.png')

# Main Execution
if __name__ == "__main__":
    # Dataset Preparation
    sequences, labels = generate_dataset()
    train_loader, val_loader, test_loader = prepare_dataloaders(sequences, labels)

    # Model Initialization
    model = CountingRNN(input_size=1, hidden_size=32)

    # Training
    train_model(model, train_loader, val_loader)

    # Load the best model for testing
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))


    # Generalization Testing
    generalization_test(model)
