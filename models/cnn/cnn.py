import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, task='classification', num_classes=4, num_conv_layers=3, dropout_rate=0, optimizer_choice='adam', device='cpu'):
        super(CNN, self).__init__()
        assert task in ['classification', 'regression'], "Task must be either 'classification' or 'regression'."
        self.task = task
        self.device = device
        
        layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = 32 * (2 ** i)
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        self.fc1 = nn.Linear(((28//(2**num_conv_layers))**2)*32*(2**(num_conv_layers-1)), 64)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, num_classes if task == 'classification' else 1)
        
        self.output_activation = nn.LogSoftmax(dim=1) if task == 'classification' else nn.Identity()
        
        self.optimizer_choice = optimizer_choice

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return self.output_activation(x)

    def fit(self, train_loader, val_loader, epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss() if self.task == 'classification' else nn.MSELoss()
        optimizer = self._get_optimizer(lr)

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.task == 'regression':
                    labels = labels.float()
                optimizer.zero_grad()
                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            val_loss = self.evaluate(val_loader, criterion)
            history['val_loss'].append(val_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        self.plot_loss(history)

    def evaluate(self, loader, criterion):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        return avg_loss

    def predict(self, loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                preds = torch.argmax(outputs, dim=1) if self.task == 'classification' else outputs.squeeze()
                predictions.append(preds.cpu())
        return torch.cat(predictions)

    def _get_optimizer(self, lr):
        if self.optimizer_choice.lower() == 'adam':
            return optim.Adam(self.parameters(), lr=lr)
        elif self.optimizer_choice.lower() == 'sgd':
            return optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer choice: choose 'adam' or 'sgd'.")

    def plot_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label="Training Loss")
        plt.plot(history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
