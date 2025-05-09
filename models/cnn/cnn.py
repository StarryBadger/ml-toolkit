import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, task='classification', num_classes=4, num_conv_layers=3, dropout_rate=0, optimizer_choice='adam', activation_function='relu', device='cpu', loss_figure_save_path='./figures/cnn_loss_plots/plot.png'):
        super(CNN, self).__init__()
        print(loss_figure_save_path)
        assert task in ['classification', 'regression'], "Task must be either 'classification' or 'regression'."
        assert activation_function in ['relu', 'sigmoid', 'tanh', 'leaky_relu'], "Activation function must be one of 'relu', 'sigmoid', 'tanh', or 'leaky_relu'."
        
        self.task = task
        self.device = device
        self.loss_figure_save_path = loss_figure_save_path
        
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
        }
        self.activation_function = activation_map[activation_function]
        
        layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = 32 * (2 ** i)
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
            layers.append(self.activation_function)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        self.fc1 = nn.Linear(((28 // (2 ** num_conv_layers)) ** 2) * 32 * (2 ** (num_conv_layers - 1)), 64)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, num_classes if task == 'classification' else 1)
        self.task = task
        self.optimizer_choice = optimizer_choice

    def forward(self, x, return_feature_maps=False):
        feature_maps = []

        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d): 
                feature_maps.append(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        
        self.output_activation = nn.LogSoftmax(dim=1) if self.task == 'classification' else nn.Identity()
        
        output = self.output_activation(x)
        return (output, feature_maps) if return_feature_maps else output


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
        epochs = range(1, len(history['train_loss']) + 1) 
        plt.plot(epochs, history['train_loss'], label="Training Loss")
        plt.plot(epochs, history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(self.loss_figure_save_path)
        plt.close()
