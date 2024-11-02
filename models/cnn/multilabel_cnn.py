import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class MultiLabelCNN(nn.Module):
    def __init__(self, 
                 num_conv_layers=3, 
                 dropout_rate=0.5, 
                 optimizer_choice='adam', 
                 activation_function='relu', 
                 device='cpu',
                 loss_figure_save_path='./figures/cnn_loss_plots/plot.png'):
        super(MultiLabelCNN, self).__init__()
        
        self.device = device
        self.optimizer_choice = optimizer_choice
        self.num_conv_layers = num_conv_layers
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
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(self.activation_function)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        fc_input_size = ((28 // (2 ** num_conv_layers)) ** 2) * 32 * (2 ** (num_conv_layers - 1))
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc3 = nn.Linear(256, 33) 
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation_function(self.fc1(x)))
        x = self.fc3(x)  
        return x

    def fit(self, train_loader, val_loader, epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = self._get_optimizer(lr)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                optimizer.zero_grad()
                
                outputs = self(inputs)
                labels = labels.float()
                loss = sum(criterion(outputs[:, i:i+11], labels[:, i:i+11]) for i in range(0, 33, 11))
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Calculate exact match accuracy
                preds = self._set_segment_max_to_one(outputs)
                correct += self.exact_match_accuracy(preds, labels)
                total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = correct / total * 100
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_acc:.2f}%")

            val_loss, val_acc = self.evaluate(val_loader, criterion)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        self.plot_loss(history)

    def evaluate(self, loader, criterion):
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                outputs = self(inputs)
                labels = labels.float()
                loss = sum(criterion(outputs[:, i:i+11], labels[:, i:i+11]) for i in range(0, 33, 11))
                total_loss += loss.item()

                preds = self._set_segment_max_to_one(outputs)
                correct += self.exact_match_accuracy(preds, labels)
                total += labels.size(0)

        avg_loss = total_loss / len(loader)
        avg_acc = correct / total * 100
        return avg_loss, avg_acc

    def predict(self, loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device).to(torch.float32)
                outputs = self(inputs)
                preds = self._set_segment_max_to_one(outputs)
                predictions.append(preds.cpu())
        return torch.cat(predictions)
    
    def _get_optimizer(self, lr):
        if self.optimizer_choice.lower() == 'adam':
            return optim.Adam(self.parameters(), lr=lr)
        elif self.optimizer_choice.lower() == 'sgd':
            return optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer choice: choose 'adam' or 'sgd'.")

    def _set_segment_max_to_one(self, tensor):
        modified_tensor = torch.zeros_like(tensor)
        
        for i in range(tensor.shape[0]):
            segment1 = tensor[i, 0:11]
            max_index1 = torch.argmax(segment1)
            modified_tensor[i, 0:11] = 0
            modified_tensor[i, max_index1] = 1 
            
            segment2 = tensor[i, 11:22]
            max_index2 = torch.argmax(segment2)
            modified_tensor[i, 11:22] = 0 
            modified_tensor[i, 11 + max_index2] = 1 

            segment3 = tensor[i, 22:33]
            max_index3 = torch.argmax(segment3)
            modified_tensor[i, 22:33] = 0  
            modified_tensor[i, 22 + max_index3] = 1  
        
        return modified_tensor

    def exact_match_accuracy(self, preds, labels):
        """Calculates the exact match accuracy between predictions and labels."""
        return (preds == labels).all(dim=1).float().sum().item()

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

        # Optional: Plot accuracy as well
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['train_acc'], label="Training Accuracy")
        plt.plot(epochs, history['val_acc'], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.savefig(self.loss_figure_save_path.replace('.png', '_accuracy.png'))
        plt.close()
