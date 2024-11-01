import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MultiLabelCNN(nn.Module):
    def __init__(self, 
                 num_conv_layers=3, 
                 dropout_rate=0.5, 
                 optimizer_choice='adam', 
                 activation_function='relu', 
                 device='cpu'):
        super(MultiLabelCNN, self).__init__()
        
        self.device = device
        self.optimizer_choice = optimizer_choice
        self.num_conv_layers = num_conv_layers

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
        self.fc1 = nn.Linear(fc_input_size, 128)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, 33) 
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation_function(self.fc1(x)))
        # x = self.dropout(self.activation_function(self.fc2(x)))
        x = self.fc3(x)  
        return x

    def fit(self, train_loader, val_loader, epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = self._get_optimizer(lr)

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                optimizer.zero_grad()
                
                outputs = self(inputs)
                labels = labels.float()
                # loss = sum(criterion(outputs[:, i:i+11], labels[:, i:i+11]) for i in range(0, 33, 11))
                new_outputs = torch.stack([outputs[:, i:i+11] for i in range(0, 33, 11)], dim=1)
                new_labels = torch.stack([labels[:, i:i+11] for i in range(0, 33, 11)], dim=1)
                loss = criterion(new_outputs,new_labels)

                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")

            val_loss = self.evaluate(val_loader, criterion)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

    def evaluate(self, loader, criterion):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                outputs = self(inputs)
                outputs = outputs.squeeze()
                labels = labels.float()
                # loss = sum(criterion(outputs[:, i:i+11], labels[:, i:i+11]) for i in range(0, 33, 11))
                new_outputs = torch.stack([outputs[:, i:i+11] for i in range(0, 33, 11)], dim=1)
                new_labels = torch.stack([labels[:, i:i+11] for i in range(0, 33, 11)], dim=1)
                loss = criterion(new_outputs,new_labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        return avg_loss

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
