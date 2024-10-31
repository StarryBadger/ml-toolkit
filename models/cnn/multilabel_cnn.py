import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MultiLabelCNN(nn.Module):
    def __init__(self, device='cpu'):
        super(MultiLabelCNN, self).__init__()

        self.device = device

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 33)  

    def forward(self, x):
        x = x.to(torch.float32)  
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def fit(self, train_loader, val_loader, epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()  # Ensure labels are long
                optimizer.zero_grad()
                
                outputs = self(inputs)
                labels = labels.float()
                loss = sum(criterion(outputs[:, i:i+11], labels[:, i:i+11]) for i in range(0, 33, 11))
                
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
                loss = sum(criterion(outputs[:, i:i+11], labels[:, i:i+11]) for i in range(0, 33, 11))
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
