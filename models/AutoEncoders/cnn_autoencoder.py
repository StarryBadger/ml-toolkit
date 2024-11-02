import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import itertools

class CNNAutoencoder(nn.Module):
    def __init__(self, num_filters=[16, 32, 64], kernel_sizes=[3, 3, 7], activation=nn.ReLU, device = 'cpu'):
        super(CNNAutoencoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=kernel_sizes[0], stride=2, padding=1),
            activation(),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_sizes[1], stride=2, padding=1),
            activation(),
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_sizes[2])
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_filters[2], out_channels=num_filters[1], kernel_size=kernel_sizes[2]),
            activation(),
            nn.ConvTranspose2d(in_channels=num_filters[1], out_channels=num_filters[0], kernel_size=kernel_sizes[1], stride=2, padding=1, output_padding=1),
            activation(),
            nn.ConvTranspose2d(in_channels=num_filters[0], out_channels=1, kernel_size=kernel_sizes[0], stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def fit(self, train_loader, num_epochs=10, learning_rate=0.001, optimizer_choice='adam'):
        if optimizer_choice.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_choice.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        criterion = nn.MSELoss()
        self.train() 
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, _ in tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', unit='batch'):
                images = images.to(self.device)
                
                outputs = self.forward(images)
                loss = criterion(outputs, images)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    def evaluate(self, test_loader):
        criterion = nn.MSELoss()
        self.eval() 
        
        total_loss = 0.0
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc='Evaluating', unit='batch'):
                images = images.to(self.device)
                
                outputs = self.forward(images)
                loss = criterion(outputs, images)
                
                total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(test_loader.dataset)
        print(f'Test Loss: {avg_loss:.4f}')
        return avg_loss
