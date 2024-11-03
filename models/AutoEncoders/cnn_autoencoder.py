import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class CNNAutoencoder(nn.Module):
    def __init__(self, num_filters=[16, 32, 64], kernel_sizes=[3, 3, 7], activation=nn.ReLU, device='cpu', save_path='figures/cnn_autoencoder_loss_plots/plot.png'):
        super(CNNAutoencoder, self).__init__()
        self.device = device
        self.save_path = save_path
        
        encoder_layers = []
        in_channels = 1
        
        for i in range(len(num_filters)):
            encoder_layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_sizes[i], stride=2, padding=1))
            encoder_layers.append(activation())
            in_channels = num_filters[i]
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for i in reversed(range(len(num_filters))):
            out_channels = num_filters[i - 1] if i > 0 else 1 
            decoder_layers.append(nn.ConvTranspose2d(num_filters[i], out_channels, kernel_size=kernel_sizes[i], stride=2, padding=1, output_padding=1 if i < len(num_filters) - 1 else 0))
            if i > 0:  
                decoder_layers.append(activation())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.train_losses = []
        self.val_losses = []

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def fit(self, train_loader, val_loader=None, num_epochs=10, learning_rate=0.001, optimizer_choice='adam'):
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
            self.train_losses.append(epoch_loss)  # Store training loss
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

            # Validation phase
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)  # Store validation loss

    def validate(self, val_loader):
        criterion = nn.MSELoss()
        self.eval() 
        
        total_loss = 0.0
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc='Evaluating', unit='batch'):
                images = images.to(self.device)
                
                outputs = self.forward(images)
                loss = criterion(outputs, images)
                
                total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(val_loader.dataset)
        print(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path)

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
