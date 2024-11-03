import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./../../')
from models.MLP.MLPRegression import MLPRegression

class AutoEncoder:
    def __init__(self, input_size, latent_size, encoder_layers, decoder_layers, learning_rate=0.01, 
                 activation='sigmoid', optimizer='sgd', wandb_log=False, plot_file_path=None):
        self.encoder = MLPRegression(input_size=input_size, hidden_layers=encoder_layers, output_size=latent_size, 
                                     learning_rate=learning_rate, activation=activation, optimizer=optimizer, 
                                     wandb_log=wandb_log)
        self.decoder = MLPRegression(input_size=latent_size, hidden_layers=decoder_layers, output_size=input_size, 
                                     learning_rate=learning_rate, activation=activation, optimizer=optimizer, 
                                     wandb_log=wandb_log)
        self.plot_file_path = plot_file_path
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train, X_validation=None, max_epochs=10, batch_size=32, early_stopping=False, patience=100):
        num_samples = X_train.shape[0]
        for epoch in range(max_epochs):
            epoch_loss = 0
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                
                latent_representation = self.encoder.forward_propagation(X_batch)
                reconstructed_X = self.decoder.forward_propagation(latent_representation)
                
                loss = np.mean((X_batch - reconstructed_X) ** 2)
                epoch_loss += loss
                
                self.decoder.backpropagation(latent_representation, X_batch, reconstructed_X)
                self.decoder.update_parameters()
                self.encoder.backpropagation(X_batch, latent_representation, latent_representation)
                self.encoder.update_parameters()

            epoch_loss /= (num_samples / batch_size)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss}")

            if X_validation is not None:
                val_loss = self.validate(X_validation)
                self.val_losses.append(val_loss)

        self.plot_loss()

    def validate(self, X_validation):
        reconstructed_X = self.reconstruct(X_validation)
        val_loss = np.mean((X_validation - reconstructed_X) ** 2)
        print(f"Validation Loss: {val_loss}")
        return val_loss

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        if self.plot_file_path:
            plt.savefig(self.plot_file_path)
        plt.show()

    def get_latent(self, X):
        return self.encoder.forward_propagation(X)
    
    def reconstruct(self, X):
        latent_representation = self.encoder.forward_propagation(X)
        reconstructed_X = self.decoder.forward_propagation(latent_representation)
        return reconstructed_X
