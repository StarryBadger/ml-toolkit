import numpy as np
import sys
sys.path.append('./../../')
from models.MLP.MLPRegression import MLPRegression
class AutoEncoder:
    def __init__(self, input_size, latent_size, encoder_layers, decoder_layers, learning_rate=0.01, activation='sigmoid', optimizer='sgd', wandb_log=False):
        self.encoder = MLPRegression(input_size=input_size, hidden_layers=encoder_layers, output_size=latent_size, 
                                     learning_rate=learning_rate, activation=activation, optimizer=optimizer, wandb_log=wandb_log)
        self.decoder = MLPRegression(input_size=latent_size, hidden_layers=decoder_layers, output_size=input_size, 
                                     learning_rate=learning_rate, activation=activation, optimizer=optimizer, wandb_log=wandb_log)

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
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss}")

                
    def get_latent(self, X):
        return self.encoder.forward_propagation(X)
    
    def reconstruct(self, X):
        latent_representation = self.encoder.forward_propagation(X)
        reconstructed_X = self.decoder.forward_propagation(latent_representation)
        return reconstructed_X
