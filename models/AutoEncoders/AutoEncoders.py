import numpy as np
from models.MLP.MLPRegression import MLPRegression
class AutoEncoder:
    def __init__(self, input_size, latent_size, encoder_layers, decoder_layers, learning_rate=0.01, activation='relu', optimizer='sgd', wandb_log=False):
        self.encoder = MLPRegression(input_size=input_size, hidden_layers=encoder_layers, output_size=latent_size, 
                                     learning_rate=learning_rate, activation=activation, optimizer=optimizer, wandb_log=wandb_log)
        self.decoder = MLPRegression(input_size=latent_size, hidden_layers=decoder_layers, output_size=input_size, 
                                     learning_rate=learning_rate, activation=activation, optimizer=optimizer, wandb_log=wandb_log)

    def fit(self, X_train, X_validation=None, max_epochs=10, batch_size=32, early_stopping=False, patience=100):
        for epoch in range(max_epochs):
            latent_representation = self.encoder.forward_propagation(X_train)
            reconstructed_X = self.decoder.forward_propagation(latent_representation)
            loss = np.mean((X_train - reconstructed_X) ** 2)
            self.decoder.backpropagation(latent_representation, X_train, reconstructed_X)
            self.decoder.update_parameters()
            self.encoder.backpropagation(X_train, latent_representation, latent_representation)
            self.encoder.update_parameters()
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {loss}")
                
    def get_latent(self, X):
        return self.encoder.forward_propagation(X)