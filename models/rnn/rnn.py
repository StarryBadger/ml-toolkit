import torch
import torch.nn as nn

class RNNBitCounter(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super(RNNBitCounter, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity="relu", dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)