
import os
import random
import nltk
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class OCRModel(nn.Module):
    def __init__(self, input_dim=65536, hidden_dim=256, num_classes=53, num_layers=2):
        super(OCRModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = x.unsqueeze(1).repeat(1, 32, 1)
        rnn_out, _ = self.rnn(x)
        rnn_out = self.layer_norm(rnn_out)
        output = self.fc(rnn_out)
        return output
