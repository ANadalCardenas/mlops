"""
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, n_features, n_hidden, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)
 """   


import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs, in_channels=1):
        """
        n_features: kept for compatibility with your config (4096), not used directly.
        n_hidden:   size of the hidden fully-connected layer.
        n_outputs:  number of classes (e.g. 15 for Chinese MNIST).
        in_channels: number of input channels (1 for grayscale).
        """
        super().__init__()

        # Convolutional feature extractor
        self.cnn = nn.Sequential(
            nn.ConstantPad2d(2, 0),
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Compute flattened size: for 64×64 input after pad+2 pools = 16×17×17
        conv_output_dim = 16 * 17 * 17

        self.mlp = nn.Sequential(
            nn.Linear(conv_output_dim, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1) 
        x = self.mlp(x)
        return x
