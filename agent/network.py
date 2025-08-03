import torch.nn as nn
import torch
import numpy as np

class MarioNet(nn.Module):
    def __init__(self, input_dims, n_actions, device = 'cpu'):
        super().__init__()
        self.device = device
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 8, stride=4).to(device),  # Input: [channels, H, W]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2).to(device),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1).to(device),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dynamically calculate linear layer input size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_dims).to(device)  # [1, channels, H, W]
            n_flatten = self.feature_extractor(dummy).shape[1]
        
        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 512).to(device),
            nn.ReLU(),
            nn.Linear(512, n_actions).to(device),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 512).to(device),
            nn.ReLU(),
            nn.Linear(512, 1).to(device)
        )

    def forward(self, x):
        # Input x: [batch_size, channels, H, W]
        features = self.feature_extractor(x)
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value