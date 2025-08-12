import torch.nn as nn
import torch
from config import H, W

class CNNPolicy(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            conv_out = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(conv_out, 512), nn.ReLU())
        self.policy_logits = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        logits = self.policy_logits(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value