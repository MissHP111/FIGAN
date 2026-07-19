import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim=100, sequence_len=256, vocab_size=17):
        super().__init__()
        self.fc1 = nn.Linear(noise_dim, 512)
        self.fc2 = nn.Linear(512, sequence_len * 64)
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, vocab_size, kernel_size=3, padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 256)
        x = F.relu(self.conv1(x))
        logits = self.conv2(x)
        return logits.permute(0, 2, 1)  # [B, 256, vocab_size]


class Critic(nn.Module):
    def __init__(self, sequence_len=256, vocab_size=17):
        super().__init__()
        self.conv1 = nn.Conv1d(vocab_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * sequence_len, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(x.size(0), -1)
        return self.fc(x)
