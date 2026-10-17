from __future__ import annotations

import math

import torch
from torch import nn


class DeepLOB(nn.Module):
    """CNN/Inception/LSTM architecture matching the authors' released layout."""

    def __init__(self, classes: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 2), stride=(1, 2)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.branch1 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(64), nn.Conv2d(64, 64, (3, 1), padding=(1, 0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.branch2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(64), nn.Conv2d(64, 64, (5, 1), padding=(2, 0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.branch3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)), nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.lstm = nn.LSTM(192, 64, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x)), dim=1)
        x = x.squeeze(-1).permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.classifier(self.dropout(x[:, -1]))


class TransformerLOB(nn.Module):
    def __init__(
        self,
        features: int = 40,
        length: int = 100,
        classes: int = 3,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(features, 64)
        self.position = nn.Parameter(torch.empty(1, length, 64))
        layer = nn.TransformerEncoderLayer(
            64,
            4,
            128,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(64))
        self.classifier = nn.Linear(64, classes)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x) * math.sqrt(64.0) + self.position[:, : x.shape[1]]
        return self.classifier(self.encoder(hidden)[:, -1])


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
