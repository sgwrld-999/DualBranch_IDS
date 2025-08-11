from __future__ import annotations

import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    def __init__(self, in_features: int, channels: int = 64, kernel_sizes=(3, 3), dropout: float = 0.25):
        super().__init__()
        k1, k2 = kernel_sizes
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=k2, padding=k2 // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),  # GlobalMaxPooling1D
            nn.Flatten(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, F]
        return self.net(x)
