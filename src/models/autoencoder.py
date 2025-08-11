from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 48, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(128, latent_dim * 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(128, latent_dim * 2), latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(128, latent_dim * 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(128, latent_dim * 2), input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
