from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionLatent(nn.Module):
    """
    Self-attention over the latent vector (feature-wise gating).
    Computes importance weights per latent dimension.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),  # importance weights in [0,1]
        )

    def forward(self, z: torch.Tensor):
        # z: [B, D]
        weights = self.score(z)
        z_att = z * weights
        return z_att, weights
