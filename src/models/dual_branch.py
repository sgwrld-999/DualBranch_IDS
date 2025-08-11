from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_branch import CNNBranch
from .bilstm_branch import BiLSTMBranch
from .autoencoder import BottleneckAutoencoder
from .attention import SelfAttentionLatent


class DualBranchModel(nn.Module):
    def __init__(self, in_features: int, cfg):
        super().__init__()
        self.cnn = CNNBranch(
            in_features=in_features,
            channels=cfg.cnn_channels,
            kernel_sizes=cfg.cnn_kernel_sizes,
            dropout=cfg.cnn_dropout,
        )
        self.lstm = BiLSTMBranch(
            in_features=in_features,
            hidden=cfg.bilstm_hidden,
            num_layers=cfg.bilstm_layers,
            dropout=cfg.bilstm_dropout,
        )

        fusion_dim = cfg.cnn_channels + (2 * cfg.bilstm_hidden)
        self.autoencoder = BottleneckAutoencoder(fusion_dim, latent_dim=cfg.latent_dim, dropout=cfg.clf_dropout)
        self.attn = SelfAttentionLatent(cfg.latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.clf_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.clf_dropout),
            nn.Linear(cfg.clf_hidden1, cfg.clf_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.clf_dropout),
            nn.Linear(cfg.clf_hidden2, 1),
        )

    def forward_branches(self, x_cnn, x_lstm):
        f_cnn = self.cnn(x_cnn)          # [B, C]
        f_lstm = self.lstm(x_lstm)       # [B, 2H]
        fused = torch.cat([f_cnn, f_lstm], dim=1)
        return fused

    def forward(self, x_cnn, x_lstm):
        fused = self.forward_branches(x_cnn, x_lstm)
        recon, z = self.autoencoder(fused)
        z_att, weights = self.attn(z)
        logits = self.classifier(z_att)
        return logits.squeeze(1), recon, z, weights

    def encode(self, x_cnn, x_lstm):
        fused = self.forward_branches(x_cnn, x_lstm)
        return self.autoencoder.encode(fused)
