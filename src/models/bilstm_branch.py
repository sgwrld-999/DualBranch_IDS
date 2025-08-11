from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMBranch(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, (h_n, c_n) = self.lstm(x)
        # Use last time step output or pooled
        # Here, concatenate last forward and last backward hidden states
        h_forward = h_n[-2, :, :]  # [B, H]
        h_backward = h_n[-1, :, :] # [B, H]
        h = torch.cat([h_forward, h_backward], dim=1)  # [B, 2H]
        return self.dropout(h)
