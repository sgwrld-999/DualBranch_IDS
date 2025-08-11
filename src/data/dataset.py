from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict


class DualBranchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_len: int = 1):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.sequence_len = max(1, int(sequence_len))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.X[idx]
        # CNN branch expects [C=1, F]
        x_cnn = torch.from_numpy(x[None, :])  # [1, F]
        # LSTM expects [T, F]; if no temporal axis, use T=1
        x_lstm = torch.from_numpy(x[None, :])  # [1, F]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return {"x_cnn": x_cnn, "x_lstm": x_lstm, "y": y}
