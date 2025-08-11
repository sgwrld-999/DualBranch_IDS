from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import cfg
from src.data.dataset import DualBranchDataset
from src.models.dual_branch import DualBranchModel


def test_forward_pass():
    B, F = 8, 50
    X = np.random.randn(B, F).astype(np.float32)
    y = np.random.randint(0, 2, size=(B,)).astype(np.int64)

    ds = DualBranchDataset(X, y)
    dl = DataLoader(ds, batch_size=4)
    model = DualBranchModel(in_features=F, cfg=cfg)

    for batch in dl:
        logits, recon, z, weights = model(batch["x_cnn"], batch["x_lstm"])
        assert logits.shape[0] == 4
        assert recon.shape[0] == 4
        assert z.shape[0] == 4
        assert weights.shape[0] == 4
        break
