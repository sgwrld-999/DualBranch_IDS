from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

from src.models.dual_branch import DualBranchModel
from src.train.trainer import pretrain_autoencoder, train_classifier
from src.train.evaluator import evaluate
from src.data.dataset import DualBranchDataset


@dataclass
class TuneResult:
    best_params: Dict
    best_value: float


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         base_cfg, n_trials: int = 15) -> TuneResult:
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Install it or set cfg.use_optuna=False.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(trial: 'optuna.trial.Trial') -> float:
        # Suggest a few core hyperparams
        base_cfg.cnn_channels = trial.suggest_categorical('cnn_channels', [32, 64, 96])
        base_cfg.bilstm_hidden = trial.suggest_categorical('bilstm_hidden', [64, 96, 128, 192])
        base_cfg.latent_dim = trial.suggest_categorical('latent_dim', [32, 48, 64])
        base_cfg.clf_dropout = trial.suggest_float('dropout', 0.2, 0.5, step=0.1)

        # Datasets
        train_ds = DualBranchDataset(X_train, y_train)
        val_ds = DualBranchDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=base_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=base_cfg.batch_size, shuffle=False)

        model = DualBranchModel(in_features=X_train.shape[1], cfg=base_cfg).to(device)

        # Light training for speed
        pretrain_autoencoder(model, train_loader, val_loader, device, epochs=2,
                             logdir=base_cfg.tb_logdir, outdir=base_cfg.outputs_dir)

        # Train classifier for a few epochs and monitor val accuracy
        # Reuse train_classifier internals by creating a small test loader = val
        train_classifier(model, train_loader, val_loader, val_loader, device,
                         epochs_frozen=2, epochs_finetune=2,
                         logdir=base_cfg.tb_logdir, outdir=base_cfg.outputs_dir,
                         class_weight=None)

        # Evaluate on val set
        res = evaluate(model, val_loader, device)
        return res.f1  # maximize F1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return TuneResult(best_params=study.best_params, best_value=study.best_value)
