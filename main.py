from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import cfg
from src.data.preprocess import load_and_split
from src.data.dataset import DualBranchDataset
from src.models.dual_branch import DualBranchModel
from src.train.trainer import pretrain_autoencoder, train_classifier, class_weights_from_labels, export_onnx, save_dynamic_quantized, save_model_as_keras
from src.utils.logger import setup_logger


def seed_everything(seed: int = 42):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=str(cfg.dataset_csv), help='Path to dataset CSV')
    p.add_argument('--epochs-pretrain', type=int, default=cfg.epochs_pretrain)
    p.add_argument('--epochs-frozen', type=int, default=cfg.epochs_frozen)
    p.add_argument('--epochs-finetune', type=int, default=cfg.epochs_finetune)
    p.add_argument('--batch-size', type=int, default=cfg.batch_size)
    p.add_argument('--latent-dim', type=int, default=cfg.latent_dim)
    p.add_argument('--sample-fraction', type=float, default=cfg.sample_fraction, help='Fraction of data to use (0.1 = 10%)')
    return p.parse_args()


def run():
    args = parse_args()
    seed_everything(cfg.random_state)

    # Setup main logger
    logs_dir = cfg.logs_dir
    logger = setup_logger('main', logs_dir, 'main.log')
    logger.info("Starting Dual-Branch CNN-BiLSTM-Autoencoder training")
    logger.info(f"Arguments: {vars(args)}")

    # allow override
    from src.config import cfg as _cfg
    _cfg.dataset_csv = Path(args.csv)
    _cfg.epochs_pretrain = args.epochs_pretrain
    _cfg.epochs_frozen = args.epochs_frozen
    _cfg.epochs_finetune = args.epochs_finetune
    _cfg.batch_size = args.batch_size
    _cfg.latent_dim = args.latent_dim
    _cfg.sample_fraction = args.sample_fraction

    logger.info(f"Dataset: {_cfg.dataset_csv}")
    logger.info(f"Epochs - Pretrain: {_cfg.epochs_pretrain}, Frozen: {_cfg.epochs_frozen}, Finetune: {_cfg.epochs_finetune}")
    logger.info(f"Batch size: {_cfg.batch_size}, Latent dim: {_cfg.latent_dim}")

    # Load and split data
    logger.info("Loading and preprocessing data")
    data = load_and_split(_cfg.dataset_csv, _cfg.label_column, _cfg.test_size, _cfg.val_size,
                          cfg.random_state, cfg.standardize, _cfg.sample_fraction)

    in_features = data.X_train.shape[1]
    logger.info(f"Dataset loaded - Features: {in_features}")
    logger.info(f"Train: {data.X_train.shape[0]}, Val: {data.X_val.shape[0]}, Test: {data.X_test.shape[0]}")
    logger.info(f"Class distribution in training - Normal: {(data.y_train == 0).sum()}, Attack: {(data.y_train == 1).sum()}")

    train_ds = DualBranchDataset(data.X_train, data.y_train, sequence_len=_cfg.sequence_len)
    val_ds = DualBranchDataset(data.X_val, data.y_val, sequence_len=_cfg.sequence_len)
    test_ds = DualBranchDataset(data.X_test, data.y_test, sequence_len=_cfg.sequence_len)

    # Use single-threaded loading to avoid multiprocessing issues
    train_loader = DataLoader(train_ds, batch_size=_cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=_cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=_cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    logger.info("Data loaders created")

    model = DualBranchModel(in_features, _cfg)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    outdir = _cfg.outputs_dir
    logdir = _cfg.tb_logdir
    logs_dir = _cfg.logs_dir

    # Stage 1: AE pretraining
    logger.info("Starting Stage 1: Autoencoder pretraining")
    pretrain_autoencoder(model, train_loader, val_loader, device, _cfg.epochs_pretrain, logdir, outdir, logs_dir)

    # Stage 2+3: Classifier
    logger.info("Starting Stage 2&3: Classifier training")
    pos_weight = class_weights_from_labels(data.y_train) if _cfg.class_weight else None
    if pos_weight is not None:
        logger.info(f"Using class weight (pos_weight): {pos_weight.item():.4f}")
    
    train_classifier(model, train_loader, val_loader, test_loader, device,
                     _cfg.epochs_frozen, _cfg.epochs_finetune, logdir, outdir, logs_dir, pos_weight)

    # Export models
    logger.info("Exporting models")
    try:
        export_onnx(model.to(device), in_features, outdir / 'model.onnx', device)
        logger.info("ONNX model exported")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
    
    try:
        save_dynamic_quantized(model, outdir / 'model_int8_dyn.pt')
        logger.info("Quantized model saved")
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")

    # Save final model in Keras format
    try:
        keras_dir = outdir.parent / 'models_keras'
        keras_dir.mkdir(parents=True, exist_ok=True)
        keras_path = keras_dir / 'final_model.keras'
        success = save_model_as_keras(model, (1, in_features), keras_path, device)
        if success:
            logger.info(f"Final model saved in Keras format: {keras_path}")
        else:
            logger.warning("Failed to save final model in Keras format")
    except Exception as e:
        logger.warning(f"Keras export failed: {e}")

    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results saved in: {outdir}")
    logger.info(f"Logs saved in: {logs_dir}")
    logger.info(f"TensorBoard logs in: {logdir}")
    logger.info("Run 'tensorboard --logdir runs/dual_branch' to view training progress")


if __name__ == '__main__':
    run()
