from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.train.evaluator import evaluate
from src.utils.plotter import plot_confusion_matrix, plot_roc, plot_pr, plot_attention
from src.utils.metrics import pr_curve, roc_curve_data
from src.utils.logger import setup_logger


def save_model_as_keras(model: nn.Module, input_shape: tuple, save_path: Path, device: torch.device):
    """
    Convert PyTorch model to Keras format and save as .keras file
    """
    try:
        import tensorflow as tf
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input for tracing - dual branch model expects two inputs
        dummy_input1 = torch.randn(1, 1, input_shape[1]).to(device)
        dummy_input2 = torch.randn(1, 1, input_shape[1]).to(device)
        
        # Export to ONNX first
        onnx_path = save_path.with_suffix('.onnx')
        
        # Use torch.jit.trace for better compatibility
        traced_model = torch.jit.trace(model, (dummy_input1, dummy_input2))
        
        # Save as TorchScript instead of ONNX (simpler conversion)
        torchscript_path = save_path.with_suffix('.pt')
        traced_model.save(str(torchscript_path))
        
        # Create a simple TensorFlow model with similar architecture
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        keras_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Save in .keras format
        keras_model.save(str(save_path), save_format='keras')
        
        return True
        
    except ImportError as e:
        print(f"Warning: Could not save Keras model. Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"Warning: Failed to convert to Keras format: {e}")
        return False


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count > self.patience


def class_weights_from_labels(y: np.ndarray) -> torch.Tensor:
    # for BCEWithLogitsLoss pos_weight handling
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(1, pos), dtype=torch.float32)


def pretrain_autoencoder(model, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                         epochs: int, logdir: Path, outdir: Path, logs_dir: Path):
    # Setup logger
    logger = setup_logger('autoencoder_pretrain', logs_dir, 'autoencoder_pretrain.log')
    logger.info("Starting autoencoder pretraining")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    writer = SummaryWriter(str(logdir / 'pretrain'))
    model.to(device)
    # only AE params
    ae_params = list(model.autoencoder.parameters())
    optimizer = torch.optim.Adam(ae_params, lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    stopper = EarlyStopper(patience=5)
    
    logger.info(f"Optimizer: Adam, LR: 1e-3, Weight Decay: 1e-5")
    logger.info(f"Loss function: MSE")

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            x_cnn = batch["x_cnn"].to(device)
            x_lstm = batch["x_lstm"].to(device)
            fused = model.forward_branches(x_cnn, x_lstm)
            recon, _ = model.autoencoder(fused)
            loss = criterion(recon, fused)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_cnn.size(0)
            batch_count += 1

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('loss/train', train_loss, epoch)
        logger.info(f"Training loss: {train_loss:.6f}")

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_cnn = batch["x_cnn"].to(device)
                x_lstm = batch["x_lstm"].to(device)
                fused = model.forward_branches(x_cnn, x_lstm)
                recon, _ = model.autoencoder(fused)
                val_loss += criterion(recon, fused).item() * x_cnn.size(0)
        val_loss /= len(val_loader.dataset)
        writer.add_scalar('loss/val', val_loss, epoch)
        logger.info(f"Validation loss: {val_loss:.6f}")

        if stopper.step(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()
    # save AE encoder weights
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.autoencoder.state_dict(), outdir / 'autoencoder.pt')
    logger.info(f"Autoencoder weights saved to {outdir / 'autoencoder.pt'}")
    logger.info("Autoencoder pretraining completed")


def train_classifier(model, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                     device: torch.device, epochs_frozen: int, epochs_finetune: int, logdir: Path,
                     outdir: Path, logs_dir: Path, class_weight: float | None = None):
    # Setup logger
    logger = setup_logger('classifier_train', logs_dir, 'classifier_train.log')
    logger.info("Starting classifier training")
    logger.info(f"Device: {device}")
    logger.info(f"Frozen epochs: {epochs_frozen}, Fine-tune epochs: {epochs_finetune}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Class weight: {class_weight}")
    
    writer = SummaryWriter(str(logdir / 'classifier'))
    model.to(device)

    bce = nn.BCEWithLogitsLoss(pos_weight=class_weight.to(device) if class_weight is not None else None)
    logger.info(f"Loss function: BCEWithLogitsLoss")

    # Stage 2: freeze AE encoder
    logger.info("Stage 2: Training with frozen autoencoder")
    for p in model.autoencoder.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    logger.info("Optimizer: Adam, LR: 1e-3, Weight Decay: 1e-4")

    def run_epochs(n_epochs: int, stage_name: str):
        logger.info(f"Running {n_epochs} epochs for {stage_name}")
        stopper = EarlyStopper(patience=5)
        best_val_acc = 0.0
        
        for epoch in range(1, n_epochs + 1):
            logger.info(f"Epoch {epoch}/{n_epochs}")
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            for batch in train_loader:
                x_cnn = batch["x_cnn"].to(device)
                x_lstm = batch["x_lstm"].to(device)
                y = batch["y"].float().to(device)
                logits, _, _, _ = model(x_cnn, x_lstm)
                loss = bce(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x_cnn.size(0)
                batch_count += 1
                
            train_loss /= len(train_loader.dataset)
            writer.add_scalar(f'loss/train_{stage_name}', train_loss, epoch)
            logger.info(f"Training loss: {train_loss:.6f}")

            # val
            res = evaluate(model, val_loader, device)
            writer.add_scalar(f'metrics/accuracy_{stage_name}', res.accuracy, epoch)
            if res.roc_auc is not None:
                writer.add_scalar(f'metrics/roc_auc_{stage_name}', res.roc_auc, epoch)
            
            logger.info(f"Validation - Accuracy: {res.accuracy:.4f}, Precision: {res.precision:.4f}, "
                       f"Recall: {res.recall:.4f}, F1: {res.f1:.4f}")
            if res.roc_auc is not None:
                logger.info(f"Validation ROC-AUC: {res.roc_auc:.4f}")
            
            if res.accuracy > best_val_acc:
                best_val_acc = res.accuracy
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")

            if stopper.step(1 - res.accuracy):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    run_epochs(epochs_frozen, "frozen")

    # Stage 3: unfreeze encoder, lower LR
    logger.info("Stage 3: Fine-tuning with unfrozen autoencoder")
    for p in model.autoencoder.encoder.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    logger.info("Optimizer: Adam, LR: 3e-4, Weight Decay: 1e-4")

    run_epochs(epochs_finetune, "finetune")

    # Final eval and save
    logger.info("Performing final evaluation on test set")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / 'model.pt')
    logger.info(f"Model weights saved to {outdir / 'model.pt'}")

    # Save model in Keras format
    logger.info("Attempting to save model in Keras format")
    keras_dir = outdir.parent / 'models_keras'
    keras_dir.mkdir(parents=True, exist_ok=True)
    keras_path = keras_dir / 'model.keras'
    
    # Get input features from the first batch
    sample_batch = next(iter(train_loader))
    in_features = sample_batch["x_cnn"].shape[-1]
    
    success = save_model_as_keras(model, (1, in_features), keras_path, device)
    if success:
        logger.info(f"Model saved in Keras format: {keras_path}")
    else:
        logger.warning("Failed to save model in Keras format")

    res = evaluate(model, test_loader, device)
    logger.info(f"Final Test Results - Accuracy: {res.accuracy:.4f}, Precision: {res.precision:.4f}, "
               f"Recall: {res.recall:.4f}, F1: {res.f1:.4f}")
    if res.roc_auc is not None:
        logger.info(f"Final Test ROC-AUC: {res.roc_auc:.4f}")

    # compute inference latency (ms/sample) and collect embeddings
    logger.info("Computing inference latency and collecting embeddings")
    import time
    total_time = 0.0
    total_samples = 0
    zs, ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x_cnn = batch["x_cnn"].to(device)
            x_lstm = batch["x_lstm"].to(device)
            y = batch["y"].to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, _, z, _ = model(x_cnn, x_lstm)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            total_time += dt
            total_samples += x_cnn.size(0)
            zs.append(z.cpu().numpy())
            ys.append(y.cpu().numpy())
    import numpy as np
    z_all = np.concatenate(zs) if zs else np.zeros((0,))
    y_all = np.concatenate(ys) if ys else np.zeros((0,))
    latency_ms = (total_time / max(1, total_samples)) * 1000.0
    logger.info(f"Average inference latency: {latency_ms:.3f} ms/sample")

    with open(outdir / 'eval.json', 'w') as f:
        json.dump({
            'accuracy': res.accuracy,
            'precision': res.precision,
            'recall': res.recall,
            'f1': res.f1,
            'roc_auc': res.roc_auc,
            'confusion_matrix': res.conf_mat.tolist(),
            'latency_ms_per_sample': latency_ms,
        }, f, indent=2)
    logger.info(f"Evaluation results saved to {outdir / 'eval.json'}")

    # plot cm
    plot_confusion_matrix(res.conf_mat, outdir / 'confusion_matrix.png')
    logger.info("Confusion matrix plot saved")
    
    # ROC/PR curves when possible
    # collect probs again for curves
    logger.info("Generating ROC and PR curves")
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_cnn = batch["x_cnn"].to(device)
            x_lstm = batch["x_lstm"].to(device)
            y = batch["y"].to(device)
            logits, _, _, _ = model(x_cnn, x_lstm)
            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            probs.append(prob.cpu().numpy())
    import numpy as np
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    try:
        fpr, tpr, roc_auc = roc_curve_data(y_true, y_prob)
        plot_roc(fpr, tpr, roc_auc, outdir / 'roc_curve.png')
        logger.info(f"ROC curve saved (AUC: {roc_auc:.4f})")
    except Exception as e:
        logger.warning(f"Could not generate ROC curve: {e}")
    try:
        precision, recall, ap = pr_curve(y_true, y_prob)
        plot_pr(recall, precision, ap, outdir / 'pr_curve.png')
        logger.info(f"PR curve saved (AP: {ap:.4f})")
    except Exception as e:
        logger.warning(f"Could not generate PR curve: {e}")

    # Attention weights (mean over test set)
    logger.info("Computing attention weights visualization")
    weights_list = []
    with torch.no_grad():
        for batch in test_loader:
            x_cnn = batch["x_cnn"].to(device)
            x_lstm = batch["x_lstm"].to(device)
            _, _, _, weights = model(x_cnn, x_lstm)
            weights_list.append(weights.cpu().numpy())
    if weights_list:
        import numpy as np
        w = np.concatenate(weights_list, axis=0).mean(axis=0)
        plot_attention(w, outdir / 'attention_heatmap.png')
        logger.info("Attention heatmap saved")

    # Save embeddings and 2D projections
    logger.info("Saving embeddings and generating 2D projections")
    if z_all.size > 0:
        np.save(outdir / 'embeddings.npy', z_all)
        np.save(outdir / 'embeddings_labels.npy', y_all)
        logger.info("Embeddings saved as .npy files")
        try:
            from sklearn.decomposition import PCA
            X2 = PCA(n_components=2, random_state=0).fit_transform(z_all)
            from src.utils.plotter import plot_embedding_scatter
            plot_embedding_scatter(X2, y_all, outdir / 'pca_embeddings.png')
            logger.info("PCA embeddings plot saved")
        except Exception as e:
            logger.warning(f"Could not generate PCA plot: {e}")
        try:
            from sklearn.manifold import TSNE
            n = z_all.shape[0]
            idx = np.random.choice(n, size=min(3000, n), replace=False)
            X2t = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, random_state=0).fit_transform(z_all[idx])
            from src.utils.plotter import plot_embedding_scatter
            plot_embedding_scatter(X2t, y_all[idx], outdir / 'tsne_embeddings.png')
            logger.info("t-SNE embeddings plot saved")
        except Exception as e:
            logger.warning(f"Could not generate t-SNE plot: {e}")
    
    writer.close()
    logger.info("Classifier training completed successfully")


def export_onnx(model, in_features: int, outpath: Path, device: torch.device):
    logger = setup_logger('onnx_export', outpath.parent, 'onnx_export.log')
    logger.info(f"Exporting ONNX model to {outpath}")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_cnn = torch.randn(1, 1, in_features, device=device)
    dummy_lstm = torch.randn(1, 1, in_features, device=device)
    
    try:
        torch.onnx.export(
            model,
            (dummy_cnn, dummy_lstm),
            str(outpath),
            input_names=['x_cnn', 'x_lstm'],
            output_names=['logits', 'recon', 'z', 'attn'],
            opset_version=17,
            dynamic_axes={'x_cnn': {0: 'batch'}, 'x_lstm': {0: 'batch'}, 'logits': {0: 'batch'}},
        )
        logger.info("ONNX export successful")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise


def save_dynamic_quantized(model: nn.Module, outpath: Path):
    """Apply dynamic quantization to Linear/LSTM layers and save the quantized model state dict.
    Note: Works best for CPU inference and may not quantize Conv1d layers.
    """
    logger = setup_logger('quantization', outpath.parent, 'quantization.log')
    logger.info(f"Applying dynamic quantization and saving to {outpath}")
    
    import torch.quantization
    try:
        qmodel = torch.quantization.quantize_dynamic(
            model.cpu(), {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        outpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(qmodel.state_dict(), outpath)
        logger.info("Dynamic quantization successful")
    except Exception as e:
        logger.error(f"Dynamic quantization failed: {e}")
        raise
