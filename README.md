# Dual-Branch CNN-BiLSTM-Autoencoder with Attention (PyTorch)

Binary IoT intrusion detection using a dual-branch architecture combining a 1D-CNN and BiLSTM, a bottleneck autoencoder, self-attention, and a dense classifier. Includes autoencoder pretraining, end-to-end fine-tuning, metrics, plots, and Optuna hooks.

## Data
Update the dataset path in `src/config.py` to your CSV (default points to Edge_IIoT_Processed_dataset.csv).

## Quickstart
1. Install deps
2. Run training (autoencoder pretrain + classifier fine-tune)
3. See results and TensorBoard logs in `runs/` and `outputs/`

## Structure
See `src/` for modules and `tests/` for smoke tests.
