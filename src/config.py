from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    dataset_csv: Path = Path("/fab3/btech/2022/siddhant.gond22b@iiitg.ac.in/semester_7/Edge_IIoT_Processed_dataset.csv")
    outputs_dir: Path = Path("outputs")
    tb_logdir: Path = Path("runs/dual_branch")
    logs_dir: Path = Path("logs")

    # Data
    label_column: str = "Attack_label"  # 0/1
    sequence_len: int = 1         # if >1, we will create rolling sequences for LSTM
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    standardize: bool = True
    sample_fraction: float = 0.1  # Use only 10% of data for faster training

    # Model (Reduced for memory efficiency)
    cnn_channels: int = 32
    cnn_kernel_sizes: tuple = (3, 3)
    cnn_dropout: float = 0.25

    bilstm_hidden: int = 64
    bilstm_layers: int = 1
    bilstm_dropout: float = 0.3

    latent_dim: int = 32

    clf_hidden1: int = 32
    clf_hidden2: int = 16
    clf_dropout: float = 0.3

    # Train (Optimized for large dataset)
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50  # Total training epochs
    epochs_pretrain: int = 3
    epochs_frozen: int = 3
    epochs_finetune: int = 5
    early_stop_patience: int = 3
    class_weight: bool = True

    # Optuna
    use_optuna: bool = False
    n_trials: int = 15


cfg = Config()
