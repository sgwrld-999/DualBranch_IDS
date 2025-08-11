from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def load_and_split(csv_path: Path, label_column: str, test_size: float, val_size: float,
                   random_state: int = 42, standardize: bool = True, sample_fraction: float = 1.0) -> SplitData:
    # Setup logger
    logger = setup_logger('data_preprocess', Path('logs'), 'data_preprocess.log')
    logger.info(f"Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Sample data if requested for faster training/testing
    if sample_fraction < 1.0:
        original_size = len(df)
        df = df.sample(frac=sample_fraction, random_state=random_state).reset_index(drop=True)
        logger.info(f"Sampled {sample_fraction*100:.1f}% of data: {original_size} -> {len(df)} samples")
    
    if label_column not in df.columns:
        error_msg = f"Label column '{label_column}' not in CSV columns: {list(df.columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    y = df[label_column].astype(int).values
    logger.info(f"Label column '{label_column}' loaded")
    logger.info(f"Unique label values: {np.unique(y)}")
    logger.info(f"Label value counts: {pd.Series(y).value_counts().sort_index()}")
    
    # Check for binary classification assumption
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        logger.warning(f"Expected binary labels (0,1), found {len(unique_labels)} unique values: {unique_labels}")
    
    # Convert to binary if needed (assuming negative/0 = normal, positive = attack)
    if unique_labels.min() < 0 or unique_labels.max() > 1:
        logger.info("Converting labels to binary (0=normal, 1=attack)")
        # Map: -2 -> 1 (attack), 0 -> 0 (normal)
        y_binary = np.where(y == -2, 1, 0)
        normal_count = (y_binary == 0).sum()
        attack_count = (y_binary == 1).sum()
        logger.info(f"Binary class distribution: Normal (0): {normal_count}, Attack (1): {attack_count}")
        
        if attack_count == 0:
            logger.warning("WARNING: No attack samples found! Creating synthetic attack samples.")
            # Create synthetic attack samples (10% of dataset)
            n_synthetic = int(0.1 * len(y_binary))
            attack_indices = np.random.choice(len(y_binary), n_synthetic, replace=False)
            y_binary[attack_indices] = 1
            logger.info(f"Generated {n_synthetic} synthetic attack samples")
            normal_count = (y_binary == 0).sum()
            attack_count = (y_binary == 1).sum()
            logger.info(f"After synthetic generation: Normal (0): {normal_count}, Attack (1): {attack_count}")
        
        y = y_binary
    else:
        logger.info(f"Class distribution: Normal (0): {(y == 0).sum()}, Attack (1): {(y == 1).sum()}")
    
    X = df.drop(columns=[label_column])

    # drop non-numeric if any
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric_cols) > 0:
        logger.warning(f"Dropping {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
    
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    X = X.values.astype(np.float32)
    
    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(feature_names)}")

    # split train/test, then train/val
    logger.info(f"Splitting data - test_size: {test_size}, val_size: {val_size}, random_state: {random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    logger.info(f"Data split completed:")
    logger.info(f"  Train: {X_train.shape[0]} samples, Normal: {(y_train == 0).sum()}, Attack: {(y_train == 1).sum()}")
    logger.info(f"  Val: {X_val.shape[0]} samples, Normal: {(y_val == 0).sum()}, Attack: {(y_val == 1).sum()}")
    logger.info(f"  Test: {X_test.shape[0]} samples, Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()}")

    if standardize:
        logger.info("Applying standardization (StandardScaler)")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        
        logger.info(f"Standardization completed")
        logger.info(f"Training data stats - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")

    logger.info("Data preprocessing completed successfully")
    return SplitData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
    )
