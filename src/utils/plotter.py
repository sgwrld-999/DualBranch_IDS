from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc_val: float, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_pr(recall: np.ndarray, precision: np.ndarray, auc_val: float, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.plot(recall, precision, label=f"AP={auc_val:.3f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_attention(weights: np.ndarray, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 2))
    sns.heatmap(weights[None, :], cmap='viridis', cbar=True)
    plt.xlabel('Latent dimensions')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_embedding_scatter(x2: np.ndarray, y: np.ndarray, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x=x2[:, 0], y=x2[:, 1], hue=y.astype(int), palette='Set1', s=10, linewidth=0)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
