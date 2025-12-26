"""
Visualization Utilities Module

This module provides functions for generating visual reports of the model's 
performance, including training loss/accuracy curves, normalized confusion 
matrices, and sample prediction grids. It is fully integrated with the 
Pydantic Configuration Engine for aesthetic and technical consistency.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Sequence, List
from pathlib import Path
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# Global logger instance
logger = logging.getLogger("medmnist_pipeline")


# =========================================================================== #
#                               VISUALIZATION FUNCTIONS                       #
# =========================================================================== #

def show_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: List[str],
    save_path: Path | None = None,
    cfg: Config | None = None,
    n: int | None = None
) -> None:
    """
    Lazy-extracts a batch from the loader and displays model predictions.
    
    Highlights correct (green) vs. incorrect (red) predictions in a grid.
    Handles denormalization using dataset stats from Config and handles
    C,H,W to H,W,C transposition for Matplotlib.

    Args:
        model (nn.Module): Trained model for inference.
        loader (DataLoader): Data-loader to sample from (usually Test/Val).
        device (torch.device): Computation device.
        classes (List[str]): Names of the target classes.
        save_path (Path | None): Output destination for the figure.
        cfg (Config | None): SSOT for grid layout, DPI, and normalization stats.
        n (int | None): Number of samples. If None, uses cfg.evaluation.n_samples.
    """
    model.eval()

    # 1. Configuration Resolution
    # Priority: Function Argument -> Config Pydantic -> Hardcoded Default
    num_samples = n or (cfg.evaluation.n_samples if cfg else 12)
    dpi = cfg.evaluation.fig_dpi if cfg else 200
    grid_cols = cfg.evaluation.grid_cols if cfg else 4
    
    # 2. Lazy Extraction & Inference
    batch = next(iter(loader))
    images_tensor, labels_tensor = batch[0], batch[1]

    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        outputs = model(images_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()
    
    images_batch = images_tensor.cpu().numpy()
    true_labels = labels_tensor.cpu().numpy().flatten()
    
    # 3. Grid Setup
    num_samples = min(num_samples, len(images_batch))
    rows = int(np.ceil(num_samples / grid_cols))
    
    # Dynamic figsize based on rows
    base_w, base_h = cfg.evaluation.fig_size_predictions if cfg else (12, 8)
    plt.figure(figsize=(base_w, (base_h / 3) * rows))

    for i in range(num_samples):
        img = images_batch[i].copy()

        # Denormalization Logic using SSOT (Config)
        if cfg:
            mean = np.array(cfg.dataset.mean).reshape(-1, 1, 1)
            std = np.array(cfg.dataset.std).reshape(-1, 1, 1)
            img = (img * std) + mean
            img = np.clip(img, 0, 1)

        # Transpose for Matplotlib: (C, H, W) -> (H, W, C)
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))

        plt.subplot(rows, grid_cols, i + 1)
        
        # Support for grayscale (MedMNIST 2D) or RGB (MedMNIST 3D/Color)
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
            
        color = "green" if true_labels[i] == preds[i] else "red"
        plt.title(f"T:{classes[true_labels[i]]}\nP:{classes[preds[i]]}", 
                  color=color, fontsize=9)
        plt.axis("off")
    
    model_name = cfg.model_name if cfg else "Model"
    plt.suptitle(f"Sample Predictions Grid — {model_name}", fontsize=14)
    plt.tight_layout()

    # 4. Save Logic
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"Predictions grid saved → {save_path.name} (DPI: {dpi})")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
        train_losses: Sequence[float],
        val_accuracies: Sequence[float],
        out_path: Path,
        cfg: Config
) -> None:
    """
    Plots training loss and validation accuracy curves on a dual-axis plot.
    Automatically saves raw numerical data to .npz for reproducibility.

    Args:
        train_losses (Sequence[float]): History of training losses.
        val_accuracies (Sequence[float]): History of validation accuracies.
        out_path (Path): Path to save the plot.
        cfg (Config): SSOT for DPI and model metadata.
    """
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Left Axis: Training Loss
    ax1.plot(train_losses, color='#e74c3c', lw=2, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='#e74c3c', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Right Axis: Validation Accuracy
    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, color='#3498db', lw=2, label="Validation Accuracy")
    ax2.set_ylabel("Accuracy", color='#3498db', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#3498db')

    plt.title(f"Training Metrics — {cfg.model_name}", fontsize=14, pad=15)
    fig.tight_layout()
    
    # Save the figure using Config DPI
    plt.savefig(out_path, dpi=cfg.evaluation.fig_dpi, bbox_inches="tight")
    logger.info(f"Training curves saved → {out_path.name}")

    # Export raw data for post-run analysis
    npz_path = out_path.with_suffix('.npz')
    np.savez(
        npz_path,
        train_losses=train_losses,
        val_accuracies=val_accuracies
    )
    logger.debug(f"Raw history data saved to {npz_path.name}")

    plt.close()


def plot_confusion_matrix(
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        classes: List[str],
        out_path: Path,
        cfg: Config
) -> None:
    """
    Generates and saves a normalized confusion matrix plot.

    Args:
        all_labels (np.ndarray): True ground truth labels.
        all_preds (np.ndarray): Predicted labels from model.
        classes (List[str]): List of class names.
        out_path (Path): Destination file path.
        cfg (Config): SSOT for Colormap and DPI settings.
    """
    # Normalized Confusion Matrix (Rows sum to 1)
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=np.arange(len(classes)),
        normalize='true',
    )
    cm = np.nan_to_num(cm) # Handle potential empty classes

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Use Dynamic ColorMap from Config
    disp.plot(
        ax=ax,
        cmap=cfg.evaluation.cmap_confusion,
        xticks_rotation=45,
        values_format='.3f'
    )

    plt.title(f"Confusion Matrix — {cfg.model_name}", fontsize=14, pad=20)
    plt.tight_layout()
    
    fig.savefig(out_path,
                dpi=cfg.evaluation.fig_dpi,
                bbox_inches="tight"
                )
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path.name}")