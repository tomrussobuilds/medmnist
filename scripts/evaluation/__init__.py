"""
Evaluation and Reporting Package

This package contains modules for model evaluation, visualization, 
and structured reporting.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Tuple, List, Final
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Logger, Config, FIGURES_DIR
from scripts.data_handler.data_handler import BloodMNISTData
from .engine import evaluate_model
from .visualization import (
    plot_confusion_matrix, save_training_curves, save_sample_predictions
)
from .reporting import create_structured_report, TrainingReport

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


def run_final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    data: BloodMNISTData,
    train_losses: List[float],
    val_accuracies: List[float],
    device: torch.device,
    use_tta: bool = False,
    cfg: Config | None = None
) -> Tuple[float, float]:
    """
    Executes the full evaluation pipeline, generating all figures and metrics.
    """
    
    # --- 1) Evaluate Model Performance ---
    all_preds, all_labels, test_acc, macro_f1 = evaluate_model(
        model, test_loader, device, use_tta=use_tta
    )

    # --- 2) Confusion Matrix Figure ---
    plot_confusion_matrix(
        all_labels,
        all_preds,
        FIGURES_DIR / f"confusion_matrix_{cfg.model_name}.png",
        cfg=cfg
    )

    # --- 3) Training Curves Figure & Data ---
    save_training_curves(
        train_losses,
        val_accuracies,
        FIGURES_DIR,
        cfg=cfg
    )

    # --- 4) Sample Predictions Figure ---
    save_sample_predictions(
        data,
        all_preds,
        FIGURES_DIR / "sample_predictions.png",
        cfg=cfg
    )

    logger.info(f"Evaluation and reporting complete → Accuracy={test_acc:.4f}, Macro F1={macro_f1:.4f}")
    return macro_f1, test_acc