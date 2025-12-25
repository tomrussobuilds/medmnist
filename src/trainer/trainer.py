"""
Model Trainer Module

This module defines the central ModelTrainer class which orchestrates the 
training lifecycle, bridging the gap between high-level configuration 
and low-level execution engines.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple, List
import logging
from pathlib import Path
from functools import partial

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config
from .engine import train_one_epoch, validate_epoch, mixup_data

# =========================================================================== #
#                                TRAINING LOGIC                               #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

class ModelTrainer:
    """
    Encapsulates the core training, validation, and scheduling logic.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        cfg: Config,
        output_dir: Path | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg

        # Hyperparameters
        self.epochs = cfg.training.epochs
        self.patience = cfg.training.patience
        self.best_acc = -1.0
        self.epochs_no_improve = 0

        # Modern AMP Support (PyTorch 2.x+)
        self.scaler = torch.amp.GradScaler(enabled=cfg.training.use_amp)

        # Mixup configuration
        self.mixup_fn = None
        if cfg.training.mixup_alpha > 0:
            self.mixup_fn = partial(
                mixup_data,
                alpha=cfg.training.mixup_alpha,
                device=device
            )
        
        # Output Management
        self.best_path = (output_dir or Path(".")) / "best_model.pth"
        self.best_path.parent.mkdir(parents=True, exist_ok=True)

        # History tracking
        self.train_losses: List[float] = []
        self.val_accuracies: List[float] = []

        logger.info(f"Trainer initialized. Best model checkpoint: {self.best_path.name}")
        
    def train(self) -> Tuple[Path, List[float], List[float]]:
        """
        Executes the main training loop with checkpointing and early stopping.
        """
        for epoch in range(1, self.epochs + 1):
            logger.info(f" Epoch {epoch:02d}/{self.epochs} ".center(60, "-"))

            # Logic: Apply MixUp only during the first portion of training if configured
            mixup_cutoff = int(self.cfg.training.cosine_fraction * self.epochs)
            current_mixup = self.mixup_fn if epoch <= mixup_cutoff else None
                
            # --- 1. Training Phase ---
            epoch_loss = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                mixup_fn=current_mixup,
                scaler=self.scaler,
                grad_clip=self.cfg.training.grad_clip
            )
            self.train_losses.append(epoch_loss)

            # --- 2. Validation Phase ---
            # validate_epoch returns a dict: {"loss": float, "accuracy": float}
            val_metrics = validate_epoch(
                model=self.model, 
                val_loader=self.val_loader, 
                criterion=self.criterion,
                device=self.device
            )
            val_acc = val_metrics["accuracy"]
            val_loss = val_metrics["loss"]
            self.val_accuracies.append(val_acc)

            # --- 3. Scheduling Phase ---
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau typically monitors validation loss
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                
            # --- 4. Checkpoint & Early Stopping Logic ---
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_path)
                logger.info(f"New best model! Val Acc: {val_acc:.4f} ↑ Checkpoint saved.")
            else:
                self.epochs_no_improve += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Loss: [T: {epoch_loss:.4f} | V: {val_loss:.4f}] | "
                f"Acc: {val_acc:.4f} (Best: {self.best_acc:.4f}) | "
                f"LR: {current_lr:.2e} | Patience: {self.patience - self.epochs_no_improve}"
            )

            if self.epochs_no_improve >= self.patience:
                logger.warning(f"Early stopping triggered at epoch {epoch}.")
                break
            
        logger.info(f"Training finished. Peak Validation Accuracy: {self.best_acc:.4f}")
        return self.best_path, self.train_losses, self.val_accuracies