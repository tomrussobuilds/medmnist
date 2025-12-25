"""
Optimization Factories Module

This module provides factory functions to instantiate PyTorch optimization 
components (optimizers, schedulers, and loss functions) based on the 
hierarchical configuration manifest.
"""

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                                  FACTORIES                                  #
# =========================================================================== #

def get_criterion(cfg: Config) -> nn.Module:
    """
    Returns the appropriate loss function for the classification task.
    
    Standardizes on CrossEntropyLoss for MedMNIST multi-class objectives.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(model: nn.Module, cfg: Config) -> optim.Optimizer:
    """
    Factory function to instantiate a task-specific optimizer.
    
    Decision Logic:
        - ResNet Variants: Uses SGD with Momentum for better generalization 
          and flatter minima in convolutional landscapes.
        - Other (ViT/Transformers): Defaults to AdamW to handle decoupled 
          weight decay and adaptive learning rates.
    """
    model_name = cfg.model_name.lower()
    
    if "resnet" in model_name:
        return optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay
        )
    
    # Robust default for modern attention-based or hybrid architectures
    return optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )


def get_scheduler(optimizer: optim.Optimizer, cfg: Config) -> lr_scheduler._LRScheduler:
    """
    Factory function to configure the learning rate trajectory.
    
    Implements a Cosine Annealing schedule that decays the learning rate 
    from the initial value down to the 'min_lr' specified in the configuration, 
    spanning the total duration of the training epochs.
    """
    return lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.training.min_lr
    )