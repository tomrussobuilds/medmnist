"""
Data Transformations Module

This module defines the image augmentation pipelines for training and 
the standard normalization for validation/testing. It also includes 
utilities for deterministic worker initialization. It supports both RGB
and Grayscale datasets dynamically.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import random
from typing import Tuple, Final

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torchvision.transforms import v2

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config


# =========================================================================== #
#                             TRANSFORMATION PIPELINES                        #
# =========================================================================== #
# Standard constants
IMG_SIZE: Final[int] = 28

# Normalization values for ImageNet (RGB)
RGB_MEAN: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
RGB_STD: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)

# Grayscale normalization values
GRAY_MEAN: Final[Tuple[float]] = (0.5,)
GRAY_STD: Final[Tuple[float]] = (0.5,)


def get_augmentations_description(cfg: Config) -> str:
    """
    Generates a descriptive string of the augmentations using values from Config.
    Used for logging and run traceability.
    """ 
    params = {
        "HFlip": cfg.augmentation.hflip,
        "Rotation": f"{cfg.augmentation.rotation_angle}°",
        "Jitter": cfg.augmentation.jitter_val,
        "ResizedCrop": f"{IMG_SIZE} (0.9, 1.0)"
    }

    descr = [f"{k}({v})" for k, v in params.items()]
    
    if cfg.training.mixup_alpha > 0:
        descr.append(f"MixUp(α={cfg.training.mixup_alpha})")
    
    return ", ".join(descr)


def worker_init_fn(worker_id: int):
    """
    Initializes random number generators (PRNGs) for each DataLoader worker.
    Crucial for maintaining augmentation diversity and reproducibility 
    when using multiple workers for lazy-loading.
    """
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed if worker_info else torch.initial_seed()
    seed = (base_seed + worker_id) % 2**32

    np.random.seed(seed)
    random.seed(seed) 
    torch.manual_seed(seed)

def get_pipeline_transforms(
        cfg: Config,
        is_rgb: bool = True
    ) -> Tuple[v2.Compose, v2.Compose]:
    """
    Defines the transformation pipelines for training and evaluation.
    """
    stats = {
        True:  {"mean": RGB_MEAN, "std": RGB_STD},
        False: {"mean": GRAY_MEAN, "std": GRAY_STD} 
    }[is_rgb]

    def get_base_ops():
        return [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        
    # Training pipeline: Focus on robust generalization
    train_transform = v2.Compose([
        *get_base_ops(),
        v2.RandomHorizontalFlip(p=cfg.augmentation.hflip),
        v2.RandomRotation(cfg.augmentation.rotation_angle),
        v2.ColorJitter(
            brightness=cfg.augmentation.jitter_val,
            contrast=cfg.augmentation.jitter_val,
            saturation=cfg.augmentation.jitter_val if is_rgb else 0.0,
        ),
        v2.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.9, 1.0),
            antialias=True,
            interpolation=v2.InterpolationMode.BILINEAR,
        ),
        v2.Normalize(**stats),
    ])
    
    # Validation/Inference pipeline: Strict consistency
    val_transform = v2.Compose([
        *get_base_ops(),
        v2.Normalize(**stats),
    ])
    
    return train_transform, val_transform